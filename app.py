#!/usr/bin/env python3
"""
Video Spelling QC — Web App
Upload multiple videos in the browser, they get processed in the background
(frame extraction -> OCR -> spellcheck), and results show up as a flagged
report per video, with thumbnails.
"""

import base64
import csv
import difflib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import pytesseract
from flask import Flask, render_template, request, jsonify, send_from_directory
from groq import Groq
from spellchecker import SpellChecker
from werkzeug.utils import secure_filename

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
HISTORY_FILE = BASE_DIR / "history.json"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
HISTORY_LOCK = threading.Lock()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
WORD_RE = re.compile(r"[A-Za-z]{3,}")
DEFAULT_FPS = 1.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB per request

# In-memory job store: job_id -> {status, videos: {filename: {...}}}
JOBS = {}
JOBS_LOCK = threading.Lock()

spell = SpellChecker()


def format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_frames(video_path: Path, out_dir: Path, fps: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-qscale:v", "3", pattern]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [(f, i / fps) for i, f in enumerate(frames)]


def ocr_frame_groq(frame_path: Path, retries: int = 3) -> str:
    """Send the frame to Groq's vision model -- reads text the way a human
    would, so it tends to silently 'auto-correct' obvious typos. Good for
    clean context, bad for catching the typos themselves."""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set — add it in Railway's Variables tab.")

    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Transcribe ALL visible text in this image exactly as it appears, "
        "including any typos or misspellings -- do NOT correct them. "
        "If there is no readable text, respond with exactly: NONE. "
        "Return only the transcribed text, nothing else."
    )

    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_completion_tokens=512,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            return "" if text.upper() == "NONE" else text
        except Exception as e:
            if attempt == retries - 1:
                print(f"Groq OCR failed on {frame_path.name}: {e}")
                return ""
            time.sleep(2 ** attempt)  # backoff on rate limits
    return ""


def ocr_frame_tesseract(frame_path: Path) -> str:
    """Literal, pixel-level OCR -- doesn't 'understand' the text, so typos
    pass through untouched. Noisier than Groq but catches what Groq smooths over."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    try:
        return pytesseract.image_to_string(gray).strip()
    except Exception:
        return ""


def dedupe_readings(readings, similarity_threshold=0.82):
    """
    Collapses consecutive frames with SIMILAR (not just identical) clean text
    into one event, carrying along the literal (Tesseract) reading too.
    readings: list of (ts, groq_text, tess_text, frame_path)
    """
    events = []
    current = None

    def norm(t):
        return re.sub(r"\s+", " ", t.lower()).strip()

    for ts, groq_text, tess_text, frame in readings:
        if not groq_text.strip():
            if current:
                events.append(current)
                current = None
            continue

        if current:
            sim = difflib.SequenceMatcher(None, norm(current["text"]), norm(groq_text)).ratio()
            if sim >= similarity_threshold:
                current["end"] = ts
                if len(groq_text) > len(current["text"]):
                    current["text"] = groq_text
                if len(tess_text) > len(current["tess_text"]):
                    current["tess_text"] = tess_text
                continue
            else:
                events.append(current)

        current = {"start": ts, "end": ts, "text": groq_text, "tess_text": tess_text, "frame": frame}

    if current:
        events.append(current)
    return events


def find_confirmed_typos(tess_text: str, groq_text: str, allowlist: set):
    """
    Cross-check: Tesseract reads text literally (typos survive), Groq reads
    it like a human (typos get silently auto-corrected). A word is a
    confirmed real typo when Tesseract's misspelling isn't a dictionary word,
    AND the dictionary's best-guess correction for it actually shows up in
    Groq's cleaned reading -- meaning that's genuinely what's on screen.
    """
    confirmed = []
    groq_lower = groq_text.lower()
    for w in WORD_RE.findall(tess_text):
        lw = w.lower()
        if lw in allowlist or lw in spell:
            continue  # not misspelled, or explicitly allowed
        correction = spell.correction(w)
        if correction and correction.lower() != lw and correction.lower() in groq_lower:
            confirmed.append(w)
    return confirmed


def check_spelling(text: str, allowlist: set):
    flagged = []
    for w in WORD_RE.findall(text):
        lw = w.lower()
        if lw in allowlist or lw in spell:
            continue
        flagged.append(w)
    return flagged


def load_history():
    with HISTORY_LOCK:
        if not HISTORY_FILE.exists():
            return []
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []


def add_history_entry(entry: dict):
    with HISTORY_LOCK:
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []
        history.insert(0, entry)
        history = history[:200]
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def parse_srt(srt_text: str):
    """Parse SRT content into a list of (start_seconds, end_seconds, text)."""
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    entries = []

    def to_seconds(ts):
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if re.match(r"^\d+$", lines[0].strip()) else 0
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[time_line_idx],
        )
        if not time_match:
            continue
        start, end = to_seconds(time_match.group(1)), to_seconds(time_match.group(2))
        text = " ".join(lines[time_line_idx + 1:])
        entries.append((start, end, text))
    return entries


def transcribe_with_groq(video_path: Path) -> list:
    """Extract audio and get a timestamped transcript back from Groq Whisper."""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set — add it in Railway's Variables tab.")

    audio_path = video_path.with_suffix(".mp3")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(audio_path)]
    subprocess.run(cmd, check=True, capture_output=True)

    try:
        with open(audio_path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=(audio_path.name, f.read()),
                model=GROQ_WHISPER_MODEL,
                response_format="verbose_json",
            )
        segments = getattr(resp, "segments", None) or []
        entries = [(seg["start"], seg["end"], seg["text"]) for seg in segments]
        if not entries:
            entries = [(0, 0, getattr(resp, "text", ""))]
        return entries
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass


def process_video_via_transcript(video_path: Path, allowlist: set, srt_entries=None):
    """Spellcheck against subtitles (given SRT) or an auto-generated audio transcript."""
    entries = srt_entries if srt_entries is not None else transcribe_with_groq(video_path)

    flags = []
    for start, end, text in entries:
        flagged_words = check_spelling(text, allowlist)
        remark = "Spelling mistake — needs fix" if flagged_words else "Matches — no issue"
        flags.append({
            "start": format_ts(start),
            "end": format_ts(end),
            "words": sorted(set(flagged_words)),
            "screen_text": text.strip(),
            "model_text": text.strip(),
            "remark": remark,
            "has_issue": bool(flagged_words),
            "thumb": None,
        })
    return flags


def process_video(job_id: str, video_path: Path, allowlist: set, fps: float,
                   method: str = "ocr", srt_entries=None):
    key = video_path.name
    with JOBS_LOCK:
        JOBS[job_id]["videos"][key]["status"] = "processing"

    job_report_dir = REPORT_DIR / job_id
    thumbs_dir = job_report_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = job_report_dir / f"_frames_{video_path.stem}"

    try:
        if method == "srt":
            flags = process_video_via_transcript(video_path, allowlist, srt_entries=srt_entries)
        else:
            frame_data = extract_frames(video_path, frame_dir, fps)
            readings = [(ts, ocr_frame_groq(fp), ocr_frame_tesseract(fp), fp) for fp, ts in frame_data]
            events = dedupe_readings(readings)

            flags = []
            for ev in events:
                confirmed_words = find_confirmed_typos(ev["tess_text"], ev["text"], allowlist)

                thumb_name = f"{video_path.stem}_{format_ts(ev['start']).replace(':', '')}.jpg"
                thumb_path = thumbs_dir / thumb_name
                try:
                    os.replace(ev["frame"], thumb_path)
                except OSError:
                    pass

                if confirmed_words:
                    remark = "Spelling mistake — needs fix"
                else:
                    remark = "Matches — no issue"

                flags.append({
                    "start": format_ts(ev["start"]),
                    "end": format_ts(ev["end"]),
                    "words": sorted(set(confirmed_words)),
                    "screen_text": ev["tess_text"].replace("\n", " ").strip(),
                    "model_text": ev["text"].replace("\n", " ").strip(),
                    "remark": remark,
                    "has_issue": bool(confirmed_words),
                    "thumb": f"/reports/{job_id}/thumbs/{thumb_name}" if thumb_path.exists() else None,
                })

            for f in frame_dir.glob("*.jpg"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                frame_dir.rmdir()
            except OSError:
                pass

        with JOBS_LOCK:
            JOBS[job_id]["videos"][key]["status"] = "done"
            JOBS[job_id]["videos"][key]["flags"] = flags

        write_csv_report(job_id)

        add_history_entry({
            "job_id": job_id,
            "video": key,
            "method": method,
            "flag_count": sum(1 for f in flags if f.get("has_issue")),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "csv_url": f"/reports/{job_id}/report.csv",
        })

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["videos"][key]["status"] = "error"
            JOBS[job_id]["videos"][key]["error"] = str(e)


def write_csv_report(job_id: str):
    job_report_dir = REPORT_DIR / job_id
    csv_path = job_report_dir / "report.csv"
    with JOBS_LOCK:
        videos = JOBS[job_id]["videos"]
        rows = []
        for vname, vdata in videos.items():
            for flag in vdata.get("flags", []):
                rows.append({
                    "video": vname,
                    "start_time": flag["start"],
                    "end_time": flag["end"],
                    "remark": flag.get("remark", ""),
                    "flagged_words": ", ".join(flag.get("words", [])),
                    "on_screen_text": flag.get("screen_text", ""),
                    "model_text": flag.get("model_text", ""),
                })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "start_time", "end_time", "remark", "flagged_words", "on_screen_text", "model_text"
        ])
        writer.writeheader()
        writer.writerows(rows)


def run_job(job_id: str, fps: float, method: str, srt_entries_map: dict):
    with JOBS_LOCK:
        video_names = list(JOBS[job_id]["videos"].keys())
        allowlist = JOBS[job_id]["allowlist"]

    for name in video_names:
        video_path = UPLOAD_DIR / job_id / name
        process_video(job_id, video_path, allowlist, fps, method=method,
                       srt_entries=srt_entries_map.get(name))

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "done"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("videos")
    allowlist_raw = request.form.get("allowlist", "")
    fps = float(request.form.get("fps", DEFAULT_FPS))
    method = request.form.get("method", "ocr")  # "ocr" or "srt"
    allowlist = {w.strip().lower() for w in allowlist_raw.splitlines() if w.strip()}

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    videos = {}
    for f in files:
        filename = secure_filename(f.filename)
        if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        save_path = job_upload_dir / filename
        f.save(save_path)
        videos[filename] = {"status": "queued", "flags": []}

    if not videos:
        return jsonify({"error": "No valid video files found"}), 400

    # Optional SRT files: matched to a video by shared filename stem
    srt_entries_map = {}
    if method == "srt":
        srt_files = request.files.getlist("srt_files")
        for sf in srt_files:
            sf_name = secure_filename(sf.filename)
            stem = Path(sf_name).stem
            matched_video = next((v for v in videos if Path(v).stem == stem), None)
            if matched_video:
                srt_entries_map[matched_video] = parse_srt(sf.read().decode("utf-8", errors="ignore"))

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing", "videos": videos, "allowlist": allowlist}

    thread = threading.Thread(target=run_job, args=(job_id, fps, method, srt_entries_map), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/history")
def history():
    return jsonify(load_history())


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        # allowlist is stored as a Python set internally (fast lookups) but
        # sets aren't JSON-serializable, so exclude it from the response.
        safe_job = {"status": job["status"], "videos": job["videos"]}
        return jsonify(safe_job)


@app.route("/reports/<job_id>/thumbs/<filename>")
def serve_thumb(job_id, filename):
    return send_from_directory(REPORT_DIR / job_id / "thumbs", filename)


@app.route("/reports/<job_id>/report.csv")
def serve_csv(job_id):
    return send_from_directory(REPORT_DIR / job_id, "report.csv", as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
