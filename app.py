#!/usr/bin/env python3
"""
Video Spelling QC — Web App
Upload multiple videos in the browser, they get processed in the background
(frame extraction -> OCR -> spellcheck), and results show up as a flagged
report per video, with thumbnails.
"""

import csv
import hashlib
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path

import cv2
import pytesseract
from flask import Flask, render_template, request, jsonify, send_from_directory
from spellchecker import SpellChecker
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

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


def ocr_frame(frame_path: Path) -> str:
    img = cv2.imread(str(frame_path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(gray).strip()


def dedupe_readings(readings):
    events = []
    prev_hash = None
    current = None

    def norm_hash(t):
        return hashlib.md5(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()

    for ts, text, frame in readings:
        if not text.strip():
            prev_hash = None
            if current:
                events.append(current)
                current = None
            continue
        h = norm_hash(text)
        if h == prev_hash and current:
            current["end"] = ts
        else:
            if current:
                events.append(current)
            current = {"start": ts, "end": ts, "text": text, "frame": frame}
        prev_hash = h
    if current:
        events.append(current)
    return events


def check_spelling(text: str, allowlist: set):
    flagged = []
    for w in WORD_RE.findall(text):
        lw = w.lower()
        if lw in allowlist or lw in spell:
            continue
        flagged.append(w)
    return flagged


def process_video(job_id: str, video_path: Path, allowlist: set, fps: float):
    key = video_path.name
    with JOBS_LOCK:
        JOBS[job_id]["videos"][key]["status"] = "processing"

    job_report_dir = REPORT_DIR / job_id
    thumbs_dir = job_report_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = job_report_dir / f"_frames_{video_path.stem}"

    try:
        frame_data = extract_frames(video_path, frame_dir, fps)
        readings = [(ts, ocr_frame(fp), fp) for fp, ts in frame_data]
        events = dedupe_readings(readings)

        flags = []
        for ev in events:
            flagged_words = check_spelling(ev["text"], allowlist)
            if not flagged_words:
                continue
            thumb_name = f"{video_path.stem}_{format_ts(ev['start']).replace(':', '')}.jpg"
            thumb_path = thumbs_dir / thumb_name
            try:
                os.replace(ev["frame"], thumb_path)
            except OSError:
                pass
            flags.append({
                "start": format_ts(ev["start"]),
                "end": format_ts(ev["end"]),
                "words": sorted(set(flagged_words)),
                "text": ev["text"].replace("\n", " ").strip(),
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

        # Update the combined CSV report
        write_csv_report(job_id)

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
                    "flagged_words": ", ".join(flag["words"]),
                    "full_ocr_text": flag["text"],
                })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "start_time", "end_time", "flagged_words", "full_ocr_text"])
        writer.writeheader()
        writer.writerows(rows)


def run_job(job_id: str, fps: float):
    with JOBS_LOCK:
        video_names = list(JOBS[job_id]["videos"].keys())
        allowlist = JOBS[job_id]["allowlist"]

    for name in video_names:
        video_path = UPLOAD_DIR / job_id / name
        process_video(job_id, video_path, allowlist, fps)

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

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing", "videos": videos, "allowlist": allowlist}

    thread = threading.Thread(target=run_job, args=(job_id, fps), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


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
