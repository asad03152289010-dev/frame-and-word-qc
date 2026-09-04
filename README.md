# Frame & Word — Video Spelling QC (Web App)

Upload multiple videos in the browser → it OCRs every frame in the
background → flags likely spelling mistakes with timestamps and
thumbnails → download a CSV report. No command line needed once deployed.

## Deploy — easiest options (free tier friendly)

### Option A: Railway (recommended, ~5 min)
1. Push this folder to a GitHub repo.
2. Go to https://railway.app → New Project → Deploy from GitHub repo.
3. Railway auto-detects the `Dockerfile` and builds it (ffmpeg + tesseract
   get installed automatically inside the container — nothing to configure).
4. Once deployed, Railway gives you a public URL — that's your always-on
   webpage.

### Option B: Render
1. Push to GitHub.
2. https://render.com → New → Web Service → connect the repo.
3. Render detects the `Dockerfile` automatically. Set instance type based on
   expected video sizes (more RAM = handles bigger files more comfortably).
4. Deploy → you get a public URL.

### Option C: Your own VPS (DigitalOcean, AWS EC2, etc.)
```bash
git clone <your-repo>
cd webapp
docker build -t video-qc .
docker run -d -p 80:8080 -v $(pwd)/uploads:/app/uploads -v $(pwd)/reports:/app/reports video-qc
```
Then point a domain at the server's IP if you want a custom URL.

## Local testing (before deploying)
```bash
pip install -r requirements.txt
# also needs ffmpeg + tesseract installed locally, see main script's README
python app.py
```
Open http://localhost:8080

## How it works
- `/` — upload page (drag-drop multiple videos, set sample rate + allow-list)
- `/upload` — saves videos, kicks off a background thread per job
- `/status/<job_id>` — polled every 2s by the page to show live progress
- Each video: frames extracted via ffmpeg → OCR'd via Tesseract → deduped →
  spellchecked → flagged events get a saved thumbnail
- `/reports/<job_id>/report.csv` — downloadable combined report

## Notes on scaling for real traffic
- Current setup processes one video at a time per job, in a background
  thread — fine for moderate use. If many people upload simultaneously,
  swap the in-memory `JOBS` dict + threading for a real task queue
  (Celery + Redis, or RQ) so jobs run across multiple workers instead of
  one process. Happy to build that version if you outgrow this.
- Uploaded videos and generated thumbnails sit on local disk — for a
  longer-lived deployment, point `UPLOAD_DIR` / `REPORT_DIR` at mounted
  cloud storage (S3, etc.) so a redeploy doesn't wipe them.
- `MAX_CONTENT_LENGTH` is set to 2GB per upload request — adjust in
  `app.py` if your videos are larger, and make sure your host's request
  size limits (e.g. Render/Railway proxy limits) match.
"# frame-and-word-qc" 
