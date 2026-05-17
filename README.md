# Video dubbing pipeline — Hatchet + Nebius

Video translation and dubbing orchestrated with Hatchet, running GPU workloads
on preemptible Nebius H200 jobs. Whisper + Coqui TTS run on GPU; ffmpeg and
MADLAD-400 translation run locally.

```
video in (object storage)
  → ffmpeg audio extraction (local)
  → Whisper transcription (Nebius H200, preemptible) ← retries on preemption
  → MADLAD-400 translation (local)
  → Coqui TTS synthesis (Nebius H200, preemptible)   ← retries on preemption
  → ffmpeg remux (local)
  → dubbed video out (object storage)
```

---

## 1. Start Hatchet locally

```bash
docker compose up -d
```

Open the dashboard: http://localhost:8090

Create an API key: Settings → API Keys → New key → copy it.

---

## 2. Configure environment

```bash
cp .env.example .env
# Fill in your tokens and IDs
```

Required values:
- `HATCHET_CLIENT_TOKEN` — from the Hatchet dashboard
- `NEBIUS_IAM_TOKEN` — from `nebius iam token create`
- `NEBIUS_PROJECT_ID` — from `nebius iam project list`
- `NEBIUS_SUBNET_ID` — from `nebius vpc subnet list`
- `NEBIUS_BUCKET_ID` — from `nebius storage bucket list`
- `NEBIUS_BUCKET_NAME` — your bucket name
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — Nebius storage keys

---

## 3. Build and push GPU containers

```bash
# Whisper
docker build -t your-org/nebius-whisper:latest containers/whisper/
docker push your-org/nebius-whisper:latest

# TTS
docker build -t your-org/nebius-tts:latest containers/tts/
docker push your-org/nebius-tts:latest
```

Update `WHISPER_IMAGE` and `TTS_IMAGE` in `.env`.

---

## 4. Upload a test video

```bash
aws s3 cp my-video.mp4 s3://your-bucket/input/my-video.mp4 \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

---

## 5. Install dependencies and start the worker

```bash
pip install -r requirements.txt
python worker.py
```

---

## 6. Trigger a run

```bash
python trigger.py --video input/my-video.mp4 --lang de --run-id demo-001
```

Watch it in the Hatchet dashboard at http://localhost:8090.

---

## Demo: simulating preemption

To show preemption recovery during the demo:

1. Trigger a run
2. Wait for `transcribe_whisper` to show as **running** in the dashboard
3. In another terminal, find the job ID from the Hatchet logs and kill it:

```bash
nebius ai job delete <JOB_ID>
```

4. The Nebius job transitions to `ERROR` state
5. The worker detects this, raises `RuntimeError`, Hatchet marks the step as
   failed and immediately retries it — a new job is created on a fresh GPU
6. Watch the dashboard: step goes red → orange (retrying) → green

This is exactly what happens automatically with real preemptible jobs.
