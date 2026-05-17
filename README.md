# Video Dubbing Pipeline with Hatchet and Nebius Serverless

This repository contains a video dubbing pipeline that turns an input video into
a translated, dubbed output video. It uses Hatchet to orchestrate the workflow
and Nebius Serverless Jobs to run the compute-heavy media and AI steps.

The goal is not to build a full video editing product. The goal is to show the
backend shape of a scalable dubbing system: one that can take a video from
object storage, process it through several specialized workloads, recover from
transient failures, and write the final dubbed video back to storage.

## What This Repo Does

The pipeline starts with a video file stored in Nebius Object Storage. A Hatchet
workflow then coordinates a sequence of jobs:

```text
input video in object storage
  -> extract audio with ffmpeg          (Nebius CPU job)
  -> transcribe speech with Whisper     (Nebius GPU job, preemptible)
  -> translate transcript with MADLAD-400 (Nebius CPU job)
  -> synthesize dubbed speech with Coqui TTS (Nebius GPU job, preemptible)
  -> merge video with dubbed audio      (Nebius CPU job)
  -> output dubbed video in object storage
```

Your local machine only runs the Hatchet worker. It never touches the video
data. Everything passes through object storage.

## Why Use Hatchet

Video dubbing is a workflow problem, not just a model invocation problem. A
single run involves multiple long-running steps, different container images,
GPU scheduling, cloud job polling, storage checks, and retries.

Hatchet is used here because it gives the pipeline:

- durable execution state for each dubbing run
- explicit task dependencies between stages
- automatic retries when a GPU job is preempted
- a clean visual timeline in the Traces view
- logs and visibility for every stage

The Hatchet worker does not perform the heavy computation itself. It submits
jobs to Nebius, waits for them to complete, validates that expected artifacts
exist in storage, and then moves the workflow to the next step.

## Why Use Nebius Serverless

Video dubbing combines workloads with very different compute needs. ffmpeg runs
well on CPU, while Whisper and TTS benefit from GPUs. Keeping all of that
capacity running all the time is wasteful.

Nebius Serverless Jobs let the pipeline run each stage on the right infrastructure:

- CPU jobs for ffmpeg and translation
- GPU jobs on preemptible L40S for Whisper and TTS
- mounted object storage so every container sees the same `/data` workspace
- isolated, reproducible execution through Docker images

When a preemptible GPU is reclaimed, Nebius sets the job state to `ERROR`. The
pipeline detects this and raises a `RuntimeError`, which Hatchet catches and
retries automatically on a new GPU.

## Architecture

```text
Hatchet Cloud (orchestration + dashboard)
        |
        v
Your machine (Hatchet worker, orchestrates only)
        |
        v
Nebius Object Storage (data bus — video, audio, transcript, output)
        |
        +-- ffmpeg CPU job        (extract audio)
        +-- Whisper GPU job       (transcribe, preemptible L40S)
        +-- MADLAD-400 CPU job    (translate)
        +-- Coqui TTS GPU job     (synthesize, preemptible L40S)
        +-- ffmpeg CPU job        (remux video)
```

## Repository Layout

```text
.
|-- worker.py                    # Starts the Hatchet worker
|-- trigger.py                   # Triggers a single dubbing run
|-- batch_trigger.py             # Triggers multiple runs staggered over time
|-- requirements.txt             # Python dependencies for the worker
|-- .env.example                 # Environment variable template
|-- pipeline/
|   |-- workflow.py              # Hatchet workflow and task definitions
|   |-- nebius.py                # Nebius job creation and polling helpers
|   |-- config.py                # Environment-based settings
|   `-- local.py                 # Object storage helpers
`-- containers/
    |-- whisper/                 # faster-whisper transcription image
    |-- translate/               # MADLAD-400 translation image
    `-- tts/                     # Coqui TTS synthesis image
```

## Requirements

- Python 3.11
- Docker
- [uv](https://github.com/astral-sh/uv) for Python environment management
- A [Hatchet Cloud](https://cloud.onhatchet.run) account (free tier works)
- Nebius credentials with access to Serverless Jobs and Object Storage
- Container images pushed to a public registry (Docker Hub works)

## Important: Build Containers on AMD64

Nebius runs x86_64 (AMD64). If you build Docker images on Apple Silicon (M1/M2/M3),
the containers will fail to start with an architecture mismatch error.

Always build and push from a Linux AMD64 machine. A Nebius CPU VM works well:

```bash
# Create a CPU VM on Nebius
nebius compute instance create \
  --parent-id YOUR_PROJECT_ID \
  --name docker-builder \
  --preset 4vcpu-16gb \
  --platform cpu-e2 \
  --subnet-id YOUR_SUBNET_ID \
  --image-family ubuntu22.04 \
  --ssh-public-key "$(cat ~/.ssh/id_rsa.pub)"

# SSH in, install Docker, build and push
sudo apt-get install -y docker.io
docker build -t your-user/nebius-whisper:latest containers/whisper/
docker push your-user/nebius-whisper:latest
```

## Configuration

```bash
cp .env.example .env
```

Fill in the required values:

```text
HATCHET_CLIENT_TOKEN     # From Hatchet Cloud dashboard -> Settings -> API Keys
NEBIUS_IAM_TOKEN         # From: nebius iam get-access-token
NEBIUS_PROJECT_ID        # From: nebius iam project list
NEBIUS_SUBNET_ID         # From: nebius vpc subnet list
NEBIUS_BUCKET_ID         # From: nebius storage bucket list
NEBIUS_BUCKET_NAME       # Your bucket name
AWS_ACCESS_KEY_ID        # Nebius storage access key
AWS_SECRET_ACCESS_KEY    # Nebius storage secret key
AWS_ENDPOINT_URL         # https://storage.eu-north1.nebius.cloud
WHISPER_IMAGE            # your-dockerhub-user/nebius-whisper:latest
TRANSLATE_IMAGE          # your-dockerhub-user/nebius-translate:latest
TTS_IMAGE                # your-dockerhub-user/nebius-tts:latest
```

> **Note on IAM tokens**: `nebius iam get-access-token` generates a short-lived
> session token (expires in ~1 hour). For long-running workflows, create a
> service account and use its credentials instead.

## Install Dependencies

Use `uv` instead of standard `venv` — standard venv has known issues on newer
macOS versions:

```bash
brew install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run the Workflow

Upload a source video to your bucket:

```bash
aws s3 cp my-video.mp4 s3://your-bucket/my-video.mp4 \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Start the worker:

```bash
python worker.py
```

Trigger a single dubbing run:

```bash
python trigger.py --video my-video.mp4 --lang de --run-id first-run
```

Trigger a batch of runs staggered 20 seconds apart:

```bash
python batch_trigger.py --video my-video.mp4 --count 5 --interval 20 --lang de
```

When the workflow completes, the final dubbed video is written to object storage:

```text
my-video_first-run_dubbed.mp4
```

Monitor runs in the Hatchet dashboard at https://cloud.onhatchet.run. The
**Traces** tab shows a Gantt chart of all pipeline steps.

## Demo: Simulating Preemption

To demonstrate GPU preemption recovery during a live demo:

1. Trigger a batch run
2. Wait for a run to reach the `transcribe_whisper` step (shown as running in the dashboard)
3. Find the Nebius compute instance running the job:

```bash
nebius compute instance list --format json
```

4. Delete it to simulate preemption:

```bash
nebius compute instance delete --id INSTANCE_ID
```

5. Watch the Hatchet dashboard: the step goes red, Hatchet retries on a new GPU,
   the step turns green. The rest of the batch continues unaffected.

## Known Limitations and Workarounds

**Object storage does not support seeking**: WAV and MP4 files cannot be written
directly to the `/data` FUSE mount because ffmpeg and scipy need to seek back to
write headers. The workaround used here is to write to `/tmp` first, then copy
the completed file to `/data`.

**Translation quality**: MADLAD-400 3B sometimes hallucinates on short texts or
unusual input. For production use, replace it with a stronger model or a
translation API.

**No time alignment**: The dubbed audio is synthesized from the full translated
text as a single pass. It will not match the timing of the original speech. 
For production dubbing, forced alignment (e.g. WhisperX) is needed.

**This is a reference pipeline**: No web UI, speaker diarization, lip sync, or
audio mixing. These are natural extensions on top of the orchestration pattern.
