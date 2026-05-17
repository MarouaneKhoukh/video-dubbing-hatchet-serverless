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
  -> extract audio with ffmpeg
  -> transcribe speech with Whisper
  -> translate transcript with MADLAD-400
  -> synthesize dubbed speech with Coqui TTS
  -> merge the original video with the dubbed audio
  -> output dubbed video in object storage
```

Each stage is isolated. CPU-friendly media steps run as CPU jobs, while Whisper
and text-to-speech run as GPU jobs. The video, transcript, translated text, and
generated audio are passed between stages through object storage instead of
through the local machine.

## Why Use Hatchet

Video dubbing is a workflow problem, not just a model invocation problem. A
single run may involve multiple long-running steps, different container images,
GPU scheduling, cloud job polling, storage checks, and retries.

Hatchet is used here as the workflow engine because it gives the pipeline:

- durable execution state for each dubbing run
- explicit task dependencies between media, transcription, translation, and TTS
- automatic retries when a job fails or a GPU worker is interrupted
- logs and visibility for each stage of the pipeline
- a clean boundary between orchestration logic and the actual AI containers

In this repo, the Hatchet worker does not perform the heavy computation itself.
It submits jobs to Nebius, waits for them to complete, validates that expected
artifacts exist in storage, and then moves the workflow to the next step.

## Why Use Nebius Serverless

Video dubbing combines workloads with very different compute needs. ffmpeg can
run well on CPU, while Whisper and TTS benefit from GPUs. Keeping all of that
capacity running all the time is wasteful, especially when dubbing jobs arrive
irregularly.

Nebius Serverless Jobs are used because they let the pipeline run each stage on
the right kind of infrastructure:

- CPU jobs for ffmpeg extraction, translation, and remuxing
- GPU jobs for Whisper transcription and Coqui TTS synthesis
- preemptible GPU capacity for cost-sensitive AI stages
- mounted object storage so every container sees the same `/data` workspace
- isolated, reproducible execution through Docker images

The workflow can treat a preempted GPU job as a retryable failure. When Nebius
reports the job as failed or interrupted, the helper code raises an error and
Hatchet retries the task on a new job.

## Architecture

The main workflow is defined in `pipeline/workflow.py`.

```text
Hatchet event: dubbing:start
        |
        v
extract_audio
  Nebius CPU job, ffmpeg
        |
        v
transcribe_whisper
  Nebius GPU job, faster-whisper
        |
        v
translate_text_step
  Nebius CPU job, MADLAD-400
        |
        v
synthesize_tts
  Nebius GPU job, Coqui TTS
        |
        v
remux_video
  Nebius CPU job, ffmpeg
```

The local worker is responsible for orchestration only. The actual video and
audio files stay in object storage and are mounted into Nebius jobs at `/data`.

## Repository Layout

```text
.
|-- worker.py                    # Starts the Hatchet worker
|-- trigger.py                   # Starts a dubbing workflow run
|-- batch_trigger.py             # Helper for triggering multiple runs
|-- docker-compose.yml           # Local Hatchet stack
|-- requirements.txt             # Python dependencies for the worker
|-- pipeline/
|   |-- workflow.py              # Hatchet workflow and task definitions
|   |-- nebius.py                # Nebius job creation and polling helpers
|   |-- config.py                # Environment-based settings
|   `-- local.py                 # Object storage and local utility helpers
`-- containers/
    |-- whisper/                 # Whisper transcription image
    |-- translate/               # MADLAD-400 translation image
    `-- tts/                     # Coqui TTS synthesis image
```

## Requirements

You need:

- Python 3.11 or newer
- Docker
- a Hatchet instance or local Hatchet stack
- Nebius credentials with access to Serverless Jobs
- a Nebius Object Storage bucket
- container images for Whisper, translation, and TTS pushed to a registry that
  Nebius can pull from

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Then fill in the required values:

```text
HATCHET_CLIENT_TOKEN
NEBIUS_IAM_TOKEN
NEBIUS_PROJECT_ID
NEBIUS_SUBNET_ID
NEBIUS_BUCKET_ID
NEBIUS_BUCKET_NAME
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL
WHISPER_IMAGE
TRANSLATE_IMAGE
TTS_IMAGE
```

The `.env` file is intentionally ignored by Git because it contains credentials.

## Build the Container Images

Build and push the workload images before starting the pipeline:

```bash
docker build -t your-registry/nebius-whisper:latest containers/whisper/
docker push your-registry/nebius-whisper:latest

docker build -t your-registry/nebius-translate:latest containers/translate/
docker push your-registry/nebius-translate:latest

docker build -t your-registry/nebius-tts:latest containers/tts/
docker push your-registry/nebius-tts:latest
```

Update `WHISPER_IMAGE`, `TRANSLATE_IMAGE`, and `TTS_IMAGE` in `.env` with the
image names you pushed.

## Run the Workflow

Start Hatchet locally if you are not using a hosted Hatchet instance:

```bash
docker compose up -d
```

Install the worker dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Upload a source video to your bucket:

```bash
aws s3 cp my-video.mp4 s3://your-bucket/input/my-video.mp4 \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Start the worker:

```bash
python worker.py
```

Trigger a dubbing run:

```bash
python trigger.py --video input/my-video.mp4 --lang de --run-id first-run
```

When the workflow completes, the final dubbed video is written back to object
storage with a key like:

```text
my-video_first-run_dubbed.mp4
```

## Operational Notes

- Keep source videos and generated artifacts in object storage, not in the Git
  repository.
- Use unique `run_id` values to avoid overwriting outputs from earlier runs.
- GPU stages are configured as retryable workflow tasks because they are the
  most likely to hit capacity or preemption issues.
- The default model choices are intended as starting points. For production
  dubbing, you may want stronger translation, speaker-aware TTS, alignment,
  voice cloning, subtitle generation, and review workflows.

## Current Limitations

This is a backend reference pipeline. It does not include a web UI, human review
step, speaker diarization, lip sync, or advanced audio mixing. Those are natural
extensions on top of the orchestration pattern used here.
