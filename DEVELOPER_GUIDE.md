# Developer Guide

Three levels of running the pipeline, from fastest iteration to full cloud deployment.


| Level                   | What runs                            | Needs                         |
| ----------------------- | ------------------------------------ | ----------------------------- |
| **1a — Python scripts** | Single step, no Docker               | `uv sync` + per-step deps     |
| **1b — Docker local**   | Full pipeline via `python -m pipeline run` | Docker + local images   |
| **2 — Cloud**           | Batched pipeline on Nebius + Hatchet | `.env` credentials, worker    |


**Start with Level 1b** via `python -m pipeline run …` — same stages, paths, and skip/resume rules as the cloud workflow. No Nebius or Hatchet credentials required.

---

## Architecture

```text
python -m pipeline run …     → LocalExecutor (docker run, ./data → /data)
python -m hatchet.trigger …  → NebiusExecutor (Hatchet tasks → pipeline.stages)
```

- **`pipeline/`** — framework-free engine: input resolution, path layout, stage logic, batching, manifests
- **`hatchet/`** — thin orchestration: workflow registration + trigger/worker CLIs
- **`jobs/`** — container entrypoints (what Docker/Nebius actually runs)

Both local and cloud use the same `pipeline.stages.*.run()` implementations and the same output layout under `runs/{run_id}/`.

---

## Project layout

```text
src/
  hatchet/              ← Hatchet adapter + Typer CLIs
    console.py          ← Rich console + logging helpers
    download.py         ← sample download commands
    trigger.py          ← batch trigger commands
    worker.py           ← Hatchet worker
    workflow.py         ← thin Hatchet workflow (delegates to pipeline.stages)
  pipeline/             ← orchestration engine (framework-free core)
    config.py           ← nested per-task settings (defaults in code)
    run.py              ← PipelineRun + run_stage / run_pipeline
    metadata.py         ← task manifests, reports, upstream input discovery
    utils.py            ← utc_now, Rich console + logging helpers
    __main__.py         ← `python -m pipeline run …` local Docker CLI
    stages/             ← per-step runners (extract, transcribe, translate, tts, remux)
    runner/             ← JobExecutor (NebiusExecutor, LocalExecutor)
    batch_runner.py     ← fan-out helper (filter → chunk → parallel jobs)
    nebius.py           ← Nebius SDK helpers (create_and_wait)
    paths.py            ← bucket-relative paths: resolve_video_keys(), build_run_items()
    storage.py          ← artifact I/O (S3 or local filesystem via use_local_artifacts)
  jobs/                 ← Nebius container entrypoints
    paths.py            ← artifact path layout (mirror of pipeline.paths)
    run_manifest.py     ← load run manifest + chunk slicing
    transcribe.py       ← ASR + alignment (faster-whisper + WhisperX)
    translate.py        ← NLLB-200
    tts.py              ← Kokoro TTS
  models/               ← model cache paths, config resolution, pre-download
    model_cache.py      ← cache helpers + ModelSpec (from config.py)
    download_models.py  ← download logic + Docker-safe argparse fallback
    download_cli.py     ← Typer + Rich CLI (host only)
    console.py          ← shared Rich console
docker/
  base-cpu.Dockerfile      ← shared CPU + torch (Mac / no-GPU local dev)
  base-cuda.Dockerfile     ← shared CUDA + torch (Nebius / Linux+NVIDIA)
  transcribe.Dockerfile
  translate.Dockerfile
  tts.Dockerfile
  docker-compose.yml    ← self-hosted Hatchet (optional)
scripts/
  download_samples.py   ← shim → hatchet.download
  download_models.py    ← pre-download weights to data/models
data/                   ← local input/output (git-ignored)
```

Docker image names follow **pipeline steps**, not model names. **Local builds** use `video-dubbing-{base,transcribe,translate,tts}:local`. Remote image URIs for Nebius jobs are set in `src/pipeline/config.py`.

**Image build stack:** three task images `FROM` a shared `video-dubbing-base:local` (build-only intermediate). Python deps are pinned in `pyproject.toml` `[dependency-groups]`; Docker installs only the relevant group per layer via `uv sync --group <name>` (no lockfile). Layer order: base (CPU or CUDA) → apt (if needed) → pyproject.toml → task deps → job script.

### CLI (Typer + Rich)

After `uv pip install -e .`:


| Task                      | Command                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| **Local pipeline**        | `python -m pipeline run sample.mp4 --run-id demo`                                 |
| Local — single stage      | `python -m pipeline run sample.mp4 --run-id demo --stage extract --force`         |
| Local — batch prefix      | `python -m pipeline run sample_batch/ --run-id batch-001 --device cpu`              |
| Download samples          | `python scripts/download_samples.py nasa --sample-size 10`                        |
| Download models           | `python scripts/download_models.py all`                                           |
| Model cache status        | `python scripts/download_models.py status`                                        |
| Download (module)         | `python -m models all` or `python -m models.download_models translate`            |
| Download samples (module) | `python -m hatchet.download nasa --sample-size 10`                                |
| Cloud trigger             | `python -m hatchet.trigger run sample.mp4 --run-id demo`                          |
| Cloud trigger (prefix)    | `python -m hatchet.trigger run sample_batch/ --run-id demo`                       |
| Start worker              | `python -m hatchet.worker`                                                        |


---

## Level 1b — Full local pipeline with Docker

Runs every step in a container via **`python -m pipeline run`**. No Nebius or Hatchet
needed — cloud credentials in `.env` are optional for local runs. Files are read/written
under `data/` (mounted as `/data` in containers).

### Prerequisites

- Docker Desktop (or Docker Engine) running
- Local images built (see below): `video-dubbing-{transcribe,translate,tts}:local` + ffmpeg image

Extract and remux use the pre-built image `lscr.io/linuxserver/ffmpeg:latest` (same as cloud).
The pipeline CLI invokes it with `--entrypoint sh -c '…'` internally — you do not need host `ffmpeg`.

### 0. Get a sample video

The `data/` directory already contains sample files if you ran `scripts/download_samples.py`.
Otherwise place any short MP4 there:

```bash
ls data/
cp /path/to/your-video.mp4 data/sample.mp4
```

We use `data/sample.mp4` as the example below.

### 1. Build / pull local images

Build AI task images from the **repo root** (`COPY src/jobs/...` and `pyproject.toml` resolve from here).
Requires [Docker BuildKit](https://docs.docker.com/build/buildkit/) (default in Docker Desktop).

**Model cache (do once):** pre-download weights to `data/models/`. Model IDs default in `src/pipeline/config.py` (override via env vars when `.env` is present). Tasks skip download when weights are already cached; re-running `download_models.py all` is safe (prints `SKIP (cached)`).

```bash
mkdir -p data/models

# Pre-download everything (reads config.py; Settings overrides when .env exists)
uv pip install -e .
# Host-only: transcribe/tts need task dependency groups — or use Docker fallback below
# uv sync --group cpu-base --group transcribe --group translate --group tts
python scripts/download_models.py all
python scripts/download_models.py status   # cache table without downloading

# Per-task (optional): transcribe | translate | tts
# python scripts/download_models.py transcribe

# Docker fallback (run once per image, or `all` on transcribe image for transcribe-only)
MODELS="$(pwd)/data/models"
docker run --rm -v "$MODELS:/data/models" --entrypoint python3 video-dubbing-transcribe:local /download_models.py transcribe
docker run --rm -v "$MODELS:/data/models" --entrypoint python3 video-dubbing-translate:local /download_models.py translate
docker run --rm -v "$MODELS:/data/models" --entrypoint python3 video-dubbing-tts:local /download_models.py tts
```

Mount the cache on every AI task container run:

```bash
MODELS="$(pwd)/data/models"
# add to each docker run:  -v "$MODELS:/data/models"
```

**Pull ffmpeg** (extract + remux — same pre-built image as Nebius; no repo Dockerfile):

```bash
docker pull lscr.io/linuxserver/ffmpeg:latest
```

**Build shared base** (pinned torch; build once):

```bash
export DOCKER_BUILDKIT=1
```


| Host                               | Base Dockerfile               | Why                                                                     |
| ---------------------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| **Mac / no NVIDIA GPU** (typical)  | `docker/base-cpu.Dockerfile`  | CUDA torch fails at import without `libcudart` even when you pass `cpu` |
| **Linux + NVIDIA GPU** (local GPU) | `docker/base-cuda.Dockerfile` | Matches Nebius; use `--gpus all` on task containers                     |


**Mac / CPU-only** (recommended for Apple Silicon):

```bash
docker build -f docker/base-cpu.Dockerfile -t video-dubbing-base:local .
```

**Linux + NVIDIA** (optional local GPU):

```bash
docker build -f docker/base-cuda.Dockerfile -t video-dubbing-base:local .
```

**Step 2 — task images** (incremental deps on base; `BASE_IMAGE` defaults to `video-dubbing-base:local`):

```bash
docker build -f docker/transcribe.Dockerfile -t video-dubbing-transcribe:local .
docker build -f docker/translate.Dockerfile  -t video-dubbing-translate:local  .
docker build -f docker/tts.Dockerfile        -t video-dubbing-tts:local        .
```

> **Apple Silicon / no GPU:** Use `base-cpu.Dockerfile` above. Transcribe accepts `cpu` as the
> device argument (shown below). Translate and TTS use CPU automatically without `--gpus all`.
> Expect ~10× slower than GPU for transcribe/TTS.

**Bump container deps:** edit pins in `pyproject.toml` under `[dependency-groups]`, then rebuild the affected image(s):

```bash
docker build -f docker/base-cpu.Dockerfile -t video-dubbing-base:local .    # if cpu-base changed (Mac)
# docker build -f docker/base-cuda.Dockerfile -t video-dubbing-base:local . # if cuda-base changed (Nebius / Linux GPU)
docker build -f docker/transcribe.Dockerfile -t video-dubbing-transcribe:local .
# … rebuild only images whose group changed
```

### 2. Run the pipeline

**Primary interface** — same command shape for single file, batch folder, full pipeline, or one stage:

```bash
uv pip install -e .

# Full pipeline — one video (Mac: always pass --device cpu)
python -m pipeline run data/sample.mp4 --run-id demo --device cpu

# Batch — all videos under a prefix
python -m pipeline run data/sample_batch/ --run-id batch-001 --device cpu

# Single stage (repeat for extract | transcribe | translate | tts | remux)
python -m pipeline run data/sample.mp4 --run-id demo --stage extract --force
python -m pipeline run data/sample.mp4 --run-id demo --stage transcribe --force --device cpu
python -m pipeline run data/sample.mp4 --run-id demo --stage translate --force
python -m pipeline run data/sample.mp4 --run-id demo --stage tts --force
python -m pipeline run data/sample.mp4 --run-id demo --stage remux --force

# Batch + single stage
python -m pipeline run data/sample_batch/ --run-id batch-001 --stage extract --force

# Multiple stages in order
python -m pipeline run data/sample.mp4 --run-id demo --stage extract --stage transcribe --device cpu

# Linux + NVIDIA GPU inside containers
python -m pipeline run data/sample.mp4 --run-id demo --device cuda --gpus

# Override target language (NLLB; default from Settings / TARGET_LANG in .env)
python -m pipeline run data/sample.mp4 --run-id demo --lang de --device cpu
```

**Input paths:** bucket-relative keys under `data/`. The `data/` prefix is stripped automatically
(`data/sample.mp4` → `sample.mp4`). A folder prefix scans for videos (e.g. `sample_batch/`).

**Outputs** land under `data/runs/{run_id}/` (same layout as the Nebius bucket):

```text
data/runs/{run_id}/extract/{stem}.wav
data/runs/{run_id}/transcribe/{stem}.txt
data/runs/{run_id}/transcribe/{stem}_aligned.json
data/runs/{run_id}/translate/{stem}.txt
data/runs/{run_id}/tts/{stem}.wav
data/runs/{run_id}/remux/{stem}.mp4
```

Input videos stay at their original keys (e.g. `data/sample.mp4` → `/data/sample.mp4` in containers).

#### CLI options

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--run-id` | `demo` | Output namespace under `runs/{run_id}/` |
| `--stage` | all 5 stages | Repeat for multiple stages in order |
| `--force` | off | Reprocess even when outputs exist |
| `--device` | `cpu` | Transcribe device (`cpu` or `cuda`) |
| `--gpus` | off | Pass `--gpus all` to GPU task containers |
| `--lang` | `settings.target_lang` | NLLB translate target |
| `--data-dir` | `<repo>/data` | Host directory mounted at `/data` |
| `--models-dir` | `<data-dir>/models` | Model cache mount |

#### Target language

Translate target comes from `--lang` or `TARGET_LANG` in `.env` (via `Settings`). TTS voice/lang
come from `settings.tts` in `config.py` (`TTS__VOICE`, `TTS__LANG`) — **not** auto-set from `--lang`.

| Step | Config / flag | Default (Spanish dub) |
| ---- | ------------- | --------------------- |
| Translate | `--lang` / `TARGET_LANG` | `es` |
| TTS | `settings.tts.voice` / `settings.tts.lang` | `af_bella` / `e` |

#### Skip / re-run behavior

Each stage skips files whose **output artifacts already exist** under `runs/{run_id}/`
(unless `--force`). To re-run after changing language or config:

```bash
python -m pipeline run data/sample.mp4 --run-id demo --stage translate --force
# or delete: rm -rf data/runs/demo/translate data/runs/demo/tts data/runs/demo/remux
```

Running a single stage expects prior stages to have completed (e.g. transcribe needs extract output at `runs/{run_id}/extract/{stem}.wav`).

#### Verify output

```bash
ls -lh data/runs/demo/extract/
ls -lh data/runs/demo/remux/    # final dubbed videos
```

---

### 2b. Manual docker runs (debugging)

Use only when debugging a single container outside the pipeline CLI.

```bash
cd /path/to/video-dubbing-hatchet-serverless
DATA="$(pwd)/data"
MODELS="$(pwd)/data/models"
INPUT=sample.mp4
STEM=sample

# Preflight — input must be a real file (not a symlink to an absolute host path)
test -f "$DATA/$INPUT" || { echo "Missing $DATA/$INPUT — run: python scripts/download_samples.py nasa"; exit 1; }
ls -lh "$DATA/$INPUT"
```

> **Symlink pitfall:** An old `data/sample.mp4` symlink (e.g. → `nasa-coronagraph-75s.mp4`) breaks
> inside Docker because the target is a host path the container cannot see. Re-download with
> `python scripts/download_samples.py nasa` (writes a real file) or set
> `INPUT=nasa-coronagraph-75s.mp4`.

Default pipeline: **English source → Spanish dub** (defaults in `config.py`; override via `.env`).

#### Re-running a step (legacy flat paths)

The manual docker commands below write **flat files** next to the input (`sample.wav`, etc.) —
not the `runs/{run_id}/` layout used by `python -m pipeline run`. Prefer the pipeline CLI;
use these only for quick container smoke tests.

**Step 1 — Extract audio** (pre-built `lscr.io/linuxserver/ffmpeg:latest`)

```bash
docker run --rm -v "$DATA:/data" lscr.io/linuxserver/ffmpeg:latest \
  -i /data/$INPUT -vn -ac 1 -ar 16000 -y /data/${STEM}.wav
```

**Step 2 — Transcribe + align** (single container: ASR + WhisperX)

```bash
docker run --rm -v "$DATA:/data" -v "$MODELS:/data/models" video-dubbing-transcribe:local \
  /data/${STEM}.wav distil-large-v3 cpu
```

Writes `/data/sample.txt` and `/data/sample_aligned.json`.

**Step 3 — Translate** (NLLB-200, EN → ES)

```bash
docker run --rm -v "$DATA:/data" -v "$MODELS:/data/models" video-dubbing-translate:local \
  /data/${STEM}.txt es /data/${STEM}_translated.txt
```

Uses `device_map="auto"` — CUDA inside the container when `--gpus all` is passed, otherwise CPU.

**Step 4 — TTS** (Kokoro, Spanish)

```bash
docker run --rm -v "$DATA:/data" -v "$MODELS:/data/models" video-dubbing-tts:local \
  /data/${STEM}_translated.txt /data/${STEM}_dubbed.wav af_bella e
```

> **Note:** `af_bella e` = voice + Kokoro Spanish pipeline (defaults in `config.py`). Kokoro may warn
> about voice/lang mismatch — audio still works for local demos.

**Step 5 — Remux** (pre-built `lscr.io/linuxserver/ffmpeg:latest`)

```bash
docker run --rm -v "$DATA:/data" lscr.io/linuxserver/ffmpeg:latest \
  -i /data/$INPUT -i /data/${STEM}_dubbed.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest \
  -y /data/${STEM}_dubbed.mp4
```

---

## Level 1a — Run individual job scripts directly

Fastest iteration loop: edit a script and run it immediately with no Docker build.
Useful for testing a single step or debugging model output.

### Setup

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

Install per-step runtime dependencies as needed:

```bash
# Transcribe (includes alignment)
uv pip install faster-whisper whisperx

# Translate (already in pyproject.toml dependencies)
uv pip install transformers torch sentencepiece accelerate protobuf

# TTS
uv pip install kokoro soundfile numpy
```

### Run individual steps

```bash
# Model cache (host default: data/models — same tree Docker mounts at /data/models)
export MODEL_CACHE_DIR="$(pwd)/data/models"   # optional; this is the default when unset

# Transcribe + align (writes sample.txt and sample_aligned.json next to the .wav)
python src/jobs/transcribe.py data/sample.wav distil-large-v3 cpu

# Translate
python src/jobs/translate.py data/sample.txt de data/sample_translated.txt

# TTS
python src/jobs/tts.py data/sample_translated.txt data/sample_dubbed.wav af_bella a
```

---

## Level 2 — Cloud pipeline (Hatchet + Nebius)

The cloud workflow processes **1 to 100+ videos per Hatchet run**. Each stage fans out
parallel Nebius jobs (controlled by per-task `batch_size` and `max_concurrent` in
`config.py`). Partial progress survives preemption — already-written S3 artifacts are
skipped on retry.

### Resume / idempotency (bucket-only state)

No in-memory or Nebius VM state is required to resume. The only durable identifiers are:


| Input         | Role                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| `video_keys`  | Original input paths in the bucket (e.g. `sample_batch/001_sample.mp4`) |
| `run_id`      | Namespace for outputs under `runs/{run_id}/`                            |
| `force`       | `--force` on trigger — reprocess all files even when outputs exist      |
| `target_lang` | Must match on re-trigger (not encoded in output paths today)            |
| TTS lang      | `settings.tts.lang` in `config.py` — **not** auto-set from `target_lang` |


Each Hatchet task recomputes paths from `video_keys` + `run_id` via `build_run_items()`,
then `unprocessed_items()` skips files whose output artifact(s) already exist in S3
(unless `force=True`).
Nebius container jobs write outputs to `/data/…` (the bucket) **per file** before
continuing to the next.

**Automatic retry:** Hatchet retries failed tasks; Nebius preemptible jobs may exit
mid-chunk — partial files stay in S3 and are picked up on the next attempt.

**Manual re-trigger** (same input path and `--run-id`; completed outputs are skipped):

```bash
python -m hatchet.trigger run sample_batch/ --run-id batch-demo
```

**Force full rewrite** (ignore existing outputs, overwrite in bucket):

```bash
python -m hatchet.trigger run sample_batch/ --run-id batch-demo --force
```

Inputs are **bucket-relative paths** (same layout as `/data/…` inside jobs). A single
`.mp4` runs one video; a folder prefix scans for videos — no manifest file needed.

**Output layout** (inputs stay at their original keys, e.g. `sample_batch/001_sample.mp4`):

```text
runs/{run_id}/manifests/{task}.json  ← task config snapshot (written before each stage)
runs/{run_id}/reports/{task}.json    ← task result + timing (written after each stage)
runs/{run_id}/extract/{stem}.wav
runs/{run_id}/transcribe/{stem}.txt
runs/{run_id}/transcribe/{stem}_aligned.json
runs/{run_id}/translate/{stem}.txt
runs/{run_id}/tts/{stem}.wav
runs/{run_id}/remux/{stem}.mp4
```

Transcribe requires **both** transcript and alignment files before a file is considered done.

### Prerequisites


| Tool       | Install                                                                                                                                 |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `uv`       | `curl -Lsf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh)                                                           |
| Nebius CLI | `curl -sSL [https://storage.eu-north1.nebius.cloud/nebius-cli/install.sh](https://storage.eu-north1.nebius.cloud/nebius-cli/install.sh) |
| AWS CLI    | `pip install awscli` (for S3 operations)                                                                                                |


### Step 1 — Nebius project setup

1. Create a project at [console.nebius.ai](https://console.nebius.ai)
2. Note your **Project ID** and **Subnet ID** (Networking → Subnets)
3. Create an **IAM service account** and generate a key (IAM → Service accounts)
4. Create an **Object Storage bucket** and generate **access keys** (Storage → Buckets)

### Step 2 — Hatchet setup

Option A — Cloud (recommended for first run):

1. Sign up at [cloud.hatchet.run](https://cloud.hatchet.run)
2. Create a tenant → Settings → API Keys → generate token

Option B — Self-hosted:

```bash
docker compose -f docker/docker-compose.yml up -d
# Open http://localhost:8080 → create a token there
```

### Step 3 — Configure `.env`

```bash
cp .env.example .env
```

Fill in **credentials only** (9 variables):

```bash
HATCHET_CLIENT_TOKEN=...
NEBIUS_IAM_TOKEN=...
NEBIUS_PROJECT_ID=...
NEBIUS_SUBNET_ID=...
NEBIUS_BUCKET_ID=...
NEBIUS_BUCKET_NAME=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
```

All tunables (image URIs, models, batch sizes, timeouts) default in `src/pipeline/config.py`.
Override via nested env vars if needed, e.g. `TRANSCRIBE__MODEL=large-v3`.

### Step 4 — Install Python dependencies

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

### Step 5 — Upload sample video(s) and model cache

```bash
aws s3 cp data/sample.mp4 s3://$NEBIUS_BUCKET_NAME/input/sample.mp4 \
  --endpoint-url $AWS_ENDPOINT_URL

# One-time: sync pre-downloaded weights (local data/models → s3://bucket/models/)
aws s3 sync data/models/ s3://$NEBIUS_BUCKET_NAME/models/ \
  --endpoint-url $AWS_ENDPOINT_URL
```

Nebius jobs mount the bucket at `/data`, so `models/` in the bucket appears as `/data/models/` inside containers (same layout as local Docker).

Or download samples and upload:

```bash
python scripts/download_samples.py nasa --sample-size 10
aws s3 cp data/sample.mp4 s3://$NEBIUS_BUCKET_NAME/input/sample.mp4 \
  --endpoint-url $AWS_ENDPOINT_URL
aws s3 sync data/sample_batch/ s3://$NEBIUS_BUCKET_NAME/sample_batch/ \
  --endpoint-url $AWS_ENDPOINT_URL
python -m hatchet.trigger run sample_batch/ --run-id batch-demo
```

### Step 6 — Build and push container images

> Nebius jobs run on **Linux/amd64**. Build on a Nebius CPU VM or use `--platform linux/amd64`.

```bash
export DOCKER_BUILDKIT=1
export REGISTRY=your-dockerhub-user   # must match image URIs in config.py

docker build --platform linux/amd64 -f docker/base-cuda.Dockerfile \
  -t video-dubbing-base:v0.1.0 .

docker build --platform linux/amd64 -f docker/transcribe.Dockerfile \
  --build-arg BASE_IMAGE=video-dubbing-base:v0.1.0 \
  --build-arg USE_CUDA_SOURCES=1 \
  -t video-dubbing-transcribe:v0.1.0 .

docker tag video-dubbing-transcribe:v0.1.0 $REGISTRY/video-dubbing-transcribe:v0.1.0
docker push $REGISTRY/video-dubbing-transcribe:v0.1.0

# repeat for translate and tts; set TRANSCRIBE__IMAGE / TRANSLATE__IMAGE / TTS__IMAGE in config
```

Or skip this step and use the pre-built defaults in `config.py`.

### Step 7 — Start the worker

```bash
# Terminal 1 — keep running
python -m hatchet.worker
```

### Step 8 — Trigger a run

```bash
# Single video (bucket-relative path under /data)
python -m hatchet.trigger run input/sample.mp4 --run-id demo-01

# All videos under a prefix
python -m hatchet.trigger run sample_batch/ --run-id batch-001

# Force rewrite (ignore existing outputs)
python -m hatchet.trigger run sample_batch/ --run-id batch-001 --force
```

Optional: `--batch-id` controls Hatchet concurrency grouping (default `default`).
Use `--force` to reprocess every file even when outputs already exist under `runs/{run_id}/`.
Host paths like `data/sample_batch/` are normalized to `sample_batch/` automatically.

### Step 9 — Retrieve output

```bash
aws s3 ls s3://$NEBIUS_BUCKET_NAME/runs/demo-01/remux/ --endpoint-url $AWS_ENDPOINT_URL

aws s3 cp s3://$NEBIUS_BUCKET_NAME/runs/demo-01/remux/sample.mp4 data/output.mp4 \
  --endpoint-url $AWS_ENDPOINT_URL
```

Final videos: `runs/{run_id}/remux/{stem}.mp4` for each input video.

---

## Pipeline DAG

```mermaid
flowchart LR
    extract_audio --> transcribe
    transcribe --> translate_text
    translate_text --> synthesize_tts
    synthesize_tts --> remux_video
```



Each task is a **fan-out** of parallel Nebius jobs. Extract and remux run shell ffmpeg inside the pre-built image `lscr.io/linuxserver/ffmpeg:latest` (`settings.extract.image` / `settings.remux.image` — no custom Dockerfile in this repo). **Every stage** writes its own manifest before running and its own report on completion — whether you run all stages together, one stage at a time, or trigger tasks individually via Hatchet.

Transcribe, translate, and TTS containers read the task manifest at `runs/{run_id}/manifests/{task}.json` plus a **chunk index**. They discover input files from the **upstream task report** (`runs/{run_id}/reports/{upstream}.json`) or by scanning the upstream artifact directory; per-file output paths follow the `runs/{run_id}/` layout.

**Task manifest** (`metadata.py` / `jobs/run_manifest.py`) — one file per stage invocation, e.g. `runs/demo/manifests/transcribe.json`:

```json
{
  "task": "transcribe",
  "run_id": "demo",
  "batch_id": "local",
  "input_count": 42,
  "target_lang": "es",
  "force": false,
  "created_at": "2026-05-21T12:00:00+00:00",
  "executor": "local-docker",
  "config": {
    "image": "mnrozhkov/nebius-transcribe:v0.1.0",
    "model": "distil-large-v3",
    "device": "cpu",
    "batch_size": 10,
    "max_concurrent": 10,
    "compute": { "gpu": false, "platform": "cpu-e2", "preset": "4vcpu-16gb" }
  }
}
```

Container argv: `/data/runs/{run_id}/manifests/transcribe.json 0` (chunk index selects the slice at `config.batch_size * chunk_idx`).

**Task report** (`metadata.py`) — one file per completed stage, e.g. `runs/demo/reports/transcribe.json`:

```json
{
  "task": "transcribe",
  "run_id": "demo",
  "status": "completed",
  "started_at": "2026-05-21T12:00:00+00:00",
  "completed_at": "2026-05-21T12:05:12+00:00",
  "manifest_key": "runs/demo/manifests/transcribe.json",
  "wall_s": 312.5,
  "timing": { "task": "transcribe", "total_files": 3, "processed_files": 3, "chunks": [...] },
  "outputs": { "transcript_keys": ["..."], "aligned_keys": ["..."] },
  "error": null
}
```

Stage status is `completed`, `skipped` (all outputs already existed), or `failed`. Each stage is independent — running only `extract` creates `manifests/extract.json` and `reports/extract.json` without touching other tasks.

---

## Configuration reference

Settings live in `src/pipeline/config.py`. Task defaults are defined on nested pydantic
models (`ExtractConfig`, `TranscribeConfig`, …). **`Settings.__init__` loads `.env` lazily**
(on first `get_settings()` call) — local pipeline runs do not require Nebius/Hatchet credentials.

Cloud-only code paths call `require_cloud_setting()` when Nebius or S3 is actually used.

Access in code as `settings.transcribe.model`, `settings.tts.voice`, etc.


| Step                      | Config path                                       | Default                             |
| ------------------------- | ------------------------------------------------- | ----------------------------------- |
| Extract / remux image     | `settings.extract.image` / `settings.remux.image` | `lscr.io/linuxserver/ffmpeg:latest` |
| Transcription model       | `settings.transcribe.model`                       | `distil-large-v3`                   |
| Transcription compute     | `settings.transcribe.compute`                     | L40S preemptible                    |
| Translation model         | `settings.translate.model`                        | `facebook/nllb-200-distilled-1.3B`  |
| Translation compute       | `settings.translate.compute`                      | CPU `8vcpu-32gb` (GPU optional)     |
| TTS voice                 | `settings.tts.voice`                              | `af_bella`                          |
| TTS lang (Kokoro)         | `settings.tts.lang`                               | `e` (Spanish pipeline)              |
| Target language (NLLB)  | `settings.target_lang` / `TARGET_LANG`            | `es`                                |
| TTS compute               | `settings.tts.compute`                            | L40S preemptible                    |
| Batch size (per task)     | `settings.<task>.batch_size`                      | `10` (50 for extract/remux)         |
| Max concurrent (per task) | `settings.<task>.max_concurrent`                  | varies                              |


Per-task Nebius platform/preset (`settings.<task>.compute`):


| Field         | Meaning                                                         |
| ------------- | --------------------------------------------------------------- |
| `gpu`         | Whether the job requests GPU                                    |
| `platform`    | Nebius platform id, e.g. `gpu-l40s-d`, `gpu-h200-sxm`, `cpu-e2` |
| `preset`      | Nebius preset id, e.g. `1gpu-16vcpu-96gb`, `8vcpu-32gb`         |
| `preemptible` | Use preemptible capacity (GPU only)                             |


Experiment with platform + batch size — edit `config.py` or override via env:

```bash
# H200 transcribe, smaller batches
TRANSCRIBE__COMPUTE__PLATFORM=gpu-h200-sxm
TRANSCRIBE__COMPUTE__PRESET=1gpu-16vcpu-200gb
TRANSCRIBE__COMPUTE__PREEMPTIBLE=false
TRANSCRIBE__BATCH_SIZE=5

# GPU translate (CUDA image; set compute.gpu=true on Nebius)
TRANSLATE__COMPUTE__GPU=true
TRANSLATE__COMPUTE__PLATFORM=gpu-l40s-d
TRANSLATE__COMPUTE__PRESET=1gpu-16vcpu-96gb
TRANSLATE__BATCH_SIZE=8
```

After each Hatchet task completes, timing is logged and returned in task output under `timing`:

```text
[timing] task=transcribe platform=gpu-l40s-d preset=1gpu-16vcpu-96gb ...
         batch_size=10 files=20/20 wall_s=842.3 per_file_s=42.1 chunk_times=[280.1, ...]
```

Use Hatchet Traces + worker logs to compare runs and find the best cost/time tradeoff.

To add a new AI step: `src/jobs/<step>.py` + `docker/<step>.Dockerfile` + stage in `src/pipeline/stages/` and a task in `src/hatchet/workflow.py`.

---

## Rebuild after code changes

```bash
# Edited src/jobs/transcribe.py → rebuild task image only (base cached)
docker build -f docker/transcribe.Dockerfile -t video-dubbing-transcribe:local .

# Edited [dependency-groups].transcribe in pyproject.toml → rebuild transcribe image
# Edited cuda-base group → rebuild base + all task images

# Edited src/hatchet/workflow.py, src/pipeline/stages/, or config.py → no rebuild needed
python -m hatchet.worker   # cloud
python -m pipeline run …   # local — no rebuild
```

---

## Troubleshooting

### Level 1b — Docker local (`python -m pipeline run`)


| Symptom                                           | Fix                                                                                                                                                                                                |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Got unexpected extra argument (run)`             | Use `python -m pipeline run …` (not `python -m pipeline sample.mp4`)                                                                                                                               |
| `Video file not found`                            | Path is relative to `data/` — use `data/sample.mp4` or `sample.mp4`; check `--data-dir`                                                                                                              |
| `COPY src/jobs/... not found`                     | Build from **repo root**; build base before task images                                                                                                                                            |
| `failed to resolve BASE_IMAGE`                    | Build base first: `docker/base-cpu.Dockerfile` (Mac) or `docker/base-cuda.Dockerfile` (Linux GPU) → tag `video-dubbing-base:local`                                                                 |
| `libcudart.so.13: cannot open shared object file` | Built with CUDA base on a machine without NVIDIA — rebuild with `docker/base-cpu.Dockerfile`, then rebuild task images                                                                             |
| `Error opening input file /data/sample.mp4`       | Input must exist under `data/`. If `sample.mp4` is a **symlink**, re-download (`python scripts/download_samples.py nasa`)                                                                            |
| `Syntax error: end of file unexpected` (ffmpeg)   | Fixed in current code — update repo; was a shell-quoting bug in local Docker executor                                                                                                              |
| Transcribe very slow                              | Expected on CPU — pass `--device cpu` (default); ~10× slower without GPU                                                                                                                           |
| Stage fails — missing prior output                | Run earlier stages first, or run full pipeline without `--stage`                                                                                                                                   |
| Out of memory (translate)                         | Smaller model: `TRANSLATE__MODEL=facebook/nllb-200-distilled-600M`, or GPU: `TRANSLATE__COMPUTE__GPU=true`                                                                                         |
| Kokoro non-English quality poor                   | Expected — Kokoro is English-primary; best results with `TTS__LANG=a`                                                                                                                              |
| Image build fails on Apple Silicon                | Use `docker/base-cpu.Dockerfile` for local dev. Use `base-cuda` + `--platform linux/amd64` only when pushing to Nebius                                                                             |
| `ValidationError` for Nebius fields on local run  | Fixed — cloud creds are optional until cloud paths run; update repo if you still see this                                                                                                          |


### Level 1a — Python scripts


| Symptom                               | Fix                                      |
| ------------------------------------- | ---------------------------------------- |
| `ModuleNotFoundError: pipeline`       | `uv pip install -e .`                    |
| `ModuleNotFoundError: faster_whisper` | `uv pip install faster-whisper whisperx` |
| `ModuleNotFoundError: kokoro`         | `uv pip install kokoro soundfile numpy`  |


### Level 2 — Cloud


| Symptom                               | Fix                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------- |
| Worker exits immediately              | Check `HATCHET_CLIENT_TOKEN` is valid                                      |
| Job stuck in `PENDING`                | Check Nebius quota; verify `NEBIUS_PROJECT_ID` and `NEBIUS_SUBNET_ID`      |
| Job `ERROR` / preempted               | Expected — Hatchet retries automatically (`settings.transcribe.retries=3`) |
| Output file missing after `COMPLETED` | Check `NEBIUS_BUCKET_NAME` and `AWS_`* credentials                         |
| Batch retry re-processes some files   | Normal — only missing artifacts are re-run; check S3 for partial outputs   |
| AMD64 build required                  | Build on a Nebius CPU VM or `docker buildx build --platform linux/amd64`   |


