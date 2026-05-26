#!/usr/bin/env bash
# Test all 5 pipeline stage images locally against a sample batch.
#
# Pulls the same registry images Nebius uses (mnrozhkov/video-dubbing-<stage>:v0.2.0)
# and runs them sequentially against the local data/ directory — same images,
# same manifests, same code paths the cloud sees, just with /data bind-mounted
# from the host instead of FUSE-mounted from S3.
#
# What this catches (no GPU needed):
#   - Image pull / COPY layer regressions
#   - Module-import errors (e.g. ctranslate2 wheel ABI vs base cuDNN — bug #7
#     surfaces at model-load if device=cuda; this script defaults to device=cpu)
#   - Container-side path resolution (auto_configure_data_root on /data)
#   - Manifest schema mismatches between orchestrator and container
#   - The full DAG: extract → transcribe → translate → tts → remux against a
#     multi-file batch
#
# What this does NOT catch:
#   - FUSE-specific bugs (#3 staged_write, #5 hf_xet) — local fs is POSIX
#   - GPU-runtime errors — we force --device cpu so transcribe/translate/tts
#     skip cuda. To exercise GPU paths, run on a Linux+NVIDIA host and pass
#     --device cuda plus add --gpus all on docker run.
#
# Pre-reqs:
#   - data/sample_batch/ has at least one .mp4 (already in the repo)
#   - data/models/ pre-populated, or HF can reach the network from inside the
#     container (recommended: run `python scripts/download_models.py all` once)
#   - Docker Desktop with BuildKit (default on macOS)
#
# Usage:
#   scripts/docker_test.sh                            # all 5 stages, sample_batch/, device=cpu
#   scripts/docker_test.sh --run-id myrun
#   scripts/docker_test.sh --source sample_file/sample.mp4
#   scripts/docker_test.sh --stage extract            # single stage only
#   scripts/docker_test.sh --force                    # reprocess all files even if outputs exist
#   VERSION=v0.2.0 scripts/docker_test.sh             # pin a tag
#
# Apple Silicon note: --platform linux/amd64 runs under QEMU emulation. Expect
# ~10× slower than native. transcribe on a 1-min sample can take 5-10 min.

set -euo pipefail

REGISTRY="${REGISTRY:-mnrozhkov}"
VERSION="${VERSION:-v0.2.0}"
PLATFORM="${PLATFORM:-linux/amd64}"
SOURCE="${SOURCE:-sample_batch/}"
RUN_ID="${RUN_ID:-docker-test-$(date +%s)}"
DEVICE="${DEVICE:-cpu}"
FORCE=0
STAGES=(extract transcribe translate tts remux)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"

print_help() {
    sed -n '2,40p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --source)   SOURCE="$2"; shift 2 ;;
        --run-id)   RUN_ID="$2"; shift 2 ;;
        --version)  VERSION="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --stage)    STAGES=("$2"); shift 2 ;;
        --force)    FORCE=1; shift ;;
        -h|--help)  print_help 0 ;;
        *)          echo "Unknown flag: $1" >&2; print_help 1 ;;
    esac
done

cd "$REPO_ROOT"

log()  { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; }

log "docker_test.sh"
echo "  registry  ${REGISTRY}"
echo "  version   ${VERSION}"
echo "  platform  ${PLATFORM}"
echo "  source    ${SOURCE}"
echo "  run_id    ${RUN_ID}"
echo "  device    ${DEVICE}"
echo "  force     ${FORCE}"
echo "  stages    ${STAGES[*]}"
echo "  data_dir  ${DATA_DIR}"

# ── Pull all images upfront ──────────────────────────────────────────────────
log "Pulling images"
for stage in "${STAGES[@]}"; do
    image="${REGISTRY}/video-dubbing-${stage}:${VERSION}"
    docker pull --platform "${PLATFORM}" "${image}" >/dev/null
    ok "${image}"
done

# ── Run each stage sequentially ──────────────────────────────────────────────
TIMINGS_KEYS=()
TIMINGS_VALS=()
PIPELINE_T0=$SECONDS

for stage in "${STAGES[@]}"; do
    image="${REGISTRY}/video-dubbing-${stage}:${VERSION}"
    manifest_host="${DATA_DIR}/runs/${RUN_ID}/manifests/${stage}.json"
    manifest_ctr="/data/runs/${RUN_ID}/manifests/${stage}.json"

    log "Stage: ${stage}"

    # Prepare manifest. Extract takes --source; downstream stages derive their
    # inputs from the upstream report on disk.
    force_flag=""
    [ "${FORCE}" = "1" ] && force_flag="--force"

    if [ "${stage}" = "extract" ]; then
        python -m pipeline prepare-manifest "${stage}" \
            --run-id "${RUN_ID}" --source "${SOURCE}" --device "${DEVICE}" ${force_flag}
    else
        python -m pipeline prepare-manifest "${stage}" \
            --run-id "${RUN_ID}" --device "${DEVICE}" ${force_flag}
    fi

    if [ ! -f "${manifest_host}" ]; then
        fail "manifest not created at ${manifest_host}"
        exit 1
    fi
    ok "manifest: runs/${RUN_ID}/manifests/${stage}.json"

    # Run the registry image against the host data dir.
    # NOTE: no --gpus flag → GPU work is skipped (--device cpu in manifest).
    STAGE_T0=$SECONDS
    if docker run --rm --platform "${PLATFORM}" \
        -v "${DATA_DIR}:/data" \
        "${image}" \
        "${manifest_ctr}"; then
        elapsed=$((SECONDS - STAGE_T0))
        ok "${stage} completed in ${elapsed}s"
        TIMINGS_KEYS+=("${stage}")
        TIMINGS_VALS+=("${elapsed}")
    else
        elapsed=$((SECONDS - STAGE_T0))
        fail "${stage} failed after ${elapsed}s — see container output above"
        exit 1
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$((SECONDS - PIPELINE_T0))
log "Summary — total ${TOTAL}s"
for i in "${!TIMINGS_KEYS[@]}"; do
    pct=0
    [ "$TOTAL" -gt 0 ] && pct=$(( TIMINGS_VALS[i] * 100 / TOTAL ))
    printf '  ✓ %-11s %5ss  (%2s%%)\n' "${TIMINGS_KEYS[$i]}" "${TIMINGS_VALS[$i]}" "${pct}"
done

echo ""
echo "Outputs under: ${DATA_DIR}/runs/${RUN_ID}/"
echo "  manifests/  reports/  extract/  transcribe/  translate/  tts/  remux/"
