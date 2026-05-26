#!/usr/bin/env bash
# Build all task images for the dubbing pipeline and (optionally) push to Docker Hub.
#
# Tags produced:
#   <REGISTRY>/video-dubbing-<task>:<VERSION>
#   <REGISTRY>/video-dubbing-<task>:latest        (only when --push)
#
# Defaults:
#   REGISTRY=mnrozhkov
#   VERSION=v0.1.0
#   PLATFORM=linux/amd64           (Nebius targets amd64)
#   BASE=cuda                      (use 'cpu' for Mac/local-only builds)
#   PUSH=0                         (set --push to push to registry)
#
# Usage:
#   scripts/docker_build.sh                                  # build all (cuda base), no push
#   scripts/docker_build.sh --push                           # build + push all
#   scripts/docker_build.sh --task transcribe --push         # one task
#   scripts/docker_build.sh --skip-base                      # skip base; reuse cached :local, else cached registry tag, else pull mnrozhkov/video-dubbing-base:<VERSION>
#   scripts/docker_build.sh --base cpu --no-platform         # local Mac smoke build (native arch, cached)
#   VERSION=v0.2.0 scripts/docker_build.sh --push            # bump version
#   REGISTRY=otheraccount scripts/docker_build.sh --push     # different Docker Hub user
#
# Pre-reqs:
#   - Docker BuildKit (default in Docker Desktop)
#   - `docker login` if pushing
#   - Run from the repo root (Dockerfiles COPY src/, scripts/, pyproject.toml from CWD)
#
# Caching notes:
#   - `--platform linux/amd64` on Apple Silicon uses BuildKit's cross-arch cache, which Docker
#     Desktop evicts more aggressively than the native cache. If you don't need amd64 (i.e. you're
#     iterating locally, not pushing to Nebius), pass `--no-platform` to build native arm64 — much
#     better layer reuse across runs.
#   - The base image is heavy (torch + uv sync). Pass `--skip-base` if you haven't changed
#     pyproject.toml, scripts/uv_sync_task.sh, or the base Dockerfile since the last build.
#   - To inspect cache: `docker buildx du`. To free space: `docker buildx prune`.

set -euo pipefail

REGISTRY="${REGISTRY:-mnrozhkov}"
VERSION="${VERSION:-v0.2.0}"
BASE="${BASE:-cuda}"                 # cuda | cpu
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH=0
SKIP_BASE=0
TASKS=()
USE_PLATFORM=1

print_usage() {
    sed -n '1,42p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --push)         PUSH=1; shift ;;
        --no-push)      PUSH=0; shift ;;
        --skip-base)    SKIP_BASE=1; shift ;;
        --task)         TASKS+=("$2"); shift 2 ;;
        --base)         BASE="$2"; shift 2 ;;
        --version)      VERSION="$2"; shift 2 ;;
        --registry)     REGISTRY="$2"; shift 2 ;;
        --platform)     PLATFORM="$2"; shift 2 ;;
        --no-platform)  USE_PLATFORM=0; shift ;;
        -h|--help)      print_usage 0 ;;
        *)              echo "Unknown flag: $1" >&2; print_usage 1 ;;
    esac
done

if [ "${#TASKS[@]}" -eq 0 ]; then
    TASKS=(extract transcribe translate tts remux)
fi

case "$BASE" in
    cuda) BASE_DOCKERFILE="docker/base-cuda.Dockerfile"; USE_CUDA_SOURCES=1 ;;
    cpu)  BASE_DOCKERFILE="docker/base-cpu.Dockerfile";  USE_CUDA_SOURCES=0 ;;
    *)    echo "Unknown --base '$BASE' (expected: cuda | cpu)" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$BASE_DOCKERFILE" ]; then
    echo "Missing $BASE_DOCKERFILE (run this from the repo root)" >&2
    exit 1
fi

BASE_LOCAL_TAG="video-dubbing-base:local"
BASE_REGISTRY_TAG="${REGISTRY}/video-dubbing-base:${VERSION}"

platform_flag=()
if [ "$USE_PLATFORM" -eq 1 ]; then
    platform_flag=(--platform "$PLATFORM")
fi

export DOCKER_BUILDKIT=1

log() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

log "Settings: registry=${REGISTRY} version=${VERSION} base=${BASE} platform=${USE_PLATFORM:+$PLATFORM} push=${PUSH} tasks=${TASKS[*]}"

# ── Base image ────────────────────────────────────────────────────────────────
if [ "$SKIP_BASE" -eq 1 ]; then
    # Resolve a usable :local tag from one of (in order):
    #   1. already-cached :local
    #   2. already-cached registry-versioned tag → retag as :local
    #   3. pull registry-versioned tag from Docker Hub → retag as :local
    if docker image inspect "$BASE_LOCAL_TAG" >/dev/null 2>&1; then
        log "Skipping base build (--skip-base); using cached ${BASE_LOCAL_TAG}"
    elif docker image inspect "$BASE_REGISTRY_TAG" >/dev/null 2>&1; then
        log "Found ${BASE_REGISTRY_TAG} locally; retagging as ${BASE_LOCAL_TAG}"
        docker tag "$BASE_REGISTRY_TAG" "$BASE_LOCAL_TAG"
    else
        log "${BASE_LOCAL_TAG} not cached; pulling ${BASE_REGISTRY_TAG} from registry"
        if ! docker pull "$BASE_REGISTRY_TAG"; then
            echo "" >&2
            echo "--skip-base passed but no base image available:" >&2
            echo "  - ${BASE_LOCAL_TAG} not cached locally" >&2
            echo "  - ${BASE_REGISTRY_TAG} could not be pulled from the registry" >&2
            echo "" >&2
            echo "Fix one of:" >&2
            echo "  - Re-run without --skip-base to rebuild the base from source" >&2
            echo "  - Push ${BASE_REGISTRY_TAG} from a machine that has it" >&2
            echo "  - Check VERSION (currently ${VERSION}) matches a published base tag" >&2
            exit 1
        fi
        docker tag "$BASE_REGISTRY_TAG" "$BASE_LOCAL_TAG"
    fi
else
    log "Building base image: ${BASE_LOCAL_TAG} (${BASE_DOCKERFILE})"
    docker build "${platform_flag[@]}" \
        -f "$BASE_DOCKERFILE" \
        -t "$BASE_LOCAL_TAG" \
        .

    # Also tag the base under the registry so task builds can pin a stable BASE_IMAGE
    # when pushing (the tag is local-only until you push it explicitly).
    docker tag "$BASE_LOCAL_TAG" "$BASE_REGISTRY_TAG"
    if [ "$PUSH" -eq 1 ]; then
        log "Pushing base image: ${BASE_REGISTRY_TAG}"
        docker push "$BASE_REGISTRY_TAG"
    fi
fi

# ── Task images ───────────────────────────────────────────────────────────────
for task in "${TASKS[@]}"; do
    dockerfile="docker/${task}.Dockerfile"
    if [ ! -f "$dockerfile" ]; then
        echo "Missing $dockerfile — skipping task '$task'" >&2
        continue
    fi

    local_tag="video-dubbing-${task}:local"
    version_tag="${REGISTRY}/video-dubbing-${task}:${VERSION}"
    latest_tag="${REGISTRY}/video-dubbing-${task}:latest"

    log "Building task image: ${version_tag}"
    docker build "${platform_flag[@]}" \
        -f "$dockerfile" \
        --build-arg "BASE_IMAGE=${BASE_LOCAL_TAG}" \
        --build-arg "USE_CUDA_SOURCES=${USE_CUDA_SOURCES}" \
        -t "$local_tag" \
        -t "$version_tag" \
        -t "$latest_tag" \
        .

    if [ "$PUSH" -eq 1 ]; then
        log "Pushing ${version_tag}"
        docker push "$version_tag"
        log "Pushing ${latest_tag}"
        docker push "$latest_tag"
    fi
done

log "Done."
if [ "$PUSH" -eq 0 ]; then
    echo "Built locally. Re-run with --push to publish to ${REGISTRY}."
fi
