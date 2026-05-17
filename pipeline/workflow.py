"""
Video dubbing pipeline - fully cloud, Hatchet workflow (SDK v1.33.x).

Your machine only orchestrates - all data stays in object storage.

Steps:
  1. extract_audio      (Nebius CPU job) - ffmpeg webm/mp4 to wav in bucket
  2. transcribe_whisper (Nebius GPU job, preemptible) - Whisper transcript.txt
  3. translate_text     (Nebius CPU job) - MADLAD-400 translated.txt
  4. synthesize_tts     (Nebius GPU job, preemptible) - Coqui TTS dubbed.wav
  5. remux_video        (Nebius CPU job) - ffmpeg merge, dubbed video in bucket
"""

import logging
from pathlib import Path

from hatchet_sdk import Context, Hatchet
from pydantic import BaseModel

from pipeline.config import settings
from pipeline.local import object_exists
from pipeline.nebius import create_and_wait

logger = logging.getLogger(__name__)

hatchet = Hatchet(debug=True)


class DubbingInput(BaseModel):
    video_key: str
    target_lang: str = settings.target_lang
    run_id: str = "demo"


workflow = hatchet.workflow(
    name="video-dubbing-pipeline",
    on_events=["dubbing:start"],
    input_validator=DubbingInput,
)

FFMPEG_IMAGE = "lscr.io/linuxserver/ffmpeg:latest"


@workflow.task(execution_timeout="900s")
async def extract_audio(input: DubbingInput, ctx: Context) -> dict:
    """Run ffmpeg on Nebius CPU job - video stays in the bucket."""
    stem = Path(input.video_key).stem
    audio_key = f"{stem}_{input.run_id}.wav"

    ctx.log(f"Launching ffmpeg CPU job: {input.video_key} -> {audio_key}")

    shell_cmd = (
        f"ffmpeg -i /data/{input.video_key} -vn -ac 1 -ar 16000 -y /tmp/audio.wav && "
        f"mkdir -p /data/audio && cp /tmp/audio.wav /data/{audio_key}"
    )

    await create_and_wait(
        name=f"ffmpeg-extract-{input.run_id}",
        image=FFMPEG_IMAGE,
        container_command="sh",
        args=f'-c "{shell_cmd}"',
        gpu=False,
        preemptible=False,
    )

    if not object_exists(audio_key):
        raise RuntimeError(f"ffmpeg job completed but audio not found at {audio_key}")

    ctx.log(f"Audio extracted: {audio_key}")
    return {"audio_key": audio_key, "stem": stem}


@workflow.task(parents=[extract_audio], execution_timeout="2700s", retries=3)
async def transcribe_whisper(input: DubbingInput, ctx: Context) -> dict:
    """Whisper on preemptible L40S - raises on preemption, Hatchet retries."""
    prev = ctx.task_output(extract_audio)
    audio_key: str = prev["audio_key"]
    stem: str = prev["stem"]

    ctx.log("Launching Whisper job on preemptible L40S")

    await create_and_wait(
        name=f"whisper-{input.run_id}",
        image=settings.whisper_image,
        args=f"/data/{audio_key} {settings.whisper_model} cuda",
        gpu=True,
        preemptible=True,
    )

    transcript_key = f"{stem}_{input.run_id}.txt"
    if not object_exists(transcript_key):
        raise RuntimeError(f"Transcript not found at {transcript_key}")

    ctx.log(f"Transcript ready: {transcript_key}")
    return {"transcript_key": transcript_key, "stem": stem}


@workflow.task(parents=[transcribe_whisper], execution_timeout="1800s")
async def translate_text_step(input: DubbingInput, ctx: Context) -> dict:
    """MADLAD-400 translation on a Nebius CPU job."""
    prev = ctx.task_output(transcribe_whisper)
    transcript_key: str = prev["transcript_key"]
    stem: str = prev["stem"]

    translated_key = f"{stem}_{input.run_id}_translated.txt"
    ctx.log(f"Launching translation CPU job, target: {input.target_lang}")

    await create_and_wait(
        name=f"translate-{input.run_id}",
        image=settings.translate_image,
        container_command="python3",
        args=f"/translate.py /data/{transcript_key} {input.target_lang} /data/{translated_key}",
        gpu=False,
        preemptible=False,
    )

    if not object_exists(translated_key):
        raise RuntimeError(f"Translation not found at {translated_key}")

    ctx.log(f"Translation done: {translated_key}")
    return {"translated_key": translated_key, "stem": stem}


@workflow.task(parents=[translate_text_step], execution_timeout="2700s", retries=3)
async def synthesize_tts(input: DubbingInput, ctx: Context) -> dict:
    """Coqui TTS on preemptible L40S."""
    prev = ctx.task_output(translate_text_step)
    translated_key: str = prev["translated_key"]
    stem: str = prev["stem"]

    dubbed_key = f"{stem}_{input.run_id}_dubbed.wav"
    ctx.log("Launching TTS job on preemptible L40S")

    await create_and_wait(
        name=f"tts-{input.run_id}",
        image=settings.tts_image,
        args=f"/data/{translated_key} {settings.tts_model} /data/{dubbed_key}",
        gpu=True,
        preemptible=True,
    )

    if not object_exists(dubbed_key):
        raise RuntimeError(f"Dubbed audio not found at {dubbed_key}")

    ctx.log(f"Dubbed audio ready: {dubbed_key}")
    return {"dubbed_key": dubbed_key, "stem": stem}


@workflow.task(parents=[synthesize_tts], execution_timeout="900s")
async def remux_video(input: DubbingInput, ctx: Context) -> dict:
    """Merge original video + dubbed audio on Nebius CPU job."""
    prev = ctx.task_output(synthesize_tts)
    dubbed_key: str = prev["dubbed_key"]
    stem: str = prev["stem"]

    output_key = f"{stem}_{input.run_id}_dubbed.mp4"
    ctx.log("Launching ffmpeg remux CPU job")

    shell_cmd = (
        f"ffmpeg -i /data/{input.video_key} -i /data/{dubbed_key} "
        f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest -y /tmp/output.mp4 && "
        f"cp /tmp/output.mp4 /data/{output_key}"
    )

    await create_and_wait(
        name=f"ffmpeg-remux-{input.run_id}",
        image=FFMPEG_IMAGE,
        container_command="sh",
        args=f'-c "{shell_cmd}"',
        gpu=False,
        preemptible=False,
    )

    if not object_exists(output_key):
        raise RuntimeError(f"Output video not found at {output_key}")

    ctx.log(f"Done! Dubbed video ready: {output_key}")
    return {"output_key": output_key, "status": "completed"}
