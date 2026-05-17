"""
Trigger a video dubbing pipeline run.

Usage:
    python trigger.py --video input/my-video.mp4
    python trigger.py --video input/my-video.mp4 --lang fr
    python trigger.py --video input/my-video.mp4 --lang de --run-id demo-001
"""

import argparse

from pipeline.workflow import DubbingInput, workflow


def main():
    parser = argparse.ArgumentParser(description="Trigger a video dubbing run")
    parser.add_argument("--video", required=True, help="Object storage key for the input video")
    parser.add_argument("--lang", default="de", help="Target language code (e.g. de, fr, es)")
    parser.add_argument("--run-id", default="demo", help="Optional label for this run")
    args = parser.parse_args()

    input_data = DubbingInput(
        video_key=args.video,
        target_lang=args.lang,
        run_id=args.run_id,
    )

    print(f"Triggering pipeline:")
    print(f"  video:   {input_data.video_key}")
    print(f"  lang:    {input_data.target_lang}")
    print(f"  run_id:  {input_data.run_id}")

    ref = workflow.run_no_wait(input_data)
    print(f"\nRun started! ID: {ref.workflow_run_id}")
    print("Watch it live: https://cloud.onhatchet.run")


if __name__ == "__main__":
    main()
