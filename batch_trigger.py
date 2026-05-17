"""
Batch trigger - fire N dubbing runs staggered over time.

Usage:
    python batch_trigger.py --video input/my-video.mp4 --count 10 --interval 20
    python batch_trigger.py --video input/my-video.mp4 --count 10 --interval 20 --lang de
"""

import argparse
import time

from pipeline.workflow import DubbingInput, workflow


def main():
    parser = argparse.ArgumentParser(description="Trigger a batch of dubbing runs")
    parser.add_argument("--video", required=True, help="Object storage key for the input video")
    parser.add_argument("--count", type=int, default=5, help="Number of runs to trigger")
    parser.add_argument("--interval", type=int, default=20, help="Seconds between each run")
    parser.add_argument("--lang", default="de", help="Target language code")
    args = parser.parse_args()

    print(f"Triggering {args.count} runs staggered {args.interval}s apart\n")

    for i in range(1, args.count + 1):
        run_id = f"demo-{i:02d}"

        input_data = DubbingInput(
            video_key=args.video,
            target_lang=args.lang,
            run_id=run_id,
        )

        ref = workflow.run_no_wait(input_data)
        print(f"[{i:02d}/{args.count}] Started {run_id} -> run ID: {ref.workflow_run_id}")

        if i < args.count:
            print(f"       Waiting {args.interval}s before next run...")
            time.sleep(args.interval)

    print(f"\nAll {args.count} runs triggered!")
    print("Watch them live: https://cloud.onhatchet.run")


if __name__ == "__main__":
    main()
