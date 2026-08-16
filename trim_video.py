#!/usr/bin/env python3
"""Trim a video to its first N frames, losslessly, no re-timing/interpolation."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input video path")
    parser.add_argument("n", type=int, help="number of frames to keep")
    parser.add_argument("output", help="output video path (use .mkv)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="frame to start the trim at (default 0)")
    args = parser.parse_args()

    cmd = ["ffmpeg", "-y", "-i", args.input]
    if args.start_frame:
        # Frame-accurate seek: -vf select is placed after -i, unlike a fast
        # keyframe-snapping -ss before -i, which would drift from the offset
        # the caller measured frame counts against.
        cmd += ["-vf", f"select='gte(n\\,{args.start_frame})'", "-vsync", "0"]
    cmd += [
        "-frames:v", str(args.n),
        "-c:v", "ffv1",
        "-an",
        args.output,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
