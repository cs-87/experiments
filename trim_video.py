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
    args = parser.parse_args()

    cmd = [
        "ffmpeg", "-y",
        "-i", args.input,
        "-frames:v", str(args.n),
        "-c:v", "ffv1",
        "-an",
        args.output,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
