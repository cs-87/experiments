from utils.video import Video_IO
from enum import Enum
from utils.bit import get_bit_string, uint32_to_hex, hex_to_uint32
from tqdm import tqdm
import numpy as np
import random
from utils.dwt import get_dwt_coeff, reconstruct_frame
from utils.patch import get_sift_patches
import cv2

from src.spread_spectrum.prn import BalancedPRNGenerator


import statistics


INPUT = "./4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

ONES = (1 << 32) - 1
ZEROS = 0

ALPHA = 3
SQUARE_SIZE = 128


def embed_wmk(patch, bit_string, prn):

    # do something to the patch
    dwt_coeffs = get_dwt_coeff(patch, level=1)

    dwt_coeffs.LL += ALPHA * prn

    return reconstruct_frame(dwt_coeffs)


def embed(video_path, output_path, watermark: int, seed=8787):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0

    # Centres already marked during the current bit run, so the mark rotates instead of
    # sitting in one place. Cleared at every run boundary: detect() replays this exact
    # history from the original video, and the flush is what makes the replay possible
    # without knowing the bits.

    pool = []

    prn_gen = BalancedPRNGenerator((64,64), seed)

    prn = prn_gen.get_blalanced_prn_for_bit_string(bit_string)

    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        h, w = frame.y.shape
        if frame is None:
            break

        patches = get_sift_patches(frame=frame.y, square_size=SQUARE_SIZE)

        for patch, _, _ in patches:

            if patch is None:
                continue

            # patch is a view into frame.y; assign into it in place so the mark is
            # visible in `frame` when it's written out below (reconstruct_frame
            # returns a new array, not a view, so `patch = ...` alone would silently
            # discard the mark)
            patch[:] = embed_wmk(patch, bit_string, prn=prn)

        count += 1
        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    print(f"embedded at {count} locations")

    video_io.release()


if __name__ == "__main__":

    embed(video_path=INPUT, output_path=OUTPUT, watermark=87108710)
