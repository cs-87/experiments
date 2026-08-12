from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import SQUARE_SIZE, get_best_patch_in_two_halves, select_candidate
from utils.pixelation import detect_pixelation
from utils.lightglue import LightGluePatchMatcher, warp_patch
from tqdm import tqdm
import numpy as np
import cv2
import random 

from utils.scene_change import detect_scene_change

INPUT = "videos/15_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "rencode_2.mp4"

SIFT_PATCHES = False

ONES = (1 << 32) - 1
ZEROS = 0

TEMP_REDUNDANCY = 5
ALPHA = 20

# Size of the constant-colour blocks. Shared by embed and detect so the two cannot
# drift apart -- the detector scores against this exact grid period.
PIXEL_SIZE = 2

coeff_1 = (4,5)
coeff_2 = (5,4)


def embed(video_path, output_path, watermark:int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0

    # Centres already marked during the current bit run, so the mark rotates instead of
    # sitting in one place. Cleared at every run boundary: detect() replays this exact
    # history from the original video, and the flush is what makes the replay possible
    # without knowing the bits.
    pool = []
    prv_frame = None
    for i in tqdm(range(frame_count),total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        if prv_frame != None:
            scene_change = detect_scene_change(frame1=frame,frame2=prv_frame)

        if i % TEMP_REDUNDANCY == 0:
            pool.clear()

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        first_half,second_half = get_best_patch_in_two_halves(frame.y)

        half = first_half if bit == 1 else second_half

        selection = select_candidate(half, pool)

        if selection is not None:
            patch, y, x, centre = selection
            wmk_frame_y = frame.y.copy()
            wmk_frame_y[y[0]:y[1], x[0]:x[1]] = pixelate_region(patch)
            frame.set_y(wmk_frame_y)
            pool.append(centre)
            count += 1

        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    print(f"embedded at {count} locations")

    video_io.release()


def pixelate_region(img, pixel_size=PIXEL_SIZE):
    """
    Pixelate a rectangular region.

    pixel_size: approximate size of each pixel block.
    """
    h,w = img.shape[:2]

    # Downsample
    small = cv2.resize(
        img,
        (max(1, w // pixel_size), max(1, h // pixel_size)),
        interpolation=cv2.INTER_LINEAR
    )

    # Upsample with nearest neighbor
    pixelated = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return pixelated


def detect(watermark_path,org_video_path, bit_length=BIT_LENGTH, block_size=PIXEL_SIZE):
    """
    Recover the embedded watermark from a (possibly re-encoded) leaked video.

    Needs the original video, not as a reference to subtract, but to recompute the
    keypoints -- and therefore the candidate patches -- that embed() chose from. It is
    also used to rescale temp_redundancy when the leak was re-encoded at a different
    frame rate, matching dwt_dct.detect.

    block_size: grid period to score against. Left as None it is auto-detected once
    from the first frame and then held fixed. Detecting per half would be wrong: the
    two halves would be scored on different grids, and the comparison below only means
    something if both sides answer the same question.

    Each frame is decided by comparing the two halves against each other rather than
    against a fixed threshold. Both halves went through the same encoder and the same
    scene, so a relative comparison cancels bitrate and content effects that would
    otherwise swamp an absolute score. This only works if we score the patch embed()
    actually marked -- scoring the top keypoint instead lands on an unmarked patch on
    most frames and the comparison degenerates to a coin flip.
    """
    wmk_io = Video_IO(watermark_path)

    org_io = Video_IO(org_video_path)
    temp_redundancy = TEMP_REDUNDANCY * (wmk_io.frame_count // org_io.frame_count)
    if temp_redundancy == 0:
        temp_redundancy = 1

    bit_array = [[] for _ in range(bit_length)]
    frames_read = 0

    # One pool per half, mirroring embed()'s single pool. Both are advanced on every
    # frame regardless of how the vote goes, which is what makes the replay independent
    # of the bits: within a run the bit is constant, so embed() marked the same half on
    # all of its frames and that half's pool matches exactly, while the other half's is
    # a harmless what-if. Divergence accumulates only across runs -- embed() never
    # advanced the unmarked half -- and the flush at each run boundary erases it.
    pool_first, pool_second = [], []

    #lightglue_matcher = LightGluePatchMatcher()

    for i in tqdm(range(wmk_io.frame_count), desc=f"detecting:{watermark_path}"):
        frame_og = org_io.read_frame()
        frame = wmk_io.read_frame()
        if frame is None or frame_og is None:
            break
        frames_read += 1

        if i % temp_redundancy == 0:
            pool_first.clear()
            pool_second.clear()

        bit_index = (i // temp_redundancy) % bit_length


        #homography = lightglue_matcher.compute_homography(frame_og.y, frame.y)

        homography = np.eye(3, dtype=np.float32)

        first_half,second_half = get_best_patch_in_two_halves(frame_og.y)

        sel_first = select_candidate(first_half, pool_first)
        sel_second = select_candidate(second_half, pool_second)

        # Advance each pool as soon as its own half yields a candidate, before any of
        # the bail-outs below. embed() advanced its pool whenever the half it was
        # marking was non-empty, so skipping a push here for an unrelated reason (the
        # other half empty, a warp off the edge) would leave this half's history one
        # entry behind for the rest of the run.
        if sel_first is not None:
            pool_first.append(sel_first[3])
        if sel_second is not None:
            pool_second.append(sel_second[3])

        if sel_first is None or sel_second is None:
            continue

        (x1, y1), (x2, y2) = sel_first[3], sel_second[3]

        wrapped_first = warp_patch(frame.y, x1, y1, homography, SQUARE_SIZE)
        wrapped_second = warp_patch(frame.y, x2, y2, homography, SQUARE_SIZE)

        if wrapped_first is None or wrapped_second is None:
            continue

        left = detect_pixelation(wrapped_first, block_size=block_size)
        right = detect_pixelation(wrapped_second, block_size=block_size)

        # embed() pixelates the first (left) half for a 1 and the second for a 0.
        if left.confidence > right.confidence:
            bit_array[bit_index].append(1)
        elif left.confidence < right.confidence:
            bit_array[bit_index].append(0)

    wmk_io.release()
    org_io.release()

    bit_str = ""
    for votes in bit_array:
        diff = votes.count(1) - votes.count(0)
        if diff > 0:
            bit_str += "1"
        elif diff < 0:
            bit_str += "0"
        else:
            bit_str += "?"

    # A short clip cannot carry every bit: only frames_read // temp_redundancy bit
    # slots were ever embedded, so trust no more than that many.
    if frames_read < temp_redundancy * bit_length:
        expected_length = frames_read // temp_redundancy
    else:
        expected_length = bit_length

    decoded = bit_str[:expected_length]
    print(f"votes: {bit_str}  decoded: {decoded}  slots carried: {expected_length}")

    if not decoded or "?" in decoded:
        print("undecided slots, watermark not recoverable")
        return None

    watermark = get_integer(decoded, expected_length)
    print(watermark)
    return watermark


if __name__ == "__main__":

    embed(video_path=INPUT, output_path=OUTPUT, watermark=87108710)
    import subprocess

    subprocess.run(["bash", "utils/compress.sh",OUTPUT,"out.mp4","6"], check=True)
    detect(watermark_path=OUTPUT, org_video_path=INPUT)