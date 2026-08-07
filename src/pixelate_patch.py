from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import SQUARE_SIZE, get_best_patch_in_two_halves
from utils.pixelation import detect_pixelation
from utils.lightglue import LightGluePatchMatcher, warp_patch
from tqdm import tqdm
import numpy as np
import cv2

INPUT = "videos/4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "rencode_2.mp4"

SIFT_PATCHES = False

ONES = (1 << 32) - 1
ZEROS = 0

TEMP_REDUNDANCY = 3
ALPHA = 20

# Size of the constant-colour blocks. Shared by embed and detect so the two cannot
# drift apart -- the detector scores against this exact grid period.
PIXEL_SIZE = 4

coeff_1 = (4,5)
coeff_2 = (5,4)


def embed(video_path, output_path, watermark:int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    for i in tqdm(range(frame_count),total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        wmk_frame_y = frame.y.copy()

        first_half,second_half = get_best_patch_in_two_halves(frame.y)

        if bit == 1:
            patch,y,x = first_half
            wmk_patch = pixelate_region(patch)


        if bit == 0:
            patch,y,x = second_half
            wmk_patch = pixelate_region(patch)

        wmk_frame_y[y[0]:y[1], x[0]:x[1]] = wmk_patch


        frame.set_y(wmk_frame_y)

        video_io.write_frame(frame, output_path)
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

    Blind: pixelation is detectable from the leak alone, so no reference frames are
    needed. org_video_path is optional and used only to rescale temp_redundancy when
    the leak was re-encoded at a different frame rate, matching dwt_dct.detect.

    block_size: grid period to score against. Left as None it is auto-detected once
    from the first frame and then held fixed. Detecting per half would be wrong: the
    two halves would be scored on different grids, and the comparison below only means
    something if both sides answer the same question.

    Each frame is decided by comparing the two halves against each other rather than
    against a fixed threshold. Both halves went through the same encoder and the same
    scene, so a relative comparison cancels bitrate and content effects that would
    otherwise swamp an absolute score.
    """
    wmk_io = Video_IO(watermark_path)

    temp_redundancy = TEMP_REDUNDANCY
    org_io = Video_IO(org_video_path)
    temp_redundancy = TEMP_REDUNDANCY * (wmk_io.frame_count // org_io.frame_count)
    if temp_redundancy == 0:
        temp_redundancy = 1

    bit_array = [[] for _ in range(bit_length)]
    frames_read = 0

    lightglue_matcher = LightGluePatchMatcher()


    for i in tqdm(range(wmk_io.frame_count), desc=f"detecting:{watermark_path}"):
        frame_og = org_io.read_frame()
        frame = wmk_io.read_frame()
        if frame is None:
            break
        frames_read += 1

        bit_index = (i // temp_redundancy) % bit_length

        if frames_read == 1:

            homography = lightglue_matcher.compute_homography(frame_og.y, frame.y)

        first_half,second_half = get_best_patch_in_two_halves(frame_og.y)

        y11,y12 = first_half[1][0],first_half[1][1]
        x11,x12 = first_half[2][0],first_half[2][1]
        y21,y22 = second_half[1][0],second_half[1][1]
        x21,x22 = second_half[2][0],second_half[2][1]

        x1,y1 = (x11+x12)//2,(y11+y12)//2
        x2,y2 = (x21+x22)//2,(y21+y22)//2


        wrapped_first = warp_patch(frame.y, x1, y1, homography, SQUARE_SIZE)
        wrapped_second = warp_patch(frame.y, x2, y2, homography, SQUARE_SIZE)

        left = detect_pixelation(wrapped_first, block_size=block_size)
        right = detect_pixelation(wrapped_second, block_size=block_size)

        # embed() pixelates the first (left) half for a 1 and the second for a 0.
        if left.confidence > right.confidence:
            bit_array[bit_index].append(1)
        elif left.confidence < right.confidence:
            bit_array[bit_index].append(0)
        else:
            bit_array[bit_index].append(None)

    wmk_io.release()

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

    watermark = get_integer(bit_str, expected_length)
    print(bit_str, watermark, expected_length)
    return watermark


if __name__ == "__main__":
    embed(video_path=INPUT, output_path=OUTPUT, watermark=8710)
    detect(watermark_path=OUTPUT, org_video_path=INPUT)