from utils.video import Video_IO, DEFAULT_CRF
from enum import Enum
from src.blur.blur import blur_region
from utils.bit import get_bit_string, BIT_LENGTH, hex_to_uint32, uint32_to_hex
from tqdm import tqdm
import numpy as np
import statistics

from src.blur.blur import score_patch_for_watermark, TEMP_REDUNDANCY, RADIUS
from src.blur.blur_impairment import BLUR_IMPAIRMENT_VIEW
from src.blur.mapping import bit_index_for_frame

from utils.transcode_n import transcode_n_times

from src.blur.blur_detect import detect

from enum import Enum

from src.blur.patch import (
    CLUSTER_K,
    get_best_patch_in_both_half,
    get_best_sift_patches_in_two_halves,
    get_middle_cluster,
    get_middle_patches,
)

PATCH_FUNCTIONS = {
    "MIDDLE": get_middle_patches,
    "SIFT": get_best_sift_patches_in_two_halves,
    "BEST_BLUR": get_best_patch_in_both_half,
}


PATCH_DECISION_ALGO = "MIDDLE"

# Spread each bit's frames across the whole clip instead of holding it still for a run.
# See src/blur/mapping.py for why, and keep the contiguous map reachable so the harness
# can A/B the two on one source video.
INTERLEAVE = False


INPUT = f"./inputs/{TEMP_REDUNDANCY}.mp4"
OUTPUT = f"./outputs/{PATCH_DECISION_ALGO}/{TEMP_REDUNDANCY}/{RADIUS}.mp4"

LEAKED = "IMG_2220.mov"

ONES = (1 << 32) - 1
ZEROS = 0


def embed(video_path, output_path, watermark: int,
          temp_redundancy=TEMP_REDUNDANCY, radius=RADIUS,
          interleave=INTERLEAVE, cluster_k=CLUSTER_K,
          patch_algo=PATCH_DECISION_ALGO, max_frames=None,
          crf=DEFAULT_CRF, verbose=True):
    """
    Write `watermark` into `video_path` and encode the result to `output_path`.

    Everything the mark depends on is an argument rather than a module global, because
    the sweep varies these and a global would make "which video is this" a question
    about import order. The defaults reproduce the module's own constants.

    cluster_k > 1 blurs that many grid cells on the marked side instead of one. The
    extra cells sit on different content, so unlike more frames of the same scene their
    measurement errors are close to independent -- that is where redundancy can actually
    come from (finding F4).
    """

    if verbose:
        print(output_path)

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path, crf=crf)
    frame_count = video_io.frame_count
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    count = 0

    # Previously this silently shortened the payload, so a short clip returned a
    # watermark of a different length than the caller asked for and the mismatch only
    # surfaced as a wrong answer at detection time. A payload that does not fit is a
    # caller error, not something to quietly renegotiate.
    if not interleave and frame_count < length * temp_redundancy:
        raise ValueError(
            f"{video_path} has {frame_count} frames; the contiguous map needs "
            f"{length} bits x {temp_redundancy} frames = {length * temp_redundancy}. "
            f"Lower temp_redundancy, pass interleave=True, or use a longer clip."
        )
    if frame_count < length:
        raise ValueError(
            f"{video_path} has {frame_count} frames, fewer than the {length} bits of "
            f"payload -- some bits would never be written."
        )

    frames = range(frame_count)
    if verbose:
        frames = tqdm(frames, total=frame_count, unit="frame", desc="embedding")

    for i in frames:
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = bit_index_for_frame(
            i, temp_redundancy, length, interleave=interleave)

        bit = int(bit_string[bit_index])

        if patch_algo == "MIDDLE":
            # The cluster generalises get_middle_patches: k=1 selects the same single
            # pair, so one code path covers both and the grid convention cannot drift.
            left_cells, right_cells = get_middle_cluster(frame.y, k=cluster_k)
            marked = left_cells if bit == 1 else right_cells
        elif patch_algo == "BEST_BLUR":
            first_half, second_half = PATCH_FUNCTIONS[patch_algo](
                frame.y, score_patch_for_watermark)
            marked = [first_half if bit == 1 else second_half]
        else:
            first_half, second_half = PATCH_FUNCTIONS[patch_algo](frame.y)
            marked = [first_half if bit == 1 else second_half]

        wmk_frame_y = None
        for half in marked:
            if half is None or half[0] is None:
                continue
            patch, y, x = half
            if wmk_frame_y is None:
                wmk_frame_y = frame.y.copy()
            # idct2 can overshoot [0,255] by a little at patch edges (ringing).
            # Assigning float straight into the uint8 frame wraps those samples --
            # -0.4 lands on 255 -- so clip before the cast.
            wmk_frame_y[y[0]:y[1], x[0]:x[1]] = np.clip(
                blur_region(patch, radius=radius), 0, 255)
            count += 1

        if wmk_frame_y is not None:
            frame.set_y(wmk_frame_y)

        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    if verbose:
        print(f"embedded at {count} locations")

    video_io.release()
    return length


def get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view"):
    impv = BLUR_IMPAIRMENT_VIEW(INPUT, imp_path, out_dir)
    impv.start()
    impv.release()


if __name__ == "__main__":

    length = embed(video_path=INPUT, output_path=OUTPUT,
                   watermark=hex_to_uint32("CAFECAFE"))  # 0x12345678

    # length = BIT_LENGTH

    # transcode_n_times(OUTPUT, "blur", 6)
    '''
    output = OUTPUT  # if 0 else "blur/transcoded_6.mp4"

    candidate_score = score_patch_for_watermark if PATCH_DECISION_ALGO == "BEST_BLUR" else None
    result = detect(org_video_path=INPUT, imp_video_path=output, temp_redundancy=TEMP_REDUNDANCY,
                    candidate_fn=PATCH_FUNCTIONS[PATCH_DECISION_ALGO], candidate_score=candidate_score, bit_length=length)
    print(result["watermark"], uint32_to_hex(
        result["watermark"]), result["bit_string"])

    margins = result["margins"]

    if len(margins) > 1:

        margins = [x for x in margins if x != 0]
        mean = statistics.mean(margins)
        std = statistics.stdev(margins)

        print(round(mean, 2), round(mean - 3*std, 2))
    '''

    # get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view")
