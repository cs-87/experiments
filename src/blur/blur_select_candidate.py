from utils.video import Video_IO
from enum import Enum
from src.blur.blur import blur_region
from utils.bit import get_bit_string
from tqdm import tqdm
import numpy as np
import random
from src.blur.blur import RADIUS, TEMP_REDUNDANCY
from src.blur.blur_impairment import BLUR_IMPAIRMENT_VIEW

from enum import Enum

from src.blur.patch import (
    select_candidate,
    score_patch_for_watermark,
    get_half_patches_grid
)


class PATCH_DECISION(Enum):
    TR = 0
    RANDOM = 1


PATCH_DECISION_ALGO = PATCH_DECISION.TR



INPUT = "~/Workspace/sample_videos/4_min_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "IMG_2220.mov"

SIFT_PATCHES = False

ONES = (1 << 32) - 1
ZEROS = 0

ALPHA = 20






def embed(video_path, output_path, watermark: int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0

    # Centres already marked during the current bit run, so the mark rotates instead of
    # sitting in one place. Cleared at every run boundary: detect() replays this exact
    # history from the original video, and the flush is what makes the replay possible
    # without knowing the bits.

    pool = []
    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        if PATCH_DECISION_ALGO == PATCH_DECISION.TR:
            if i % TEMP_REDUNDANCY == 0:

                pool.clear()

        first_halves, second_halves = get_half_patches_grid(frame.y)

        halves = first_halves if bit == 1 else second_halves

        if halves is not None and len(halves) > 0:

            if PATCH_DECISION_ALGO == PATCH_DECISION.TR:

                selected_patch,y,x,centre = select_candidate(halves,pool,score_patch_for_watermark)

            elif PATCH_DECISION_ALGO == PATCH_DECISION.RANDOM:
                i = random.randint(0,len(halves)-1)
                selected_patch,y,x = halves[i]

            wmk_frame_y = frame.y.copy()
            # idct2 can overshoot [0,255] by a little at patch edges (ringing).
            # Assigning float straight into the uint8 frame wraps those samples --
            # -0.4 lands on 255 -- so clip before the cast.
            wmk_frame_y[y[0]:y[1], x[0]:x[1]
                        ] = np.clip(blur_region(selected_patch), 0, 255)
            if PATCH_DECISION_ALGO == PATCH_DECISION.TR:
                pool.append(centre)

            frame.set_y(wmk_frame_y)
            count += 1

        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    print(f"embedded at {count} locations")

    video_io.release()


def get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view"):
    impv = BLUR_IMPAIRMENT_VIEW(INPUT, imp_path, out_dir)
    impv.start()
    impv.release()


if __name__ == "__main__":

    embed(video_path=INPUT, output_path=OUTPUT, watermark=ZEROS)

    # detect(watermark_path=OUTPUT, org_video_path=INPUT)

    #transcode_n_times(OUTPUT, "blur", 6)

    get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view")
