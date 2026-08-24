from utils.video import Video_IO
from enum import Enum
from src.blur.blur import blur_region
from utils.bit import get_bit_string, BIT_LENGTH, hex_to_uint32, uint32_to_hex
from tqdm import tqdm
import numpy as np
import statistics

from src.blur.blur import score_patch_for_watermark, TEMP_REDUNDANCY
from src.blur.blur_impairment import BLUR_IMPAIRMENT_VIEW

from utils.transcode_n import transcode_n_times

from src.blur.blur_detect import detect

from enum import Enum

from src.blur.patch import (
    get_best_patch_in_both_half,
    get_best_sift_patches_in_two_halves,
    get_middle_patches,
)


class PATCH_DECISION(Enum):
    MIDDLE = 0
    SIFT = 1
    BEST_BLUR = 2


PATCH_FUNCTIONS = {
    PATCH_DECISION.MIDDLE: get_middle_patches,
    PATCH_DECISION.SIFT: get_best_sift_patches_in_two_halves,
    PATCH_DECISION.BEST_BLUR: get_best_patch_in_both_half,
}


PATCH_DECISION_ALGO = PATCH_DECISION.BEST_BLUR



INPUT = "/home/csx87/Workspace/sample_videos/4_min_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "IMG_2220.mov"

ONES = (1 << 32) - 1
ZEROS = 0




def embed(video_path, output_path, watermark: int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0


    if frame_count < length*TEMP_REDUNDANCY:
        if frame_count < TEMP_REDUNDANCY:
            raise ValueError(f"Video should have at least {TEMP_REDUNDANCY} frames")
        length = frame_count // TEMP_REDUNDANCY 

    # Centres already marked during the current bit run, so the mark rotates instead of
    # sitting in one place. Cleared at every run boundary: detect() replays this exact
    # history from the original video, and the flush is what makes the replay possible
    # without knowing the bits.
    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        if PATCH_DECISION_ALGO == PATCH_DECISION.BEST_BLUR:

            first_half, second_half = PATCH_FUNCTIONS[PATCH_DECISION_ALGO](frame.y, score_patch_for_watermark)

        else:
            first_half, second_half = PATCH_FUNCTIONS[PATCH_DECISION_ALGO](frame.y)

        half = first_half if bit == 1 else second_half


        if half is not None:
            patch, y, x = half
            wmk_frame_y = frame.y.copy()
            # idct2 can overshoot [0,255] by a little at patch edges (ringing).
            # Assigning float straight into the uint8 frame wraps those samples --
            # -0.4 lands on 255 -- so clip before the cast.
            wmk_frame_y[y[0]:y[1], x[0]:x[1]
                        ] = np.clip(blur_region(patch), 0, 255)
            frame.set_y(wmk_frame_y)
            count += 1

        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    print(f"embedded at {count} locations")

    video_io.release()
    return length


def get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view"):
    impv = BLUR_IMPAIRMENT_VIEW(INPUT, imp_path, out_dir)
    impv.start()
    impv.release()



if __name__ == "__main__":


    length = embed(video_path=INPUT, output_path=OUTPUT, watermark=ZEROS)

    #length = BIT_LENGTH

    #transcode_n_times(OUTPUT, "blur", 6)

    output = OUTPUT #if 0 else "blur/transcoded_6.mp4"
    

    candidate_score = score_patch_for_watermark if PATCH_DECISION_ALGO == PATCH_DECISION.BEST_BLUR else None
    result = detect(org_video_path=INPUT , imp_video_path=output,temp_redundancy=TEMP_REDUNDANCY, candidate_fn=PATCH_FUNCTIONS[PATCH_DECISION_ALGO],candidate_score=candidate_score, bit_length=length)
    print(result["watermark"],uint32_to_hex(result["watermark"]) ,result["bit_string"])

    margins = result["margins"]

    if len(margins) > 1:

        margins = [x for x in margins if x!= 0]
        mean = statistics.mean(margins)
        std = statistics.stdev(margins)

        print(round(mean,2),round(mean - 3*std,2))  

    #get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view")
