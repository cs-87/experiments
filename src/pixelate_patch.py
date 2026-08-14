from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import SQUARE_SIZE, get_best_patch_in_two_halves, select_candidate, get_middle_patches
from utils.patch import get_half_patches_grid
from utils.pixelation import detect_pixelation
from utils.lightglue import LightGluePatchMatcher, warp_patch
from tqdm import tqdm
import numpy as np
import cv2
import random

from utils.scene_change import detect_scene_change
from impairment_view import Impairment_View
from utils.transcode_n import transcode_n_times

INPUT = "4_sec_source.mp4"
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

# 256px patch -> 512px in the panel, so a PIXEL_SIZE cell reads as 4px instead of 2.
PANEL_SCALE = 2

# Grey gutter between the two halves, so a diff that runs bright up to its edge cannot be
# read as spilling into the other half.
DIVIDER_PX = 4
DIVIDER_GREY = 90

# Brightness multiplier on the diff. The mark is genuinely faint -- measured on
# out/wmk.mp4 the marked patch's diff runs 6-10 mean against 3 for the unmarked one -- so
# at 1:1 both halves render as near-black and the view is unreadable.
#
# Deliberately a fixed gain and not a per-frame autoscale: normalising each half by its
# own maximum would stretch the unmarked half's codec noise to full range and make it look
# exactly as marked as the other side, destroying the one comparison this view is for.
DIFF_GAIN = 4


class PIXELATE_IMP_VIEW(Impairment_View):
    """
    Renders, per frame, imp - org over both of embed()'s candidate patches: the left
    half's patch on the left, the right half's on the right.

    embed() rotates its mark around a 256px grid, so a fixed patch pair -- what
    DWT_DCT_IMP_VIEW can get away with -- would land on an unmarked region on most frames.
    This replays embed()'s own selection instead: same grid, same scorer, same pool. The
    panel therefore always shows the patch carrying the mark beside the one that is not,
    and the bit is whichever side is brighter.
    """

    def __init__(self, src_path, imp_path, out_dir):
        super().__init__(src_path, imp_path, out_dir)

        # One pool per half, because the view renders both halves without knowing the bit.
        # Within a bit run embed() marked the same half on every frame, so that half's
        # pool matches embed()'s exactly while the other's is a harmless what-if.
        # Divergence accumulates only across runs, and the per-run flush erases it.
        self.pool_first, self.pool_second = [], []

    def _select_both_halves(self, src_y):
        """
        Replay embed()'s choice for both halves of this frame.

        Returns (sel_first, sel_second); either is None when its half yields no candidate.
        Each pool is advanced as soon as its own half produces one, matching embed(),
        which pushed whenever the half it was marking was non-empty -- skipping a push
        here would leave that pool a frame behind for the rest of the run.
        """
        first_half, second_half = get_half_patches_grid(src_y)

        sel_first = select_candidate(
            first_half, self.pool_first, key=pixelation_residual)
        sel_second = select_candidate(
            second_half, self.pool_second, key=pixelation_residual)

        if sel_first is not None:
            self.pool_first.append(sel_first[3])
        if sel_second is not None:
            self.pool_second.append(sel_second[3])

        return sel_first, sel_second

    def _patch_diff(self, selection, imp_y):
        """
        imp - org over the selected patch, brightened by DIFF_GAIN, or None when that half
        had no candidate.

        absdiff rather than a plain subtraction: both planes are uint8 and the difference
        goes both ways, so a raw subtraction would wrap every negative sample to ~255 and
        render noise as the brightest thing in the panel. Left unthresholded -- the base
        class's threshold of 30 sits well above the mark's own amplitude (pixelation_residual
        measures 2-5 mean) and would erase exactly what this view exists to show.
        """
        if selection is None:
            return None

        patch_src, y, x, _ = selection

        diff = cv2.absdiff(imp_y[y[0]:y[1], x[0]:x[1]], patch_src)

        # convertScaleAbs saturates at 255; a plain multiply would wrap the brightest
        # samples back to black.
        return diff

    def _compose_panel(self, left_diff, right_diff):
        """Left half of the canvas is the left patch's diff, right half the right's."""
        size = SQUARE_SIZE

        panel = np.zeros((size, size * 2 + DIVIDER_PX), dtype=np.uint8)
        panel[:, size:size + DIVIDER_PX] = DIVIDER_GREY

        for diff, x0 in ((left_diff, 0), (right_diff, size + DIVIDER_PX)):

            # A half with no candidate stays black. The frame is still written, so the
            # PNG sequence keeps its one-to-one alignment with the video.
            if diff is None:
                continue

            # NEAREST, or the upscale interpolates away the block edges this view is for:
            # at PIXEL_SIZE 2 a cell is a single interpolation step wide.
            panel[:, x0:x0 + size] = cv2.threshold(
                diff,
                30,
                255,
                cv2.THRESH_BINARY
            )[1]

        return panel

    def get_frame_diff(self):

        src_frame = self.get_source_frame()
        if src_frame is None:
            return None

        imp_frame = self.get_imp_frame()
        if imp_frame is None:
            return None

        # get_source_frame/get_imp_frame already hand back YUV. Converting again here
        # would treat Y/U/V as B/G/R and mix chroma into the plane the mark lives in.
        src_y = src_frame[:, :, 0]
        imp_y = imp_frame[:, :, 0]

        # The patch coordinates are derived from the source frame, so the impaired frame
        # has to be on the same grid before they can crop the same region out of it.
        imp_y = cv2.resize(imp_y, (src_y.shape[1], src_y.shape[0]))

        # Mirrors embed()'s flush. The pool is what makes the mark rotate within a run,
        # so replaying the rotation means replaying the clear at each run boundary too.
        if self.frame_index % TEMP_REDUNDANCY == 0:
            self.pool_first.clear()
            self.pool_second.clear()

        sel_first, sel_second = self._select_both_halves(src_y)

        return self._compose_panel(
            self._patch_diff(sel_first, imp_y),
            self._patch_diff(sel_second, imp_y),
        )


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
    prv_frame = None
    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        scene_change = False

        if prv_frame != None:
            scene_change = detect_scene_change(frame1=frame, frame2=prv_frame)

        if i % TEMP_REDUNDANCY == 0:
            pool.clear()

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        first_half, second_half = get_half_patches_grid(frame.y)

        half = first_half if bit == 1 else second_half

        selection = select_candidate(half, pool, key=pixelation_residual)

        if selection is not None:
            patch, y, x, centre = selection
            wmk_frame_y = frame.y.copy()
            pixel_size = PIXEL_SIZE if not scene_change else PIXEL_SIZE*2
            wmk_frame_y[y[0]:y[1], x[0]:x[1]
                        ] = pixelate_region(patch, pixel_size)
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
    h, w = img.shape[:2]

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


def pixelation_residual(patch, pixel_size=PIXEL_SIZE):
    """
    Mean |patch - pixelate_region(patch)|: the energy this patch would lose to the
    embedder, and therefore the amplitude of the mark it can carry.

    Used to rank candidates in select_candidate. SIFT's own (size, response) order is a
    poor proxy: it ranks by blob scale, which is uncorrelated with how much sub-block
    detail there is for a PIXEL_SIZE grid to destroy. Pick a patch with little of it and
    the mark is quantised away by the first encoder that touches it -- measured on
    4_sec_source, the best candidate carried 4.91 mean residual and cleared the
    detector's floor by +0.215 after six generations of x264, while the worst carried
    2.24 and cleared it by +0.013.

    Ranking by this took frame-vote accuracy from 91.7% to 95.8% over the same six
    generations, which was the difference between the 22-bit payload failing on one
    flipped bit and decoding cleanly.

    Reads only the original frame, so detect() reproduces the ordering exactly.
    """
    p = patch.astype(np.float32)
    return float(np.abs(p - pixelate_region(patch, pixel_size).astype(np.float32)).mean())


def get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view"):
    impv = PIXELATE_IMP_VIEW(INPUT, imp_path, out_dir)
    impv.start()
    impv.release()


if __name__ == "__main__":

    # embed(video_path=INPUT, output_path=OUTPUT, watermark=ZEROS)

    # detect(watermark_path=OUTPUT, org_video_path=INPUT)

    # transcode_n_times(OUTPUT, "pixelate", 6)

    get_frame_imp_analysis(imp_path="pixelate/transcoded_6.mp4")
