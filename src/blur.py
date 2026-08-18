from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import SQUARE_SIZE, get_two_halves, get_best_patch_in_two_halves, select_candidate, get_middle_patches, get_middle_split_col
from utils.patch import get_half_patches_grid, get_grid_patches
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

LEAKED = "IMG_2220.mov"

SIFT_PATCHES = False

ONES = (1 << 32) - 1
ZEROS = 0

TEMP_REDUNDANCY = 120
ALPHA = 20

# DCT radial cutoff of the blur. Shared by embed and the impairment view so the two
# cannot drift apart -- the view scores the cliff at this exact radius.
#
# On a 256x256 patch the coefficient radius runs to sqrt(2)*256 = 362, so the cutoff
# has to sit well inside that to remove real content. At 300 only the far diagonal
# corner is nulled -- 6% of coefficients holding 0.006% of the AC energy -- and H.264
# refills that band with more blocking noise than it removed, leaving the mark
# undetectable (cliff margin x1.13, i.e. nothing). 200 nulls 52% of coefficients for
# 0.55% of AC energy: still visually subtle, but the cliff survives encoding at x11.
RADIUS = 200

# Width of the annuli the view compares either side of RADIUS, in coefficients.
CLIFF_BAND = 24

# Below this below/above energy ratio the frame is reported as undecided rather than
# guessed at. Natural content sits near x2-3; a surviving blur cliff is x10 upwards.
CLIFF_MIN_MARGIN = 4.0

class BLUR_IMPAIRMENT_VIEW(Impairment_View):

    def __init__(self, src_path, imp_path, out_dir):
            super().__init__(src_path, imp_path, out_dir)
    
            # One pool per half, because the view renders both halves without knowing the bit.
            # Within a bit run embed() marked the same half on every frame, so that half's
            # pool matches embed()'s exactly while the other's is a harmless what-if.
            # Divergence accumulates only across runs, and the per-run flush erases it.
            self.pool_first, self.pool_second = [], []
    
            self.matcher = LightGluePatchMatcher()
            self.homography = np.eye(3, dtype=np.float32)

    """
    Blind read-out of the embedded bit.

    Blind in the sense that matters: it never opens the source video and never uses the
    marked coordinates. What it is allowed to know is the *scheme* -- SQUARE_SIZE, that
    the block is aligned to that grid, the DCT cutoff RADIUS, and that the left/right
    half of frame centre encodes 1/0. It scores every grid cell and lets the winner fall
    out of the scores, so it finds the block rather than being told where it is.
    """

    def cliff_score(self, patch):
        """
        How abruptly the patch's spectrum dies at RADIUS.

        Mean coefficient energy in the annulus just inside the cutoff over that just
        outside it. blur_region() zeroes everything beyond RADIUS, so on a marked patch
        the denominator collapses to the codec's quantisation floor and the ratio
        explodes. Natural content decays as a smooth power law across the same two
        annuli and stays near x2-3.

        Deliberately a *ratio of adjacent bands* rather than an absolute high-frequency
        measure: absolute smoothness would just elect whichever cell happens to be sky
        or letterbox, while the cliff is a signature only the DCT mask leaves.
        """
        h, w = patch.shape
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(X**2 + Y**2)

        energy = dct.dct2(patch)**2

        below = energy[(dist >= RADIUS - CLIFF_BAND) & (dist < RADIUS)].mean()
        above = energy[(dist >= RADIUS) & (dist < RADIUS + CLIFF_BAND)].mean()

        return below / (above + 1e-9)

    def get_frame_diff(self):

        org_frame = self.get_source_frame()
        if org_frame is None:
            return None

        imp_frame = self.get_imp_frame()
        if imp_frame is None:
            return None


        if self.frame_index == 0:
            # A rejected homography is not a reason to abandon the run: for a
            # pixel-aligned re-encode identity is the correct answer anyway, and the
            # cliff scores below will show plainly if the alignment is in fact wrong.
            self.homography = self.matcher.compute_homography(org_frame, imp_frame)
            if self.homography is None:
                print("homography rejected by validation, falling back to identity")
                self.homography = np.eye(3, dtype=np.float32)
        
        H, W = imp_frame.shape

        residual = np.zeros(imp_frame.shape, dtype=np.float32)
        cells = []

        for patch, y, x in get_grid_patches(org_frame):

            h, w = patch.shape

            imp_patch = warp_patch(
                imp_frame,x=x[0],y=y[0],homography=self.homography,patch_size=SQUARE_SIZE
            )

            # A cell whose projection leaves the impaired frame has nothing to measure.
            # Leave its residual black and keep it out of the scoring rather than
            # scoring an absence -- a partly-missing patch reads as a spectral cliff and
            # would win the vote for the wrong reason.
            if imp_patch is None:
                continue

            # Keep ONLY high-frequency DCT coefficients
            mask = get_blur_mask(radius=RADIUS, h=h, w=w, lower=False)

            filtered_dct = dct.dct2(imp_patch) * mask

            # Back to the spatial domain, as magnitude -- the backdrop is only there to
            # give the eye some context for where the scores sit.
            residual[y[0]:y[1], x[0]:x[1]] = np.abs(dct.idct2(filtered_dct))

            # Scored on the *impaired* patch, never the original: the cliff is the mark's
            # signature as it survived encoding, and org_frame's own patch carries none.
            cells.append((self.cliff_score(imp_patch), y, x))

        return self.render(residual, cells, H, W)

    def render(self, residual, cells, H, W):
        """Draw the scores over the residual so the bit is readable at a glance."""

        # Percentile stretch, not min-max: one hot pixel anywhere in the frame would
        # otherwise set the ceiling and crush the whole backdrop to black.
        ceiling = np.percentile(residual, 99.5)
        backdrop = np.clip(residual / (ceiling + 1e-9), 0, 1)
        # Gamma lift, purely cosmetic -- the residual is bottom-heavy and stays
        # invisible on a linear ramp.
        backdrop = (np.power(backdrop, 0.45) * 255).astype(np.uint8)

        canvas = cv2.cvtColor(backdrop, cv2.COLOR_GRAY2BGR)
        canvas = (canvas * 0.55).astype(np.uint8)

        # Two cells is the minimum a margin can be computed from. Fewer means the
        # alignment put almost the whole grid off-frame, which is a homography problem
        # and not something to report a bit for.
        if len(cells) < 2:
            cv2.putText(canvas, f"NO SCORABLE CELLS ({len(cells)})", (24, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 80, 230), 3, cv2.LINE_AA)
            return canvas

        cells = sorted(cells, reverse=True, key=lambda c: c[0])
        best, runner = cells[0], cells[1]
        margin = best[0] / (runner[0] + 1e-9)

        for score, y, x in cells:
            # Every cell gets its grid box and score, so a misfire is visible as a
            # near-tie instead of hiding behind a confident-looking red box.
            cv2.rectangle(canvas, (x[0], y[0]), (x[1] - 1, y[1] - 1), (40, 40, 40), 1)
            cv2.putText(canvas, f"{score:.1f}", (x[0] + 8, y[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1, cv2.LINE_AA)

        # The boundary that separates bit 1 from bit 0. Same helper the embedder splits
        # on -- deriving a midpoint here instead would disagree with it whenever the
        # frame centre is not itself on a grid line.
        split = get_middle_split_col(W)
        cv2.line(canvas, (split, 0), (split, H), (0, 140, 200), 2, cv2.LINE_AA)

        decided = margin >= CLIFF_MIN_MARGIN
        score, y, x = best
        colour = (0, 230, 0) if decided else (0, 160, 230)

        cv2.rectangle(canvas, (x[0], y[0]), (x[1] - 1, y[1] - 1), colour, 3)
        cv2.putText(canvas, f"{score:.1f}", (x[0] + 8, y[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        bit = 1 if (x[0] + x[1]) // 2 < split else 0
        side = "LEFT" if bit == 1 else "RIGHT"

        if decided:
            headline = f"BIT = {bit}   ({side} of centre)"
        else:
            headline = f"UNDECIDED  (would read {bit}, {side})"

        cv2.putText(canvas, headline, (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, colour, 3, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"frame {self.frame_index}   cliff {score:.1f} vs next {runner[0]:.1f}"
                    f"   margin x{margin:.1f}   (need x{CLIFF_MIN_MARGIN:.0f})",
                    (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
                    cv2.LINE_AA)
        
        return canvas




def embed(video_path, output_path, watermark: int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0

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

        first_half, second_half = get_middle_patches(frame.y)

        half = first_half if bit == 1 else second_half

        if half is not None:
            patch, y, x = half[0]
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


def get_blur_mask(radius,h,w,lower=True):

    Y,X = np.ogrid[:h,:w]
    # Distance of each DCT coefficient from the DC corner. Named `dist` rather than
    # reusing `radius`, which would shadow the cutoff passed in by the caller.
    dist = np.sqrt(X**2 + Y **2)

    if lower is False:
        mask = (dist>radius).astype(np.float32)

    else:
        mask = (dist<=radius).astype(np.float32)

    return mask
    


def blur_region(img, radius=RADIUS):
    h, w = img.shape[:2]
    dct_coeff = dct.dct2(img)

    mask = get_blur_mask(radius,h,w)

    filtered_dct = dct_coeff*mask
    filtered_patch = dct.idct2(filtered_dct)
    
    
    return filtered_patch


def get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view"):
    impv = BLUR_IMPAIRMENT_VIEW(INPUT, imp_path, out_dir)
    impv.start()
    impv.release()


if __name__ == "__main__":

    #embed(video_path=INPUT, output_path=OUTPUT, watermark=ONES)

    # detect(watermark_path=OUTPUT, org_video_path=INPUT)

    #transcode_n_times(OUTPUT, "blur", 6)

    get_frame_imp_analysis(imp_path=OUTPUT, out_dir="imp_view")
