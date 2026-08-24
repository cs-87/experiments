from utils.video import Video_IO
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from src.blur.patch import SQUARE_SIZE, get_two_halves, get_best_patch_in_two_halves, select_candidate, get_middle_patches, get_middle_split_col
from utils.patch import get_half_patches_grid, get_grid_patches
from utils.pixelation import detect_pixelation
from utils.lightglue import LightGluePatchMatcher, warp_patch
from tqdm import tqdm
import numpy as np
import cv2
import random

from utils.scene_change import detect_scene_change
from src.impairment_view import Impairment_View
from utils.transcode_n import transcode_n_times





INPUT = "4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "IMG_2220.mov"

SIFT_PATCHES = False

ONES = (1 << 32) - 1
ZEROS = 0

ALPHA = 20



# Width of the annuli the view compares either side of RADIUS, in coefficients.
CLIFF_BAND = 24

# Below this below/above energy ratio the frame is reported as undecided rather than
# guessed at. Natural content sits near x2-3; a surviving blur cliff is x10 upwards.
CLIFF_MIN_MARGIN = 4.0


import numpy as np
import utils.dct

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
TEMP_REDUNDANCY = 3

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



def score_patch_for_watermark(patch, w_delta=1.0, w_texture=0.5, w_clip=2.0):
    """
    How good `patch` is as a home for the mark, given the modification the embedder
    would make: `marked = blur_region(patch); delta = marked - patch`.

    score = w_delta * delta_strength - w_clip * clip_risk + w_texture * texture

    Each term is normalised to the same 0-1-ish scale (a fraction of the 255 pixel
    range) so the weights trade off directly against each other instead of one term
    dominating just because its units happen to be bigger:

    - delta_strength: RMS of `delta` / 255. A stronger, more robust mark scores higher.
    - texture: std of the original `patch` / 255. Busy content masks the mark visually
      and tolerates a stronger one before it becomes noticeable, so it adds to the score.
    - clip_risk: how far `marked = patch + delta` overshoots [0, 255], averaged over the
      patch and divided by 255. Clipping quietly weakens the mark (the embedder clips
      before writing back) and can leave a visible flat edge, so it is the most heavily
      weighted, subtracted term.

    Higher is better. Pass as the `key` to select_candidate as
    `lambda patch: score_patch_for_watermark(patch, blur_region(patch) - patch)`.
    """


    wmk_patch = blur_region(patch)
    delta =wmk_patch - patch
    patch = patch.astype(np.float32)
    delta = delta.astype(np.float32)
    marked = patch + delta

    delta_strength = np.sqrt(np.mean(delta ** 2)) / 255.0
    texture = np.std(patch) / 255.0

    overshoot = np.maximum(0, marked - 255) + np.maximum(0, -marked)
    clip_risk = np.mean(overshoot) / 255.0

    return w_delta * delta_strength + w_texture * texture - w_clip * clip_risk
