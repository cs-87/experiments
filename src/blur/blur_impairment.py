import cv2
from utils.scene_change import detect_scene_change
from src.impairment_view import Impairment_View
import utils.dct as dct
from src.blur.blur import RADIUS, get_blur_mask
import numpy as np
from utils.lightglue import LightGluePatchMatcher, warp_patch

from src.blur.patch import SQUARE_SIZE, get_patch, get_best_patch_in_two_halves, select_candidate, get_middle_patches, get_middle_split_col
from src.blur.patch import get_half_patches_grid, get_grid_patches

# Width of the annuli the view compares either side of RADIUS, in coefficients.
CLIFF_BAND = 24

# Below this below/above energy ratio the frame is reported as undecided rather than
# guessed at. Natural content sits near x2-3; a surviving blur cliff is x10 upwards.
CLIFF_MIN_MARGIN = 4.0

# hf_loss_score (non-blind) is bounded in [0, 1] -- the fraction of the original
# patch's HF energy that blur_region() removed -- so unlike the blind cliff ratio a
# plain absolute cutoff is the right test, not a ratio: a marked cell should sit close
# to 1 (blur_region nulled it), an untouched cell close to 0 (the codec barely touches
# HF energy at this radius). A ratio would explode on two untouched cells that both
# happen to score near zero, exactly like it does for the blind case's noise floor.
HF_LOSS_MIN_SCORE = 0.5
HF_LOSS_MIN_MARGIN = 0.2


LIGHTGLUE = False

class BLUR_IMPAIRMENT_VIEW(Impairment_View):

    def __init__(self, src_path, imp_path, out_dir, blind=False):
            super().__init__(src_path, imp_path, out_dir)

            self.blind = blind

            # One pool per half, because the view renders both halves without knowing the bit.
            # Within a bit run embed() marked the same half on every frame, so that half's
            # pool matches embed()'s exactly while the other's is a harmless what-if.
            # Divergence accumulates only across runs, and the per-run flush erases it.
            self.pool_first, self.pool_second = [], []

            if LIGHTGLUE:
    
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


        if LIGHTGLUE:
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

            if LIGHTGLUE:

                imp_patch = warp_patch(
                    imp_frame,x=x[0],y=y[0],homography=self.homography,patch_size=SQUARE_SIZE
                )

            else:
                imp_patch, y,x = get_patch(x[0],y[0],imp_frame)

            # A cell whose projection leaves the impaired frame has nothing to measure.
            # Leave its residual black and keep it out of the scoring rather than
            # scoring an absence -- a partly-missing patch reads as a spectral cliff and
            # would win the vote for the wrong reason.
            if imp_patch is None:
                continue

            # Keep ONLY high-frequency DCT coefficients
            mask = get_blur_mask(radius=RADIUS, h=h, w=w, lower=False)

            filtered_imp_dct = dct.dct2(imp_patch) * mask
            filtered_org_dct = dct.dct2(patch) * mask

            # Back to the spatial domain, as magnitude -- the backdrop is only there to
            # give the eye some context for where the scores sit.

            if self.blind:
                residual[y[0]:y[1], x[0]:x[1]] = np.abs(dct.idct2(filtered_imp_dct))

            else:
                # What got removed (org - imp), not what's left -- a real mark is the
                # brightest thing on the canvas, everywhere untouched goes to black.
                residual[y[0]:y[1], x[0]:x[1]] = np.abs(dct.idct2(filtered_org_dct - filtered_imp_dct))

            # Scored on the *impaired* patch, never the original: the cliff is the mark's
            # signature as it survived encoding, and org_frame's own patch carries none.
            if self.blind:
                score = self.cliff_score(imp_patch)
            else:
                score= self.hf_loss_score(org_patch=patch,imp_patch=imp_patch)
            cells.append((score, y, x))


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

        for score, y, x in cells:
            # Every cell gets its grid box and score, so a misfire is visible as a
            # near-tie instead of hiding behind a confident-looking red box.
            cv2.rectangle(canvas, (x[0], y[0]), (x[1] - 1, y[1] - 1), (40, 40, 40), 1)
            cv2.putText(canvas, f"{score:.2f}", (x[0] + 8, y[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1, cv2.LINE_AA)

        # The boundary that separates bit 1 from bit 0. Same helper the embedder splits
        # on -- deriving a midpoint here instead would disagree with it whenever the
        # frame centre is not itself on a grid line.
        split = get_middle_split_col(W)
        cv2.line(canvas, (split, 0), (split, H), (0, 140, 200), 2, cv2.LINE_AA)

        if self.blind:
            decided, score_line = self.blind_decision(best, runner)
        else:
            decided, score_line = self.non_blind_decision(best, runner)

        score, y, x = best
        colour = (0, 230, 0) if decided else (0, 160, 230)

        cv2.rectangle(canvas, (x[0], y[0]), (x[1] - 1, y[1] - 1), colour, 3)
        cv2.putText(canvas, f"{score:.2f}", (x[0] + 8, y[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        bit = 1 if (x[0] + x[1]) // 2 < split else 0
        side = "LEFT" if bit == 1 else "RIGHT"

        if decided:
            headline = f"BIT = {bit}   ({side} of centre)"
        else:
            headline = f"UNDECIDED  (would read {bit}, {side})"

        cv2.putText(canvas, headline, (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, colour, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {self.frame_index}   {score_line}",
                    (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
                    cv2.LINE_AA)

        return canvas

    def blind_decision(self, best, runner):
        """Cliff-ratio test: confident only once the winner clears the field by a
        multiple, since a surviving blur cliff reads x10 upwards while natural content
        sits near x2-3."""
        best_score, runner_score = best[0], runner[0]
        margin = best_score / (runner_score + 1e-9)
        decided = margin >= CLIFF_MIN_MARGIN
        score_line = (f"cliff {best_score:.1f} vs next {runner_score:.1f}"
                      f"   margin x{margin:.1f}   (need x{CLIFF_MIN_MARGIN:.0f})")
        return decided, score_line

    def non_blind_decision(self, best, runner):
        """Absolute test on hf_loss_score: it is already normalised to [0, 1] against
        the known original patch, so unlike the blind cliff ratio there is no noise
        floor to divide by -- the winner just has to clear an absolute HF-loss level
        and lead the runner-up by an absolute margin."""
        best_score, runner_score = best[0], runner[0]
        margin = best_score - runner_score
        decided = best_score >= HF_LOSS_MIN_SCORE and margin >= HF_LOSS_MIN_MARGIN
        score_line = (f"hf-loss {best_score:.2f} vs next {runner_score:.2f}"
                      f"   margin {margin:.2f}"
                      f"   (need >={HF_LOSS_MIN_SCORE:.2f}, margin >={HF_LOSS_MIN_MARGIN:.2f})")
        return decided, score_line


    def hf_loss_score(self, org_patch, imp_patch):
        org = org_patch.astype(np.float32)
        imp = imp_patch.astype(np.float32)

        org_dct = dct.dct2(org)
        imp_dct = dct.dct2(imp)

        mask = get_blur_mask(
            radius=RADIUS,
            h=org.shape[0],
            w=org.shape[1],
            lower=False
        ).astype(bool)

        org_hf = np.sum(org_dct[mask] ** 2)
        imp_hf = np.sum(imp_dct[mask] ** 2)

        # Fraction of original HF energy that survived
        survival = imp_hf / (org_hf + 1e-9) # 1e-9 to avoid zero division

        # Larger = stronger evidence of HF removal
        return 1.0 - survival


