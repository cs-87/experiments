from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import get_grid_patches, get_sift_patches, get_middle_patches
from utils.patch import get_nxn_block, SQUARE_SIZE
from tqdm import tqdm
import numpy as np
from utils.scene_change import detect_scene_change
from impairment_view import Impairment_View
import cv2
from utils.transcode_n import transcode_n_times

INPUT = "4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "leak.mp4"

BLOCK_SIZE = 8


ONES = (1 << 32) - 1
ZEROS = 0

TEMP_REDUNDANCY = 5
ALPHA = 2

C1 = (4, 5)
C2 = (5, 4)

WATERMARK = ONES

# Coefficient shift embed() applies to C1 of every LL block in the marked patch.
DELTA = ALPHA * 10

# Per-block decision threshold. Compression eats part of DELTA -- measured survival on
# out/wmk.mp4 is ~12 of 20 -- so half the nominal shift sits comfortably between the
# marked population and the ~0-mean noise of the unmarked one.
BLOCK_T = DELTA / 2

# Separation between the two patch scores below which the frame is called out as a weak
# decision. Deliberately looser than BLOCK_T: at BLOCK_T a third of the frames trip it,
# including ones separated 41 blocks to 3. A quarter of the nominal shift isolates the
# genuinely close ones -- empirically the first frame of each bit run.
MARGIN_T = DELTA / 4

# 256px patch -> 512px in the panel, so each LL block reads as a 64px cell.
PANEL_SCALE = 2

# BGR
GREEN = (80, 220, 80)
GREY = (150, 150, 150)
RED = (60, 60, 235)
AMBER = (40, 190, 235)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class DWT_DCT_IMP_VIEW(Impairment_View):
    """
    Renders, per frame, whether the left or the right middle patch carries the mark.

    The mark is a +DELTA shift of DCT coefficient C1 in every 8x8 block of the patch's
    level-2 Haar LL subband. A single block carries an SNR of only ~3 against re-encode
    noise, which is why the raw per-block map alone is unreadable; the mean over a
    patch's 64 blocks carries ~24 and is what actually decides the bit. The panel shows
    both: the thresholded block grid for texture, and the numeric score for the verdict.
    """

    def __init__(self, src_path, imp_path, out_dir, watermark=WATERMARK):
        super().__init__(src_path, imp_path, out_dir)
        self.bit_string, self.bit_length = get_bit_string(watermark)

    def _get_patch_scores(self, patch_src, patch_imp):
        """
        Signed per-block C1 deltas between the source and impaired patch.

        Returns (delta_LL, score, hits): the LL-domain delta map, its mean, and how many
        blocks clear BLOCK_T. Deltas stay signed and unscaled -- clipping them to [0, 255]
        would keep only the positive half of the unmarked patch's zero-mean noise and
        render it as speckle rather than flat black.
        """
        LL_src = dwt.get_dwt_coeff(patch_src).LL
        LL_imp = dwt.get_dwt_coeff(patch_imp).LL

        delta_LL = np.zeros_like(LL_src, dtype=np.float32)
        deltas = []

        for block_src, yb, xb in get_nxn_block(LL_src, block_size=BLOCK_SIZE):

            block_imp = LL_imp[yb[0]:yb[1], xb[0]:xb[1]]

            delta = dct.dct2(block_imp)[C1] - dct.dct2(block_src)[C1]

            delta_LL[yb[0]:yb[1], xb[0]:xb[1]] = delta
            deltas.append(delta)

        deltas = np.array(deltas, dtype=np.float32)

        return delta_LL, float(deltas.mean()), int((deltas > BLOCK_T).sum())

    def _render_block_map(self, delta_LL):
        """Binary block grid: white where the block clears BLOCK_T, black otherwise."""
        binary = np.where(delta_LL > BLOCK_T, 255, 0).astype(np.uint8)

        # NEAREST keeps the block edges hard -- interpolating would blur the grid back
        # into the smear this view exists to replace.
        scaled = cv2.resize(
            binary,
            (SQUARE_SIZE * PANEL_SCALE, SQUARE_SIZE * PANEL_SCALE),
            interpolation=cv2.INTER_NEAREST
        )

        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    def _compose_panel(self, left, right, decoded, expected, margin):
        """
        left/right are (block_map, score, hits). Draws both maps side by side with the
        winner outlined in green, and a footer carrying the verdict.
        """
        size = SQUARE_SIZE * PANEL_SCALE
        margin_px = 20
        header_h = 40
        label_h = 28
        stats_h = 56
        footer_h = 44

        width = margin_px * 3 + size * 2
        height = (margin_px + header_h + label_h + size + stats_h + footer_h)

        panel = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.putText(panel, f"frame {self.frame_index}",
                    (margin_px, margin_px + 24), FONT, 0.7, WHITE, 1, cv2.LINE_AA)

        y_label = margin_px + header_h + 20
        y_map = margin_px + header_h + label_h
        y_stats = y_map + size + 24

        sides = (
            ("LEFT  (bit 1)", left, margin_px, decoded == 1),
            ("RIGHT (bit 0)", right, margin_px * 2 + size, decoded == 0),
        )

        for name, (block_map, score, hits), x, is_winner in sides:

            colour = GREEN if is_winner else GREY

            cv2.putText(panel, name, (x, y_label),
                        FONT, 0.62, colour, 2 if is_winner else 1, cv2.LINE_AA)

            panel[y_map:y_map + size, x:x + size] = block_map

            cv2.rectangle(panel, (x - 2, y_map - 2),
                          (x + size + 1, y_map + size + 1),
                          colour, 2 if is_winner else 1)

            cv2.putText(panel, f"score {score:+6.2f}", (x, y_stats),
                        FONT, 0.58, colour, 1, cv2.LINE_AA)
            cv2.putText(panel, f"hits  {hits:3d}/64", (x, y_stats + 24),
                        FONT, 0.58, colour, 1, cv2.LINE_AA)

        y_footer = height - 16

        ok = decoded == expected
        verdict, verdict_colour = ("OK", GREEN) if ok else ("MISMATCH", RED)

        cv2.putText(panel,
                    f"DECODED {decoded}   EXPECTED {expected}",
                    (margin_px, y_footer), FONT, 0.65, WHITE, 1, cv2.LINE_AA)

        cv2.putText(panel, verdict, (margin_px + 330, y_footer),
                    FONT, 0.65, verdict_colour, 2, cv2.LINE_AA)

        cv2.putText(panel, f"margin {margin:5.2f}", (margin_px + 470, y_footer),
                    FONT, 0.65, WHITE, 1, cv2.LINE_AA)

        # Expected on the first frame of a bit run, where the encoder's inter-prediction
        # still carries the previous run's mark forward into the other patch.
        if margin < MARGIN_T:
            cv2.putText(panel, "LOW CONFIDENCE", (margin_px + 640, y_footer),
                        FONT, 0.65, AMBER, 2, cv2.LINE_AA)

        return panel

    def get_frame_diff(self):

        src_frame = self.get_source_frame()
        if src_frame is None:
            return None

        imp_frame = self.get_imp_frame()
        if imp_frame is None:
            return None

        # get_source_frame/get_imp_frame already hand back YUV. Converting again here
        # would treat Y/U/V as B/G/R and blend chroma into the luma plane, diluting the
        # mark from ~12 to ~1.3 against an unchanged ~3 noise floor.
        src_frame = src_frame[:, :, 0]
        imp_frame = imp_frame[:, :, 0]

        # Match impaired frame to source dimensions
        imp_frame = cv2.resize(
            imp_frame,
            (src_frame.shape[1], src_frame.shape[0])
        )

        patches = get_middle_patches(src_frame)

        if patches is None:
            return None

        src_left_patch, s_l_y, s_l_x = patches[0][0]
        src_right_patch, s_r_y, s_r_x = patches[1][0]

        if src_left_patch is None or src_right_patch is None:
            return None

        # Extract corresponding patches from impaired frame
        imp_left_patch = imp_frame[
            s_l_y[0]:s_l_y[1],
            s_l_x[0]:s_l_x[1]
        ]

        imp_right_patch = imp_frame[
            s_r_y[0]:s_r_y[1],
            s_r_x[0]:s_r_x[1]
        ]

        delta_l, score_l, hits_l = self._get_patch_scores(
            src_left_patch,
            imp_left_patch
        )

        delta_r, score_r, hits_r = self._get_patch_scores(
            src_right_patch,
            imp_right_patch
        )

        # embed() marks patches[0] -- the left patch -- for bit 1.
        decoded = 1 if score_l > score_r else 0
        margin = abs(score_l - score_r)

        bit_index = (self.frame_index // TEMP_REDUNDANCY) % self.bit_length
        expected = int(self.bit_string[bit_index])

        return self._compose_panel(
            (self._render_block_map(delta_l), score_l, hits_l),
            (self._render_block_map(delta_r), score_r, hits_r),
            decoded,
            expected,
            margin
        )


def embed(video_path, output_path, watermark: int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        wmk_frame_y = frame.y.copy()

        patches = get_middle_patches(frame.y)

        half = patches[0] if bit == 1 else patches[1]

        if not half:
            continue

        for patch, y, x in half:

            dwt_coeffs = dwt.get_dwt_coeff(patch)

            patch_LL = dwt_coeffs.LL

            for block, yb, xb in get_nxn_block(patch_LL, block_size=BLOCK_SIZE):

                if block is None:
                    print("block empty")
                    continue

                block_LL_dct_coeff = dct.dct2(block)

                c1 = block_LL_dct_coeff[C1]

                block_LL_dct_coeff[C1] = c1 + ALPHA*10

                patch_LL[yb[0]:yb[1], xb[0]:xb[1]
                         ] = dct.idct2(block_LL_dct_coeff)

        dwt_coeffs.LL = patch_LL

        recon = dwt.reconstruct_frame(dwt_coeffs)
        wmk_frame_y[y[0]:y[1], x[0]:x[1]] = np.clip(recon, 0, 255)

        frame.set_y(wmk_frame_y)

        video_io.write_frame(frame, output_path)
    video_io.release()


# embed(video_path=INPUT, output_path=OUTPUT, watermark=WATERMARK)

transcode_n_times(input_video=OUTPUT, output_dir="transcode", n=6)
imp_view = DWT_DCT_IMP_VIEW(INPUT, "transcode/transcoded_6.mp4", "imp_view")

# imp_view = DWT_DCT_IMP_VIEW(INPUT, OUTPUT, "imp_view")

imp_view.start()
imp_view.release()
# detect(org_video_path=INPUT, watermark_path=LEAKED)
