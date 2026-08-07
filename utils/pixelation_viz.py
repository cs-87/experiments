"""
Visualisations for the pixelation detector.

Kept separate from utils/pixelation.py so that module honours the numpy+OpenCV-only
constraint and stays importable in environments without matplotlib.

    from utils.pixelation_viz import plot_pixelation_report
    plot_pixelation_report(half, block_size=4, save_path="out/report.png")
"""

import cv2
import matplotlib

matplotlib.use("Agg")   # file output only; no display needed on a headless box
import matplotlib.pyplot as plt
import numpy as np

from utils.pixelation import (
    THRESHOLDS,
    _boundary_interior_ratio,
    _fold,
    _gradient_profiles,
    _to_gray,
    block_variance_map,
    detect_pixelation,
    estimate_block_size,
    fft_spectrum,
)


def _crop_for_display(gray, max_side=512):
    """Large frames hide the block grid once matplotlib downsamples them."""
    h, w = gray.shape
    if max(h, w) <= max_side:
        return gray
    return gray[:max_side, :max_side]


def plot_pixelation_report(image, block_size=None, save_path=None, title=None):
    """
    Render every detector cue for one image region as a single figure.

    Panels: the region itself, block variance heatmap, Sobel gradient magnitude,
    Laplacian, 2D FFT log-magnitude, the folded gradient phase profile (the clearest
    single view of whether a grid exists), and the score breakdown.

    Returns the matplotlib Figure. Pass save_path to also write a PNG.
    """
    gray = _to_gray(image)
    if block_size is None:
        block_size = estimate_block_size(gray)

    result = detect_pixelation(gray, block_size=block_size)
    view = _crop_for_display(gray)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    header = title or "pixelation report"
    fig.suptitle(
        f"{header} — confidence {result.confidence:.3f} @ block size {block_size}",
        fontsize=15,
        weight="bold",
    )

    # 1. the region
    axes[0, 0].imshow(view, cmap="gray", interpolation="nearest")
    axes[0, 0].set_title(f"region (top {view.shape[0]}x{view.shape[1]})")

    # 2. block variance heatmap: pixelated blocks are uniformly dark
    var_map = block_variance_map(view, block_size)
    im = axes[0, 1].imshow(var_map, cmap="inferno", interpolation="nearest")
    axes[0, 1].set_title(f"block variance (per {block_size}x{block_size})")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # 3. Sobel gradient magnitude: a grid shows as a regular lattice
    sobel = np.hypot(
        cv2.Sobel(view, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(view, cv2.CV_32F, 0, 1, ksize=3),
    )
    axes[0, 2].imshow(sobel, cmap="magma", interpolation="nearest")
    axes[0, 2].set_title("sobel gradient magnitude")

    # 4. Laplacian
    lap = cv2.Laplacian(view, cv2.CV_32F, ksize=3)
    axes[0, 3].imshow(np.abs(lap), cmap="viridis", interpolation="nearest")
    axes[0, 3].set_title("|laplacian|")

    # 5. 2D FFT: a grid of period N puts symmetric peaks at multiples of size/N
    axes[1, 0].imshow(fft_spectrum(view), cmap="gray", interpolation="nearest")
    axes[1, 0].set_title("2D FFT log-magnitude")

    # 6. folded phase profile: the money shot. One tall bar at the last offset means
    #    every block boundary lands on the same phase -- i.e. a real grid.
    col_profile, row_profile = _gradient_profiles(gray)
    folded_cols = _fold(col_profile, block_size)
    folded_rows = _fold(row_profile, block_size)
    offsets = np.arange(block_size)
    width = 0.4
    axes[1, 1].bar(offsets - width / 2, folded_cols, width, label="columns")
    axes[1, 1].bar(offsets + width / 2, folded_rows, width, label="rows")
    axes[1, 1].axvline(block_size - 1, color="red", ls="--", lw=1, label="boundary offset")
    axes[1, 1].set_title("gradient folded mod block size")
    axes[1, 1].set_xlabel("offset within block")
    axes[1, 1].legend(fontsize=8)

    # 7. profile spectrum, with the grid harmonic marked
    n = len(col_profile)
    spectrum = np.abs(np.fft.rfft(col_profile - col_profile.mean()))
    axes[1, 2].semilogy(spectrum, lw=0.7)
    harmonic = min(int(round(n / block_size)), len(spectrum) - 1)
    axes[1, 2].axvline(harmonic, color="red", ls="--", lw=1, label=f"n/{block_size}")
    axes[1, 2].set_title("column-profile spectrum")
    axes[1, 2].set_xlabel("frequency bin")
    axes[1, 2].legend(fontsize=8)

    # 8. score breakdown
    axes[1, 3].axis("off")
    names = list(result.scores)
    values = [result.scores[n_] for n_ in names]
    colors = ["#2a9d8f" if v < 0.5 else "#e76f51" for v in values]
    bars = axes[1, 3].barh(names, values, color=colors)
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].axis("on")
    axes[1, 3].set_title(f"metric scores → {result.confidence:.3f}")
    for bar, value in zip(bars, values):
        axes[1, 3].text(
            min(value + 0.02, 0.9), bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center", fontsize=9,
        )

    for ax in axes.flat:
        if ax.get_images():
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"wrote {save_path}")
    return fig


def plot_halves_comparison(frame_y, block_size=None, save_path=None):
    """
    Side-by-side confidence for the two halves of a frame -- the view that explains a
    single bit decision in src/pixelate_patch.py:detect.
    """
    from utils.patch import get_two_halves

    first_half, second_half = get_two_halves(frame_y)
    if block_size is None:
        block_size = estimate_block_size(frame_y)

    left = detect_pixelation(first_half[0], block_size=block_size)
    right = detect_pixelation(second_half[0], block_size=block_size)
    bit = 1 if left.confidence > right.confidence else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"bit = {bit}  (left {left.confidence:.3f} vs right {right.confidence:.3f}, "
        f"block size {block_size})",
        fontsize=14, weight="bold",
    )

    axes[0].imshow(first_half[0], cmap="gray")
    axes[0].set_title(f"left half — {left.confidence:.3f}")
    axes[1].imshow(second_half[0], cmap="gray")
    axes[1].set_title(f"right half — {right.confidence:.3f}")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    names = list(left.scores)
    y = np.arange(len(names))
    axes[2].barh(y - 0.2, [left.scores[n] for n in names], 0.4, label="left")
    axes[2].barh(y + 0.2, [right.scores[n] for n in names], 0.4, label="right")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names)
    axes[2].set_xlim(0, 1)
    axes[2].set_title("per-metric scores")
    axes[2].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"wrote {save_path}")
    return fig


if __name__ == "__main__":
    # Example usage: report the pixelated and clean halves of the watermarked clip.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.patch import get_two_halves
    from utils.video import Video_IO

    os.makedirs("out", exist_ok=True)
    video_io = Video_IO("out/wmk.mp4")
    frame = video_io.read_frame()
    video_io.release()

    size = estimate_block_size(frame.y)
    first_half, second_half = get_two_halves(frame.y)
    plot_pixelation_report(second_half[0], size, "out/report_pixelated.png", "right half")
    plot_pixelation_report(first_half[0], size, "out/report_clean.png", "left half")
    plot_halves_comparison(frame.y, size, "out/report_halves.png")
