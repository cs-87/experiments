"""
Sweep harness for the blur watermark: (RADIUS x TR x condition) on one fixed source.

This exists because the previous sweep could not answer the question it was asked.
`blur_recv_two_patches.py` read `INPUT = f"./inputs/{TEMP_REDUNDANCY}.mp4"`, so raising
TEMP_REDUNDANCY silently swapped the source clip -- `inputs/` is symlinked so that
32 bits x TR frames exactly fills each one, which means TR=3 was measured on a 4-second
0.24 Mbps intro and TR=120 on 128 seconds of quite different material. The observed
"robustness falls as TR rises" was the clips changing, not TR. Everything here takes its
frames from one video and one video only, and TR moves nothing but the frame->bit map.

Three other things had to be held still before any number meant anything:

  encode      -- Video_IO wrote MPEG-4 Part 2 at whatever bitrate OpenCV chose, measured
                 between 3 and 20 Mbps on these clips. Now H.264 at a fixed CRF.
  frame count -- every configuration gets the same N frames, so each bit gets N/32
                 frames whatever TR is, and TR only controls how those frames are
                 *clumped*. That makes contiguous-versus-interleaved a fair comparison
                 rather than a comparison of two frame budgets.
  detector    -- the energy baseline and the matched filter are computed in the same
                 frame pass, from the same patches, after the same alignment. Two
                 detectors reading different pixels is two experiments, not an A/B.

Attacks reuse src/retrieval/capture_sim.py rather than growing a second attack model
here, and are applied in memory frame by frame, so a condition costs a decode rather
than another 1080p file on disk.

  # the full grid the plan asks for
  python -m src.blur.eval_harness sweep

  # one cell of it, printed rather than appended
  python -m src.blur.eval_harness sweep --radius 110 --tr 30 --conditions clean --no-log

  # visibility and false-positive checks
  python -m src.blur.eval_harness visibility --radius 110 --tr 30
  python -m src.blur.eval_harness fp --radius 110 --tr 30

Every sweep row is appended to src/blur/FINDINGS.md as markdown and to
src/blur/findings.jsonl as one JSON object, so a later session picks up from the file
rather than from a scrollback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.blur.blur_detect_mf import (
    GeometricAligner,
    MFConfig,
    bit_correct_rate,
    decode,
    energy_decode,
    frame_accuracy,
    frames_to_payload,
    hard_bits,
    iter_y,
    scan,
)
from src.blur.blur_recv_two_patches import embed
from src.blur.patch import middle_pair_cells
from src.retrieval.capture_sim import CaptureSim, get_condition
from utils.bit import BIT_LENGTH
from utils.video import DEFAULT_CRF

SOURCE = "inputs/120.mp4"
PAYLOAD = 0xCAFECAFE
# A different payload with the same population count, so a false-positive test cannot
# be passed by accident on a bit-balance cue.
OTHER_PAYLOAD = 0x0F3C5A69

# 960 frames = 32 s at 30 fps: exactly one payload cycle at TR=30, thirty cycles at
# TR=1, and the same content either way.
FRAMES = 960

MARKED_DIR = Path("outputs/harness")
FINDINGS_MD = Path("src/blur/FINDINGS.md")
FINDINGS_JSONL = Path("src/blur/findings.jsonl")

STATS = ("a", "z", "nc")


def use_tag(tag):
    """
    Point the findings files at a per-shard name.

    The grid is ~18 hours on this box run end to end, so it is sharded across a few
    processes. Two processes appending a multi-line markdown table to one file will
    interleave their rows -- the JSONL append survives it, the table does not -- so each
    shard writes its own pair and the shards are concatenated afterwards. No tag keeps
    the original single-file behaviour.
    """
    global FINDINGS_MD, FINDINGS_JSONL
    if not tag:
        return
    FINDINGS_MD = FINDINGS_MD.with_name(f"FINDINGS.{tag}.md")
    FINDINGS_JSONL = FINDINGS_JSONL.with_name(f"findings.{tag}.jsonl")


# ---------------------------------------------------------------------------
# marked videos, cached on disk
# ---------------------------------------------------------------------------


def marked_path(source, radius, tr, cluster_k, frames, crf, payload):
    stem = Path(source).stem
    tag = "il" if tr == 1 else f"tr{tr}"
    return MARKED_DIR / (f"{stem}_n{frames}_r{radius}_{tag}_k{cluster_k}"
                         f"_crf{crf}_{payload:08x}.mp4")


def build_marked(source=SOURCE, radius=110, tr=30, cluster_k=1, frames=FRAMES,
                 crf=DEFAULT_CRF, payload=PAYLOAD, rebuild=False, verbose=True):
    """
    Embed once per (radius, TR, K) and keep it, because the conditions all attack the
    same marked video and re-encoding it per condition would put an extra, uncontrolled
    generation of coding between the embedder and the measurement.

    TR=1 *is* the interleaved map -- `(i // 1) % 32` is `i % 32` -- so it is spelled
    that way rather than carried as a second flag that could disagree with the first.
    """
    out = marked_path(source, radius, tr, cluster_k, frames, crf, payload)
    if out.exists() and not rebuild:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    embed(video_path=source, output_path=str(out), watermark=payload,
          temp_redundancy=tr, radius=radius, interleave=(tr == 1),
          cluster_k=cluster_k, max_frames=frames, crf=crf, verbose=verbose)
    return out


def transcode(path, crf, preset="medium"):
    """One extra generation of H.264, as the mildest realistic attack."""
    out = Path(path).with_suffix(f".x{crf}.mp4")
    if out.exists():
        return out
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
         "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", preset, str(out)],
        check=True)
    return out


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------


def leak_frames(marked, condition, frames, seed=0):
    """
    Luma planes of the leak under `condition`.

    "clean" is the marked video itself -- already one H.264 generation past the
    original, which is the distribution encode, not an attack. "crfNN" adds a second
    generation. Everything else is a capture_sim condition applied per frame in memory.
    """
    if condition == "clean":
        yield from iter_y(marked, limit=frames)
        return

    if condition.startswith("crf"):
        yield from iter_y(transcode(marked, int(condition[3:])), limit=frames)
        return

    cap = cv2.VideoCapture(str(marked))
    if not cap.isOpened():
        raise ValueError(f"could not open {marked}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sim = CaptureSim(get_condition(condition), (w, h), seed=seed)
        i = 0
        while frames is None or i < frames:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(sim.apply(frame), cv2.COLOR_BGR2YUV)[:, :, 0]
            i += 1
    finally:
        cap.release()


# Conditions whose geometry moves, and which therefore need stage 0 to do anything.
# Listing them rather than always aligning keeps the cost off the runs that cannot
# benefit -- and makes it visible which rows the aligner is responsible for.
GEOMETRIC = {"mild", "moderate", "severe", "extreme", "perspective_only"}


# ---------------------------------------------------------------------------
# one measurement
# ---------------------------------------------------------------------------


def measure(source, marked, condition, cfg, frames, payload,
            lightglue=False, homography_every=30, refine=True, progress=False):
    """One (config, condition) cell: both detectors, from one pass over the frames."""
    aligner = None
    if condition in GEOMETRIC:
        aligner = GeometricAligner(lightglue=lightglue, every=homography_every,
                                   refine=refine)

    t0 = time.time()
    result = scan(iter_y(source, limit=frames),
                  leak_frames(marked, condition, frames),
                  cfg, aligner=aligner, progress=progress, total=frames)
    elapsed = time.time() - t0

    row = {
        "condition": condition,
        "radius": cfg.radius,
        "tr": cfg.temp_redundancy,
        "map": "interleaved" if cfg.interleave else "contiguous",
        "cluster_k": cfg.cluster_k,
        "frames": int(result["frame"].size),
        "skipped": int(result["skipped"]),
        "seconds": round(elapsed, 1),
    }

    # The baseline is read at phase 0: it has no phase search, which is part of what is
    # being compared, not a handicap imposed on it.
    base = energy_decode(result, cfg=cfg, phase=0)
    row["bcr_energy"] = bit_correct_rate(payload, base["bit_string"], cfg.bit_length)

    for stat in STATS:
        got = decode(result, cfg=cfg, stat=stat)
        row[f"bcr_{stat}"] = bit_correct_rate(payload, hard_bits(got["L"]),
                                              cfg.bit_length)
        row[f"phase_{stat}"] = got["phase"]
        row[f"presence_{stat}"] = round(got["presence"], 2)
        row[f"frameacc_{stat}"] = round(
            frame_accuracy(result, payload, cfg=cfg, stat=stat, phase=got["phase"]), 4)
        # Weakest bit: 32/32 with a margin of nothing is not the same result as 32/32
        # with room to spare, and only the second one is worth acting on.
        row[f"minz_{stat}"] = round(float(np.min(np.abs(got["bit_z"]))), 2)

    best = max(STATS, key=lambda s: (row[f"bcr_{s}"], row[f"minz_{s}"]))
    curve = frames_to_payload(result, payload, cfg=cfg, stat=best)
    row["best_stat"] = best
    row["frames_to_32"] = curve["first_full"]
    row["curve"] = curve["curve"]
    return row, result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

COLUMNS = [
    ("condition", "condition", "{}"),
    ("radius", "R", "{}"),
    ("tr", "TR", "{}"),
    ("map", "map", "{}"),
    ("cluster_k", "K", "{}"),
    ("bcr_energy", "energy", "{}/32"),
    ("bcr_a", "mf a", "{}/32"),
    ("bcr_z", "mf z", "{}/32"),
    ("bcr_nc", "mf nc", "{}/32"),
    ("frameacc_a", "frame acc a", "{:.3f}"),
    ("presence_a", "presence a", "{:.2f}"),
    ("minz_a", "min |z| a", "{:.2f}"),
    ("frames_to_32", "frames->32", "{}"),
]


def format_table(rows):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = []
    for row in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            cells.append("-" if value is None else fmt.format(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def sweep_key(condition, radius, tr, cluster_k):
    """Identity of one measured cell, used to skip it on a resumed run."""
    return f"{condition}|{radius}|{tr}|{cluster_k}"


def completed_cells():
    """
    Cells already measured, read back from the JSONL.

    The grid is hours long, so it has to survive being killed. Every row is flushed to
    disk as it is produced and the run is restartable from the file rather than from the
    beginning -- the marked videos were already cached this way, the measurements were
    not, and a shard killed near the end used to lose every cell it had measured.
    """
    if not FINDINGS_JSONL.exists():
        return set()
    keys = set()
    for line in FINDINGS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A row half-written when the process died. The cell simply gets remeasured.
            continue
        if row.get("kind") in (None, "sweep") and "condition" in row:
            keys.add(sweep_key(row["condition"], row["radius"], row["tr"],
                               row.get("cluster_k", 1)))
    return keys


def start_findings(header):
    """Open a run's markdown block once, so rows can be appended to it one at a time."""
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not FINDINGS_MD.exists():
        FINDINGS_MD.write_text(
            "# Blur watermark: measured results\n\n"
            "Appended by `src/blur/eval_harness.py`. Every row is one\n"
            "(radius, TR, cluster, condition) cell measured on one fixed source video\n"
            "with a fixed CRF, both detectors read from the same frame pass.\n"
            "The machine-readable copy is `findings.jsonl` beside this file.\n")
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    with FINDINGS_MD.open("a") as fh:
        fh.write(f"\n\n## {header}\n\n{head}\n{rule}\n")


def append_row(row, stamp=None):
    """
    Flush one measured cell to both findings files immediately.

    Each write is a single short append, which the OS keeps atomic, so a kill between
    cells truncates the file at a row boundary rather than corrupting it.
    """
    cells = []
    for key, _, fmt in COLUMNS:
        value = row.get(key)
        cells.append("-" if value is None else fmt.format(value))
    with FINDINGS_MD.open("a") as fh:
        fh.write("| " + " | ".join(cells) + " |\n")
        fh.flush()
    payload = dict(row)
    if stamp:
        payload["stamp"] = stamp
    with FINDINGS_JSONL.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
        fh.flush()


def append_findings(rows, header):
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not FINDINGS_MD.exists():
        FINDINGS_MD.write_text(
            "# Blur watermark: measured results\n\n"
            "Appended by `src/blur/eval_harness.py`. Every row is one\n"
            "(radius, TR, cluster, condition) cell measured on one fixed source video\n"
            "with a fixed CRF, both detectors read from the same frame pass.\n"
            "The machine-readable copy is `findings.jsonl` beside this file.\n")
    with FINDINGS_MD.open("a") as fh:
        fh.write(f"\n\n## {header}\n\n{format_table(rows)}\n")
    with FINDINGS_JSONL.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim(a, b):
    """
    Global SSIM on the luma patch, gaussian-weighted, standard constants.

    Written out rather than pulled from skimage to keep the harness's dependencies the
    same as the rest of the tree.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    ssim = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)))
    return float(ssim.mean())


def visibility(source, marked, frames, cluster_k, dump_dir=None):
    """
    What the mark costs to look at: PSNR and SSIM over the cells that carry it, PSNR
    over the whole frame for scale, and a temporal flicker term.

    The flicker term is the point of measuring this at all once the map is interleaved.
    Contiguous runs hold a cell blurred for TR frames, which a codec is happy to keep
    via SKIP macroblocks and an eye reads as a soft patch; interleaving flips which side
    is blurred every frame, and a 240x240 block pulsing at up to 15 Hz is a different
    artefact entirely.

    It has to be measured on something the mark actually changes. `flicker_luma_std` --
    the standard deviation over time of the cell's mean luma difference -- cannot do it:
    blur_region nulls the DCT coefficients *beyond* the cutoff and DC is not one of
    them, so a blurred cell has exactly the original's mean luma and the statistic reads
    the same codec noise (~0.2) marked or not. It is kept only to show that.

    `flicker_rms_std` is the real one: the per-cell RMS difference from the original
    rises when that cell is blurred and falls when it is not, so its standard deviation
    over time is precisely the pulsing amplitude. `mark_rms` is that difference's mean,
    i.e. how strong the mark is in the first place -- flicker only means something
    relative to it, which is what `flicker_ratio` reports.
    """
    org_cap = cv2.VideoCapture(str(source))
    imp_cap = cv2.VideoCapture(str(marked))
    patch_psnr, patch_ssim, full_psnr = [], [], []
    diffs = None
    rms = None

    try:
        for _ in range(frames):
            ok_a, fa = org_cap.read()
            ok_b, fb = imp_cap.read()
            if not (ok_a and ok_b):
                break
            ya = cv2.cvtColor(fa, cv2.COLOR_BGR2YUV)[:, :, 0]
            yb = cv2.cvtColor(fb, cv2.COLOR_BGR2YUV)[:, :, 0]
            full_psnr.append(_psnr(ya, yb))

            pairs = middle_pair_cells(ya, count=cluster_k)
            if diffs is None:
                diffs = [[] for _ in range(2 * len(pairs))]
                rms = [[] for _ in range(2 * len(pairs))]
            for p, pair in enumerate(pairs):
                for side, (patch, y, x) in enumerate(pair):
                    other = yb[y[0]:y[1], x[0]:x[1]]
                    patch_psnr.append(_psnr(patch, other))
                    patch_ssim.append(_ssim(patch, other))
                    d = other.astype(np.float64) - patch.astype(np.float64)
                    diffs[2 * p + side].append(float(d.mean()))
                    rms[2 * p + side].append(float(np.sqrt((d ** 2).mean())))
    finally:
        org_cap.release()
        imp_cap.release()

    if dump_dir:
        dump_frames(source, marked, dump_dir, frames)

    mark_rms = float(np.mean([np.mean(r) for r in rms]))
    flicker_rms = float(np.mean([np.std(r) for r in rms]))
    return {
        "patch_psnr_db": round(float(np.mean(patch_psnr)), 2),
        "patch_ssim": round(float(np.mean(patch_ssim)), 4),
        "frame_psnr_db": round(float(np.mean(full_psnr)), 2),
        # Blind to this mark by construction; kept as the control.
        "flicker_luma_std": round(float(np.mean([np.std(d) for d in diffs])), 3),
        "mark_rms": round(mark_rms, 3),
        "flicker_rms_std": round(flicker_rms, 3),
        "flicker_ratio": round(flicker_rms / mark_rms, 3) if mark_rms > 1e-9 else 0.0,
    }


def dump_frames(source, marked, dump_dir, frames, count=6):
    """Side-by-side crops around frame centre, to be looked at rather than scored."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    org_cap = cv2.VideoCapture(str(source))
    imp_cap = cv2.VideoCapture(str(marked))
    want = set(np.linspace(0, max(frames - 1, 1), count).astype(int).tolist())
    try:
        for i in range(frames):
            ok_a, fa = org_cap.read()
            ok_b, fb = imp_cap.read()
            if not (ok_a and ok_b):
                break
            if i not in want:
                continue
            h, w = fa.shape[:2]
            y0, y1 = max(0, h // 2 - 300), min(h, h // 2 + 300)
            x0, x1 = max(0, w // 2 - 500), min(w, w // 2 + 500)
            cv2.imwrite(str(dump_dir / f"f{i:05d}.png"),
                        np.hstack([fa[y0:y1, x0:x1], fb[y0:y1, x0:x1]]))
    finally:
        org_cap.release()
        imp_cap.release()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_sweep(args):
    """
    Walk the grid, flushing every cell to the findings files as it is measured.

    Nothing is held until the end: a killed run is restarted with --resume and picks up
    at the first cell that is not already in the JSONL. Marked videos and transcodes
    were already cached on disk, so a resume re-encodes nothing either.
    """
    rows = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (f"sweep {stamp} -- source={args.source} frames={args.frames} "
              f"crf={args.crf} payload=0x{PAYLOAD:08X} lightglue={args.lightglue}")
    done = completed_cells() if args.resume else set()
    opened = False

    for radius in args.radius:
        for tr in args.tr:
            for k in args.cluster_k:
                pending = [c for c in args.conditions
                           if sweep_key(c, radius, tr, k) not in done]
                if not pending:
                    print(f"skip r={radius} tr={tr} k={k}: all conditions done",
                          flush=True)
                    continue

                # Built only once a cell actually needs it, so a resume that has
                # nothing left to do for this configuration does not encode a video.
                marked = build_marked(args.source, radius, tr, k, args.frames,
                                      args.crf, PAYLOAD, rebuild=args.rebuild,
                                      verbose=not args.quiet)
                cfg = MFConfig(radius=radius, temp_redundancy=tr,
                               interleave=(tr == 1), cluster_k=k,
                               control_pairs=args.control_pairs,
                               band_width=args.band_width)
                for condition in pending:
                    row, _ = measure(args.source, marked, condition, cfg,
                                     args.frames, PAYLOAD,
                                     lightglue=args.lightglue,
                                     homography_every=args.homography_every,
                                     progress=args.progress)
                    rows.append(row)
                    print(format_table([row]).splitlines()[-1], flush=True)
                    if not args.no_log:
                        if not opened:
                            start_findings(header)
                            opened = True
                        append_row(row, stamp)
    return rows


def cmd_visibility(args):
    rows = []
    for radius in args.radius:
        for tr in args.tr:
            for k in args.cluster_k:
                marked = build_marked(args.source, radius, tr, k, args.frames,
                                      args.crf, PAYLOAD, verbose=not args.quiet)
                dump = (Path(args.dump) / f"r{radius}_tr{tr}_k{k}"
                        if args.dump else None)
                row = {"radius": radius, "tr": tr, "cluster_k": k,
                       "map": "interleaved" if tr == 1 else "contiguous"}
                row.update(visibility(args.source, marked, args.frames, k, dump))
                rows.append(row)
                print(row, flush=True)

    if not args.no_log:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        keys = ["radius", "tr", "map", "cluster_k", "patch_psnr_db", "patch_ssim",
                "frame_psnr_db", "mark_rms", "flicker_rms_std", "flicker_ratio",
                "flicker_luma_std"]
        head = "| " + " | ".join(keys) + " |"
        rule = "|" + "|".join("---" for _ in keys) + "|"
        body = ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
        with FINDINGS_MD.open("a") as fh:
            fh.write(f"\n\n## visibility {stamp} -- source={args.source} "
                     f"frames={args.frames} crf={args.crf}\n\n"
                     + "\n".join([head, rule] + body) + "\n")
        # The JSONL copy the merge step expects; without it the visibility rows lived
        # only in the markdown and could not be read back programmatically.
        with FINDINGS_JSONL.open("a") as fh:
            for r in rows:
                fh.write(json.dumps({"kind": "visibility", "stamp": stamp, **r}) + "\n")
    return rows


def cmd_fp(args):
    """
    False positives: the presence statistic on video that is not carrying this payload.

    Three cases, and they fail differently. Unmarked video is the easy one. A *different
    payload* is the one that matters: the mark is there, so any statistic that only
    measures "was something blurred" will fire, and only a phase-and-payload-aware score
    separates them. The wrong-phase case shows what the search is buying.
    """
    rows = []
    for radius in args.radius:
        for tr in args.tr:
            for k in args.cluster_k:
                cfg = MFConfig(radius=radius, temp_redundancy=tr,
                               interleave=(tr == 1), cluster_k=k,
                               control_pairs=args.control_pairs,
                               band_width=args.band_width)
                mine = build_marked(args.source, radius, tr, k, args.frames,
                                    args.crf, PAYLOAD, verbose=not args.quiet)
                other = build_marked(args.source, radius, tr, k, args.frames,
                                     args.crf, OTHER_PAYLOAD, verbose=not args.quiet)
                unmarked = transcode(args.source, args.crf)

                for name, path, payload in (("marked", mine, PAYLOAD),
                                            ("other-payload", other, OTHER_PAYLOAD),
                                            ("unmarked", unmarked, None)):
                    result = scan(iter_y(args.source, limit=args.frames),
                                  iter_y(path, limit=args.frames), cfg,
                                  progress=args.progress, total=args.frames)
                    got = decode(result, cfg=cfg, stat=args.stat)
                    row = {"case": name, "radius": radius, "tr": tr, "cluster_k": k,
                           "stat": args.stat,
                           "presence": round(got["presence"], 2),
                           "recovered": f"0x{got['watermark']:08X}",
                           "phase": got["phase"]}
                    if payload is not None:
                        row["bcr"] = bit_correct_rate(
                            payload, hard_bits(got["L"]), cfg.bit_length)
                    rows.append(row)
                    print(row, flush=True)

    if not args.no_log:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with FINDINGS_JSONL.open("a") as fh:
            for row in rows:
                fh.write(json.dumps({"kind": "fp", "stamp": stamp, **row}) + "\n")
    return rows


def int_list(text):
    return [int(v) for v in text.split(",") if v]


def str_list(text):
    return [v for v in text.split(",") if v]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--radius", type=int_list, default=[80, 110, 140, 200])
    parser.add_argument("--tr", type=int_list, default=[1, 3, 5, 10, 30],
                        help="frame->bit block size; 1 is the interleaved map")
    parser.add_argument("--cluster-k", type=int_list, default=[1])
    parser.add_argument("--control-pairs", type=int, default=8)
    parser.add_argument("--band-width", type=int, default=48)
    parser.add_argument("--conditions", type=str_list,
                        default=["clean", "crf23", "mild", "moderate", "severe",
                                 "blur_only", "moire_only"])
    parser.add_argument("--lightglue", action="store_true",
                        help="estimate a homography for the geometric conditions")
    parser.add_argument("--homography-every", type=int, default=30)
    parser.add_argument("--stat", default="a", choices=list(STATS))
    parser.add_argument("--dump", default=None, help="write sample frames here")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already present in the findings jsonl")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--tag", default=None,
                        help="suffix the findings files, for sharded runs")
    parser.add_argument("command", choices=["sweep", "visibility", "fp"])

    args = parser.parse_args(argv)
    use_tag(args.tag)
    return {"sweep": cmd_sweep, "visibility": cmd_visibility, "fp": cmd_fp}[
        args.command](args)


if __name__ == "__main__":
    main()
