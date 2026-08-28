"""
Resumable attack sweep.

One video decode per cell, and every metric and decoding variant derived from the
evidence that decode produced. A sweep of decoding choices should cost one pass over
the video, not one pass per question asked of it.

Rows are appended to a JSONL as each cell finishes, so a killed run loses one cell
rather than the sweep, and --resume skips what is already there. The markdown table
is regenerated from the JSONL rather than appended to, so it can never disagree with
it.

Three cases are run per cell, and all three are needed:

    marked      the payload that was embedded must come back
    unmarked    the same clip with no mark must return NO_WATERMARK
    other       a clip carrying a DIFFERENT valid ID must return THAT ID

The third is the one a presence statistic passes and an identifier must not. The two
payloads are chosen with the same popcount so nothing can succeed on a bit-balance cue.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.spread_spectrum.detect.pipeline import Detector, DetectorConfig
from src.spread_spectrum.embed import embed
from src.spread_spectrum.eval.attacks import apply_attack, build_suite
from src.spread_spectrum.ids import CodewordSet
from utils.bit import get_bit_string, uint32_to_hex

OUT_DIR = Path("outputs/spread_spectrum")
FINDINGS = Path("src/spread_spectrum/findings.jsonl")
TABLE = Path("src/spread_spectrum/FINDINGS.md")

COLUMNS = [("attack", 22), ("case", 9), ("scale", 7), ("sites", 6), ("bcr", 5),
           ("exact", 6), ("id_ok", 6), ("accept", 7), ("s1", 8), ("s2", 9),
           ("minz", 6), ("f2id", 5), ("sec", 6)]


def same_popcount_pair(ids, seed=0):
    """
    Two valid IDs with identical popcount.

    Equal popcount so the false-positive case cannot be passed on a bit-balance cue --
    a detector that has learned nothing but "how many ones" would otherwise separate
    them and look like it works.
    """
    rng = np.random.default_rng(seed)
    pc = np.bitwise_count(ids.ids.astype(np.uint32))
    for target in (16, 15, 17, 14, 18):
        pool = ids.ids[pc == target]
        if pool.size >= 2:
            a, b = rng.choice(pool, 2, replace=False)
            return int(a), int(b)
    raise ValueError("no two IDs share a popcount")


def marked_path(source, payload, cfg, frames):
    stem = Path(source).stem
    return (OUT_DIR / f"{stem}_n{frames}_a{cfg.alpha_label}_s{cfg.square_size}"
                      f"_l{cfg.level}_{uint32_to_hex(payload)}.mp4")


class _Cfg:
    """Embedding side of a cell, kept apart from DetectorConfig deliberately."""
    def __init__(self, alpha=3.0, square_size=128, level=1, seed=8787,
                 min_separation=None):
        self.alpha, self.square_size, self.level = alpha, square_size, level
        self.seed, self.min_separation = seed, min_separation
        self.alpha_label = f"{alpha:g}".replace(".", "p")


def build_marked(source, payload, cfg, frames, rebuild=False, verbose=True):
    path = marked_path(source, payload, cfg, frames)
    if path.exists() and not rebuild:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  embedding {uint32_to_hex(payload)} -> {path.name}")
    embed(str(source), str(path), payload, seed=cfg.seed, alpha=cfg.alpha,
          square_size=cfg.square_size, level=cfg.level,
          min_separation=cfg.min_separation, max_frames=frames, progress=False)
    return path


def measure(detector, video, attack, expected, frames, stride, workdir):
    t0 = time.time()
    stream = apply_attack(attack, video, limit=frames, stride=stride, workdir=workdir)
    try:
        result, per_frame = detector.detect_frames(stream)
    except ValueError:                       # no evidence at all in any frame
        return {"frames_seen": 0, "sites": 0, "bcr": 0, "exact": False,
                "id_ok": False, "accept": False, "s1": 0.0, "s2": 0.0,
                "minz": 0.0, "f2id": None, "scale": None, "rotation": None,
                "geo_z": None, "sec": round(time.time() - t0, 1)}

    from src.spread_spectrum.detect.localise import GeometrySearch
    gz, gbest, gid = GeometrySearch.lock_quality(detector.geometry_table or [])
    row = {
        "scale": round(detector.geometry[0], 4) if detector.geometry else 1.0,
        "rotation": round(detector.geometry[1], 3) if detector.geometry else 0.0,
        "geo_z": round(gz, 1),
        "frames_seen": len(per_frame),
        "sites": result.n_obs,
        "m_eff": round(result.m_eff, 1),
        "recovered": result.hex,
        "accept": bool(result.accepted),
        "reason": result.reason,
        "s1": round(result.s1, 2),
        "s1_second": round(result.s1_second, 2),
        "margin": round(result.margin, 2),
        "s2": round(result.s2, 1),
        "minz": round(result.min_abs_z, 2),
        "total_weight": float(result.total_weight),
        "sec": round(time.time() - t0, 1),
    }
    if expected is None:
        row.update({"bcr": None, "exact": None, "id_ok": None, "f2id": None})
    else:
        bits, _ = get_bit_string(expected)
        row["bcr"] = int(sum(a == b for a, b in zip(result.bits, bits)))
        row["exact"] = bool(result.bits == bits)
        row["id_ok"] = bool(result.accepted and result.watermark_id == expected)
        row["f2id"] = next((n for n, r in detector.decode_curve(per_frame)
                            if r.accepted and r.watermark_id == expected), None)
    return row


def completed(path):
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                                  # a half-written final line
        done.add((r.get("source"), r.get("attack"), r.get("case"), r.get("tag")))
    return done


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def render_table(jsonl, out):
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    head = "| " + " | ".join(n for n, _ in COLUMNS) + " |"
    rule = "|" + "|".join("-" * (w + 2) for _, w in COLUMNS) + "|"
    body = []
    for r in rows:
        cells = []
        for name, _ in COLUMNS:
            v = r.get(name)
            cells.append("" if v is None else
                         ("yes" if v is True else "no" if v is False else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    out.write_text("# Spread-spectrum detector: measured results\n\n"
                   "Regenerated from `findings.jsonl`; edit that, not this.\n\n"
                   + "\n".join([head, rule] + body) + "\n")


def cmd_sweep(args):
    ids = CodewordSet.recommended(args.id_count)
    payload, other = same_popcount_pair(ids)
    suite = build_suite()
    names = ([n.strip() for n in args.attacks.split(",")] if args.attacks
             else list(suite))
    unknown = [n for n in names if n not in suite]
    if unknown:
        raise SystemExit(f"unknown attacks: {unknown}")

    ecfg = _Cfg(alpha=args.alpha, square_size=args.square_size, level=args.level,
                seed=args.seed, min_separation=args.min_separation)
    dcfg = DetectorConfig(seed=args.seed, square_size=args.square_size,
                          level=args.level, whiten=args.whiten,
                          weighting=args.weighting, max_sites=args.max_sites,
                          fpr=args.fpr, geometry=args.geometry, device=args.device)
    detector = Detector(ids, dcfg)
    print(detector)
    print(f"payload {uint32_to_hex(payload)}   decoy {uint32_to_hex(other)} "
          f"(popcount {int(np.bitwise_count(np.uint32(payload)))} both)")

    done = completed(FINDINGS) if args.resume else set()
    workdir = Path(args.workdir)

    for source in args.sources.split(","):
        source = source.strip()
        marked = build_marked(source, payload, ecfg, args.frames, args.rebuild)
        decoy = (build_marked(source, other, ecfg, args.frames, args.rebuild)
                 if "other" in args.cases else None)
        for name in names:
            for case in args.cases.split(","):
                key = (source, name, case, args.tag)
                if key in done:
                    continue
                video = {"marked": marked, "unmarked": source, "other": decoy}[case]
                expect = {"marked": payload, "unmarked": None, "other": other}[case]
                # Geometry is a per-video constant, so it is re-locked per cell.
                # The table has to be cleared too, or a cell that never runs the
                # search reports the previous cell's hypotheses as its own.
                detector.geometry = None
                detector.geometry_table = None
                row = {"source": source, "attack": name, "case": case,
                       "tag": args.tag, "frames": args.frames,
                       "payload": uint32_to_hex(payload),
                       "alpha": ecfg.alpha, "square_size": ecfg.square_size,
                       "level": ecfg.level, "whiten": dcfg.whiten,
                       "weighting": dcfg.weighting, "geometry": dcfg.geometry,
                       "stamp": args.stamp}
                row.update(measure(detector, str(video), suite[name], expect,
                                   args.frames, args.stride, workdir))
                append_row(FINDINGS, row)
                print("| " + " | ".join(
                    f"{row.get(n) if row.get(n) is not None else '':>{w}}"[:w + 4]
                    for n, w in COLUMNS) + " |")
    render_table(FINDINGS, TABLE)
    print(f"\nwrote {TABLE}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--sources", default="inputs/30.mp4")
    ap.add_argument("--attacks", default=None, help="comma list; default is all")
    ap.add_argument("--cases", default="marked,unmarked,other")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--id-count", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--square-size", type=int, default=128)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--min-separation", type=int, default=None)
    ap.add_argument("--seed", type=int, default=8787)
    ap.add_argument("--whiten", default="lm_lvn")
    ap.add_argument("--weighting", default="wiener")
    ap.add_argument("--max-sites", type=int, default=48)
    ap.add_argument("--fpr", type=float, default=1e-6)
    ap.add_argument("--geometry", default="none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workdir", default="/tmp/ss_sweep")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--stamp", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    cmd_sweep(ap.parse_args(argv))


if __name__ == "__main__":
    main()
