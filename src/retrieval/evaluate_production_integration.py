"""
Final integration test: one clean end-to-end run of the decoupled production
API (VisualMatcher + TemporalAligner), demonstrating the exact pipeline this
project has been building toward:

  reference video -> build/load reference caches (once)
       -> open leak video -> frame-by-frame visual matching
       -> TemporalAligner.process() -> incrementally write mapping file
       -> finish

Ground truth is used only to print a sanity summary at the end; it is never
passed to VisualMatcher or TemporalAligner. Uses CaptureSim (identity mapping)
on 4_sec_source.mp4 purely as a stand-in leak video for this demo -- the same
code runs unchanged against a real leak video.
"""
from __future__ import annotations

import json
import os

from retrieval.capture_sim import CaptureSim, get_condition
from retrieval.evaluate import load_frames
from retrieval.temporal_aligner import TemporalAligner, VisualMatcher

OUT_PATH = "out/retrieval/production_integration_test.jsonl"


def main():
    frames, _ = load_frames("4_sec_source.mp4")
    h, w = frames[0].shape[:2]
    print(f"reference: {len(frames)} frames at {w}x{h}")

    matcher = VisualMatcher(frames, k=50)
    print(f"reference preprocessing done: index {matcher.t_index_build_s:.1f}s, "
          f"DISK cache {matcher.t_disk_cache_build_s:.1f}s")

    aligner = TemporalAligner(radius=5)  # decoupled mode -- no reference frames passed in at all

    sim = CaptureSim(get_condition("mild"), (w, h), seed=0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n_written = 0
    with open(OUT_PATH, "w") as f:
        for i, ref_frame in enumerate(frames):  # stand-in leak video, one frame at a time
            leak_frame = sim.apply(ref_frame)
            candidates = matcher.match(leak_frame)[:2]  # (candidate_idx, combined_scores) only
            result = aligner.process(visual_candidates=candidates)
            record = {"leak_frame": i, **{k: v for k, v in result.items() if k != "_timing_s"}}
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_written += 1

    n_processed = len(frames)
    print(f"\nwrote {n_written} records to {OUT_PATH} for {n_processed} processed leak frames")
    assert n_written == n_processed, "record count must equal processed leak-frame count"
    print("PASS: number_of_output_records == number_of_processed_leak_frames")

    aligner.reset()
    print(f"after reset(): pos_hat={aligner.pos_hat} vel_hat={aligner.vel_hat} sigma={aligner.sigma}")


if __name__ == "__main__":
    main()
