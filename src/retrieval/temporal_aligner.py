"""
Production streaming temporal aligner.

Given a reference (original) video, preprocessed once, and a stream of leak
frames arriving one at a time, produces for every leak frame:

  estimated_original_frame  -- a single best-guess center
  candidate_frames          -- a ranked +/-radius neighborhood around it

The visual matcher is frozen and reused exactly as Stages 2/3 defined it:

  SSCD -> top-K -> DISK/LightGlue -> RANSAC -> combined score

This module adds nothing to that matcher. It only adds a scalar,
constant-velocity (position + velocity + uncertainty) streaming filter that
fuses each frame's already-computed candidate scores with a SOFT temporal
prior, plus a proximity-based ranking of the output candidate region. See
PRODUCTION_TEMPORAL_ALIGNER.md for the derivation and justification of every
constant below.

Two usage modes, both exercised in production:

  # legacy / convenience -- the aligner builds and owns its own VisualMatcher:
  aligner = TemporalAligner(reference_frames, k=50, radius=5)
  result = aligner.process(leak_frame_bgr)

  # decoupled production mode -- visual matching and temporal alignment are
  # separate components, wired together by the caller:
  matcher = VisualMatcher(reference_frames, k=50)
  aligner = TemporalAligner(radius=5)                      # no reference frames needed
  result = aligner.process(visual_candidates=matcher.match(leak_frame_bgr))

Both paths run through the exact same temporal-update-and-ranking code, so
every behavior (including the hardening below) applies identically either way.

Hard architectural boundary: this file never imports, calls, or inspects
anything related to watermark detection, and no method here ever receives a
ground-truth frame index. Ground truth may only be used by an evaluation
driver that sits outside this module.
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from retrieval.encoders import build_encoder
from retrieval.evaluate_stage2_scores import SCORE_METHODS, scores_from_raw
from retrieval.index import build_index
from retrieval.lightglue_verify import Stage2Verifier

# Fixed constants, stated once, never fit to any test set (see PRODUCTION_TEMPORAL_ALIGNER.md).
T_GATE = 3.0          # gating penalty scale, same convention as Stage 3b's local rule
Q_PROCESS = 1.0        # process noise (frame^2) added to sigma every step
R0_MEASUREMENT = 4.0   # measurement-noise numerator (frame^2) for a maximally confident visual pick
SIGMA_INIT = 100.0     # bootstrap / re-bootstrap uncertainty (frame^2), ~10-frame std-dev
MISMATCH_RESET_STREAK = 3  # consecutive persistent large-residual frames before re-bootstrapping

# Stage 4B hardening (see PRODUCTION_TEMPORAL_ALIGNER.md section 5 for the derivation): a
# single-frame mismatch this many multiples of the current gate radius is treated as an
# unmistakable discontinuity rather than ordinary noise, and triggers an immediate
# re-bootstrap instead of waiting for MISMATCH_RESET_STREAK consecutive frames. Derived
# from two already-measured data points -- Stage 4A's leak-106 case (mismatch/gate_radius
# ratio ~16.4, must NOT trigger this) and Stage 4B's discontinuity case (ratio ~125, MUST
# trigger this) -- not tuned by rerunning the discontinuity test until it passed.
LARGE_MISMATCH_MULTIPLIER = 50.0


class VisualMatcher:
    """
    The frozen visual-matching stage, extracted so it can be built and reused
    independently of any particular TemporalAligner instance: SSCD top-K
    retrieval against a reference index built once, then DISK/LightGlue/RANSAC
    verification against that shortlist, producing the frozen `combined` score.

    match() never sees temporal state and never ranks anything -- it returns
    the raw (candidate_original_frame, combined_score) pairs for a single leak
    frame, in SSCD-rank order.
    """

    def __init__(self, reference_frames: list[np.ndarray], encoder_name: str = "sscd",
                device: str = "cuda", backend: str = "torch", max_keypoints: int = 1024,
                min_matches: int = 10, ransac_thresh: float = 5.0, k: int = 50):
        self.device = device
        self.k = k
        self.min_matches = min_matches
        self.n_ref = len(reference_frames)

        t0 = time.time()
        self.encoder = build_encoder(encoder_name, device)
        ref_emb = self.encoder.encode(reference_frames)
        self.index = build_index(ref_emb, backend, device)
        self.t_index_build_s = time.time() - t0

        t0 = time.time()
        self.verifier = Stage2Verifier(reference_frames, device, max_keypoints=max_keypoints,
                                       min_matches=min_matches, ransac_thresh=ransac_thresh)
        self.t_disk_cache_build_s = time.time() - t0

    def match(self, leak_frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns (candidate_idx, combined_scores, timing_s) for one leak frame."""
        t0 = time.time()
        q_emb = self.encoder.encode([leak_frame_bgr])
        t_sscd = time.time() - t0

        t0 = time.time()
        _, idx_topk = self.index.search(q_emb, self.k)
        candidate_idx = idx_topk[0]
        t_search = time.time() - t0

        t0 = time.time()
        raw_list = self.verifier.verify_frame_raw(leak_frame_bgr, candidate_idx)
        method_scores, _ = scores_from_raw(raw_list, self.min_matches)
        combined = method_scores["combined"]
        t_lightglue = time.time() - t0

        return candidate_idx, combined, {"sscd": t_sscd, "search": t_search, "lightglue": t_lightglue}


class TemporalAligner:
    """
    Streaming leak-frame -> original-frame aligner. See module docstring for
    the two supported usage modes.
    """

    def __init__(self, reference_frames: list[np.ndarray] | None = None, encoder_name: str = "sscd",
                device: str = "cuda", backend: str = "torch", max_keypoints: int = 1024,
                min_matches: int = 10, ransac_thresh: float = 5.0,
                k: int = 50, radius: int = 5):
        self.device = device
        self.k = k
        self.radius = radius
        self.min_matches = min_matches

        if reference_frames is not None:
            # Legacy/convenience mode: build our own internal VisualMatcher,
            # exactly as every Stage 4A/4B script already expects.
            self.matcher: VisualMatcher | None = VisualMatcher(
                reference_frames, encoder_name, device, backend, max_keypoints,
                min_matches, ransac_thresh, k)
            self.n_ref = self.matcher.n_ref
            self.t_index_build_s = self.matcher.t_index_build_s
            self.t_disk_cache_build_s = self.matcher.t_disk_cache_build_s
            # Backward-compatible aliases: Stage 4A/4B's own evaluation scripts
            # (not modified by this stage) reach into aligner.encoder/index/
            # verifier directly (e.g. the pre-flight LightGlue-call-count
            # check), so these must keep resolving exactly as before.
            self.encoder = self.matcher.encoder
            self.index = self.matcher.index
            self.verifier = self.matcher.verifier
        else:
            # Decoupled production mode: no reference frames, no matcher --
            # the caller supplies visual_candidates to process() directly.
            self.matcher = None
            self.n_ref = None
            self.t_index_build_s = 0.0
            self.t_disk_cache_build_s = 0.0

        self.timing_totals = {"sscd_s": 0.0, "search_s": 0.0, "lightglue_s": 0.0, "temporal_s": 0.0}
        self.n_processed = 0
        self.reset()

    def reset(self):
        """Clear temporal state back to bootstrap. Reference index/DISK cache are untouched."""
        self.pos_hat: float | None = None
        self.vel_hat: float = 0.0
        self.sigma: float = SIGMA_INIT
        self._mismatch_streak: int = 0
        self._initialized: bool = False

    def _rank_region(self, center: int, vel_hat: float, candidate_scores: dict[int, float]) -> tuple[list[int], list[float]]:
        """
        Rank [center-radius, center+radius] (clipped) primarily by proximity to
        center; ties at the same |distance| broken by (a) the direction vel_hat
        implies, (b) real LightGlue-verified visual evidence where available.
        Never reruns LightGlue -- candidate_scores only contains positions that
        were already in this frame's own top-K.
        """
        n_ref = self.n_ref if self.n_ref is not None else (max(candidate_scores) + 1 if candidate_scores else center + 1)
        lo, hi = max(0, center - self.radius), min(n_ref - 1, center + self.radius)
        positions = list(range(lo, hi + 1))
        forward_first = vel_hat >= 0

        def sort_key(p: int):
            d = p - center
            ad = abs(d)
            dir_rank = 0 if (d >= 0) == forward_first else 1  # tie-break 1: direction consistent with vel_hat
            visual = candidate_scores.get(p, 0.0)
            return (ad, dir_rank, -visual)  # tie-break 2: any real visual evidence, higher first

        ranked = sorted(positions, key=sort_key)
        scores = [float(candidate_scores.get(p, 0.0)) for p in ranked]
        return ranked, scores

    def process(self, leak_frame_bgr: np.ndarray | None = None,
               visual_candidates: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
        """
        Process exactly one leak frame. No knowledge of its position in any
        sequence is used.

        Either pass `leak_frame_bgr` (legacy mode -- this instance's own
        internal VisualMatcher computes the candidates), or pass
        `visual_candidates=(candidate_idx, combined_scores)` directly
        (decoupled production mode -- no visual matching happens here at all).
        """
        if visual_candidates is not None:
            candidate_idx, combined = visual_candidates
            candidate_idx = np.asarray(candidate_idx)
            combined = np.asarray(combined)
            t_sscd = t_search = t_lightglue = 0.0  # measured by the caller's own matcher, not here
        else:
            if self.matcher is None:
                raise ValueError("process() needs either leak_frame_bgr (with a matcher built at "
                                "construction) or visual_candidates -- neither was provided")
            candidate_idx, combined, timing = self.matcher.match(leak_frame_bgr)
            t_sscd, t_search, t_lightglue = timing["sscd"], timing["search"], timing["lightglue"]

        t0 = time.time()
        best_raw_pos = int(np.argmax(combined))
        raw_visual_prediction = int(candidate_idx[best_raw_pos])
        raw_visual_score = float(combined[best_raw_pos])

        max_score = combined.max() if combined.max() > 0 else 1.0
        norm_visual = combined / max_score

        if not self._initialized:
            # Bootstrap: no prior trajectory exists yet, so the only honest thing
            # to do is trust the raw visual pick outright.
            self.pos_hat = float(raw_visual_prediction)
            self.vel_hat = 0.0
            self.sigma = SIGMA_INIT
            self._initialized = True
        else:
            # Gated-validation update: a standard tracking technique that avoids a
            # subtle but serious failure mode of a naive visual/temporal blend --
            # if the "accepted" candidate used to correct the filter is ITSELF
            # chosen by trading off against the filter's own current prediction,
            # the update can become self-confirming (predicted_pos stays put ->
            # the gate suppresses the true, moving candidate -> the resulting
            # "residual" is artificially ~0 -> the filter grows falsely confident
            # it was right -> the gate tightens further next frame), producing a
            # permanent lock with no residual signal to trigger recovery. Measured
            # directly during development: the estimate froze at frame 0 forever
            # while the frozen visual matcher's own raw predictions correctly
            # tracked upward. See PRODUCTION_TEMPORAL_ALIGNER.md.
            #
            # The fix: first test which candidates fall within a hard gate around
            # the PREDICTED position (independent of their visual score), then
            # pick the most visually-supported candidate only among those that
            # passed -- an honest, independent measurement, never traded off
            # against the very prediction it will be compared to.
            #
            # A second, related pitfall surfaced when K is a large fraction of
            # the reference set (e.g. K=50 of 120 frames): "some candidate falls
            # in the gate" is then satisfied almost by density alone, even when
            # the true content has moved well outside the gate -- so gating on
            # "is the gate empty" essentially never fires, and the filter can
            # drift onto a weak nearby distractor indefinitely. The mismatch
            # streak must instead track whether the frame's single BEST visual
            # candidate (raw_visual_prediction, independent of gating) agrees
            # with the trajectory -- that is a real, density-independent signal
            # of "the visual matcher currently believes something inconsistent
            # with our trajectory," which is what should accumulate toward a
            # re-bootstrap. See PRODUCTION_TEMPORAL_ALIGNER.md.
            predicted_pos = self.pos_hat + self.vel_hat
            sigma_pred = self.sigma + Q_PROCESS
            gate_radius = T_GATE * np.sqrt(sigma_pred)

            if abs(raw_visual_prediction - predicted_pos) > LARGE_MISMATCH_MULTIPLIER * gate_radius:
                # Stage 4B hardening: an unmistakable single-frame discontinuity
                # (e.g. a real cut/seek), not ordinary noise -- don't wait for
                # MISMATCH_RESET_STREAK consecutive frames, and don't let a
                # partial blend chase this one huge residual and explode
                # vel_hat in the meantime. Trust the new visual evidence
                # immediately. See PRODUCTION_TEMPORAL_ALIGNER.md.
                self.pos_hat = float(raw_visual_prediction)
                self.vel_hat = 0.0
                self.sigma = SIGMA_INIT
                self._mismatch_streak = 0
            else:
                raw_in_gate = abs(raw_visual_prediction - predicted_pos) <= gate_radius
                self._mismatch_streak = 0 if raw_in_gate else self._mismatch_streak + 1

                if self._mismatch_streak >= MISMATCH_RESET_STREAK:
                    # Persistent disagreement, not a one-off: re-bootstrap on the
                    # current frame's raw visual evidence so the tracker can follow
                    # a genuine trajectory change instead of staying locked waiting
                    # for the old trajectory to return.
                    self.pos_hat = float(raw_visual_prediction)
                    self.vel_hat = 0.0
                    self.sigma = SIGMA_INIT
                    self._mismatch_streak = 0
                else:
                    in_gate = np.abs(candidate_idx.astype(float) - predicted_pos) <= gate_radius
                    if in_gate.any():
                        in_gate_idx = np.where(in_gate)[0]
                        best_in_gate = in_gate_idx[np.argmax(norm_visual[in_gate_idx])]
                        measurement = float(candidate_idx[best_in_gate])
                        measurement_norm_visual = float(norm_visual[best_in_gate])
                    else:
                        # Nothing in gate at all this single frame (rare) -- fall
                        # back to the raw pick as the measurement rather than
                        # skipping the update outright.
                        measurement = float(raw_visual_prediction)
                        measurement_norm_visual = float(norm_visual[best_raw_pos])

                    residual = measurement - predicted_pos
                    R = R0_MEASUREMENT / max(measurement_norm_visual, 1e-3)
                    K_gain = sigma_pred / (sigma_pred + R)
                    self.pos_hat = predicted_pos + K_gain * residual
                    self.vel_hat = self.vel_hat + K_gain * residual
                    self.sigma = (1.0 - K_gain) * sigma_pred

        estimated_original_frame = int(round(self.pos_hat))
        confidence = 1.0 / (1.0 + self.sigma / (self.radius ** 2))

        candidate_scores = {int(candidate_idx[i]): float(combined[i]) for i in range(len(candidate_idx))}
        ranked_frames, ranked_scores = self._rank_region(estimated_original_frame, self.vel_hat, candidate_scores)
        t_temporal = time.time() - t0

        self.timing_totals["sscd_s"] += t_sscd
        self.timing_totals["search_s"] += t_search
        self.timing_totals["lightglue_s"] += t_lightglue
        self.timing_totals["temporal_s"] += t_temporal
        self.n_processed += 1

        return {
            "estimated_original_frame": estimated_original_frame,
            "candidate_frames": ranked_frames,
            "candidate_temporal_scores": ranked_scores,
            "confidence": confidence,
            "raw_visual_prediction": raw_visual_prediction,
            "raw_visual_score": raw_visual_score,
            "temporal_adjusted_prediction": estimated_original_frame,
            "_timing_s": {"sscd": t_sscd, "search": t_search, "lightglue": t_lightglue, "temporal": t_temporal},
        }
