"""Stage 7b - the dense-flow identity referee: did the yard NAMES slip?

PROVENANCE
----------
Yash's `experiments/independent/flow_check.py`, restated here.

THE QUESTION NOTHING ELSE CAN ANSWER
------------------------------------
Every other instrument in either parent project compares a grid against
painted yard lines. **A grid shifted by exactly one painted line lands on
painted lines.** So a whole-step identity slip is invisible to:

    paint coverage      the grid is on paint, just the wrong paint
    density             still one grid line per painted line
    err_yd              measured to the nearest line, which it matches
    the ladder residual unchanged - the spacing is right, the names are not
    the camera residual  unchanged, for the same reason

This happened for real: Yash's clip_06 f751 slipped by one line and 69 frames
inherited a 5 yd offset with every one of those metrics healthy. Two
independent instruments found it, and this is one of them.

HOW IT WORKS
------------
Dense Farneback flow on the turf between two frames says where the pixels
actually went. The two frames' poses also predict where they went: lift a point
to the field through the first pose, drop it back through the second. If the
yard NAMES are consistent between the two frames, those agree. If the second
frame's names slipped by a line, the prediction is off by a whole line spacing
in the image - a jump the pixels did not make.

So the test is comparative, not absolute: the solved pair must explain the flow
BETTER than the same pair with one frame relabelled by +-5 yd. Validated by
Yash on clips 01, 04 and 05 - true pairs a median 6-16 px, injected slips
137-229 px. That separation is the whole reason to trust it.

**ONLY TRUSTED FOR GAPS OF <= 3 FRAMES.** Farneback over a longer gap on a
panning camera is not reliable, and a non-answer must be read as a non-answer:
Yash's `flow_bridge` could not resolve clip_01's 16-frame gap f132-148 and
reported that rather than guessing, which is the correct behaviour.

RELABELLING A POSE
------------------
Shifting the yard names by `d` means the pose that called a point X must now
call it X + d. The new pose therefore satisfies H'(X + d) = H(X), i.e.
H' = H @ T where T translates the downfield coordinate by -d. That is an exact
field-space operation on the homography - no refitting, so the comparison
isolates the NAMING and changes nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import YARDS_PER_CELL, px_scale

#: Largest frame gap the flow is trusted over.
MAX_GAP = 3
#: Sample grid density on the turf, and the minimum usable sample count.
SAMPLE_STEP = 24          # px at PX_REF_W
MIN_SAMPLES = 40
#: Downsample factor for the flow computation. Farneback at full 1356-wide
#: resolution is slow and no more accurate here; the displacements being
#: distinguished are a line spacing apart (tens of px), not sub-pixel.
FLOW_SCALE = 0.5
#: A pair is FLAGGED when a relabelled pose explains the flow better than the
#: solved one by at least this ratio. Not 1.0: the two hypotheses are never
#: exactly equal, and a bare "better" would flag noise. Yash's validated
#: separation is an order of magnitude, so a factor of 1.5 is deep inside it.
FLAG_RATIO = 1.5


@dataclass
class FlowVerdict:
    frame_a: int
    frame_b: int
    gap: int
    n_samples: int = 0
    err_solved: float | None = None      # px, median flow disagreement
    err_minus: float | None = None       # with frame B relabelled -5 yd
    err_plus: float | None = None        # with frame B relabelled +5 yd
    verdict: str = "no_answer"           # consistent | slip | no_answer
    best_shift: float = 0.0
    note: str = ""

    def as_row(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in vars(self).items()}


def relabel(H, d_yards: float) -> np.ndarray:
    """The same pose with its yard names shifted by `d_yards`.

    H' = H @ T, T translating the downfield coordinate by -d, so that
    H'(X + d, Y) == H(X, Y). Exact; nothing is refitted.
    """
    T = np.array([[1.0, 0.0, -float(d_yards)],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    return np.asarray(H, float) @ T


def _turf_mask(frame, turf) -> np.ndarray:
    return turf.turf_mask(frame, k=3.0)


def _predict(pts, H_a, H_b):
    """Where `pts` in frame A should land in frame B, given both poses."""
    try:
        inv = np.linalg.inv(np.asarray(H_a, float))
    except np.linalg.LinAlgError:
        return None
    P = np.column_stack([pts, np.ones(len(pts))]) @ inv.T
    w = P[:, 2]
    ok = np.abs(w) > 1e-9
    if not ok.any():
        return None
    F = np.empty((len(pts), 2))
    F[:] = np.nan
    F[ok] = P[ok, :2] / w[ok, None]
    Q = np.column_stack([F, np.ones(len(F))]) @ np.asarray(H_b, float).T
    w2 = Q[:, 2]
    ok2 = np.abs(w2) > 1e-9
    out = np.empty((len(pts), 2))
    out[:] = np.nan
    out[ok2] = Q[ok2, :2] / w2[ok2, None]
    return out


def check_pair(frame_a, frame_b, H_a, H_b, turf, frame_ia: int = -1,
               frame_ib: int = -1, hud=None) -> FlowVerdict:
    """Compare the solved pair against +-one-line relabellings of frame B."""
    gap = abs(int(frame_ib) - int(frame_ia))
    v = FlowVerdict(frame_a=int(frame_ia), frame_b=int(frame_ib), gap=gap)
    if gap > MAX_GAP:
        v.note = "gap %d > %d; flow not trusted" % (gap, MAX_GAP)
        return v
    if H_a is None or H_b is None:
        v.note = "a pose is missing"
        return v

    s = px_scale(frame_a.shape[1])
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    fs = FLOW_SCALE
    ga_s = cv2.resize(ga, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA)
    gb_s = cv2.resize(gb, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(ga_s, gb_s, None, 0.5, 3,
                                        max(5, int(21 * s * fs)), 3, 5, 1.2, 0)

    # Sample on the turf only, away from the HUD: the HUD does not move with
    # the camera, so flow there is zero and would drag every estimate.
    m = _turf_mask(frame_a, turf)
    if hud is not None:
        m = m & ~hud
    step = max(4, int(round(SAMPLE_STEP * s)))
    ys, xs = np.mgrid[0:frame_a.shape[0]:step, 0:frame_a.shape[1]:step]
    xs, ys = xs.ravel(), ys.ravel()
    keep = m[ys, xs]
    xs, ys = xs[keep], ys[keep]
    if len(xs) < MIN_SAMPLES:
        v.note = "only %d turf samples" % len(xs)
        return v
    pts = np.column_stack([xs, ys]).astype(float)

    fx = flow[(ys * fs).astype(int).clip(0, flow.shape[0] - 1),
              (xs * fs).astype(int).clip(0, flow.shape[1] - 1), 0] / fs
    fy = flow[(ys * fs).astype(int).clip(0, flow.shape[0] - 1),
              (xs * fs).astype(int).clip(0, flow.shape[1] - 1), 1] / fs
    observed = pts + np.column_stack([fx, fy])

    def err_for(Hb):
        pred = _predict(pts, H_a, Hb)
        if pred is None:
            return None
        d = np.linalg.norm(pred - observed, axis=1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if len(d) >= MIN_SAMPLES else None

    v.n_samples = int(len(pts))
    v.err_solved = err_for(H_b)
    v.err_minus = err_for(relabel(H_b, -YARDS_PER_CELL))
    v.err_plus = err_for(relabel(H_b, +YARDS_PER_CELL))
    if v.err_solved is None:
        v.note = "could not evaluate the solved pair"
        return v

    alts = [(v.err_minus, -YARDS_PER_CELL), (v.err_plus, +YARDS_PER_CELL)]
    alts = [(e, d) for e, d in alts if e is not None]
    if not alts:
        v.verdict = "consistent"
        v.note = "no alternative evaluable"
        return v
    best_e, best_d = min(alts, key=lambda t: t[0])
    if best_e * FLAG_RATIO < v.err_solved:
        v.verdict = "slip"
        v.best_shift = float(best_d)
        v.note = ("a %+.0f yd relabelling explains the flow %.1fx better"
                  % (best_d, v.err_solved / max(best_e, 1e-6)))
    else:
        v.verdict = "consistent"
        v.best_shift = 0.0
        v.note = ("solved %.1f px vs best alternative %.1f px"
                  % (v.err_solved, best_e))
    return v


def inject_slip(H, d_yards: float = YARDS_PER_CELL):
    """A deliberately slipped pose, for validating the referee itself.

    **Build the instrument a known answer before trusting it on data.** Yash
    lost a full round to a version of a different test that returned
    3.5-10.5 deg and looked like a flat refutation; it was feeding two
    vanishing POINTS to a form that measures the angle between two LINES, and
    only a synthetic check with a known answer caught it.
    """
    return relabel(H, d_yards)
