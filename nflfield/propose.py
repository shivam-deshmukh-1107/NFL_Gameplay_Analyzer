"""Stage 4c - run both identity proposers on one line set and compare them.

THIS IS THE ONE GENUINELY NEW THING IN THE MERGE. Neither parent project had a
second opinion on line identity, and identity is precisely where both were
bitten (see `ladder.py`'s docstring for the two incidents).

WHY THE COMPARISON IS MEANINGFUL AND FREE
-----------------------------------------
The two proposers were written independently, in different parameterisations of
the same 1-D projective model, and they return the same quantity:

  * `lattice.fit(s)` gives an integer lattice index per sorted position, so
    `np.diff(index)` is the rung count between consecutive lines.
  * `ladder.assign_ladder(ys)` returns those rung counts directly.

So `LatticeFit.steps` and `assign_ladder(...)[0]` are directly comparable, as
long as both are handed the SAME positions - which is why clustering happens
once, here, before either runs.

WHAT AGREEMENT DOES AND DOES NOT PROVE
--------------------------------------
Agreement is evidence, not proof, and the limit is known exactly: **both
proposers are blind to the same scene-scale degeneracy.** A ladder and the same
ladder doubled have identically zero projective residual, and a comb score at
pitch `alpha` also scores well at `alpha/2`. Shivam guards it with "take the
coarsest pitch within tolerance" and Yash with a prior toward the previous
pose's spacing - two different heuristics over one unobservable, so agreeing
here says nothing about it. The scale degeneracy is caught downstream by
`MAX_JUMP_YD`, and nowhere else.

What agreement DOES bear on is the thing that actually goes wrong per frame: a
stray line (turf texture, a banner edge, a drawn marker that leaked) changing
the rung pattern. The two proposers weight that differently - the comb score
dilutes, the beam search pays a residual - so a stray that fools one often does
not fool the other.

HOW DISAGREEMENT SHOULD BE TREATED IS AN OPEN QUESTION
------------------------------------------------------
Deliberately unanswered in code. `compare()` only reports. Whether a
disagreement should refuse the frame, downgrade it to TRACKED, or merely be
logged is to be decided from the measured disagreement rate on all sources
(MERGE_PLAN.md Phase 3 step 2) - never from the assumption that two
disagreeing estimators mean one is wrong, which on this footage would be a
guess about which.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import lattice as LAT
from . import ladder as LAD
from .config import LONG_CUT_PX, SHIVAM_REF_W, det_scale

#: A split edge is a gap FAR smaller than the gaps on either side of it.
#:
#: THE PROBLEM, measured by Yash and reproduced here. The same painted line is
#: ~8 px wide at the top of the picture and 15-20 px at the bottom, so its two
#: LSD edges can both survive the clustering tolerance and arrive as two
#: "lines" ~10-13 px apart where the real 5-yard gap at that depth is 185 px.
#: `assign_ladder` then gives both the same yard label, or inflates the whole
#: ladder, and the frame is refused.
#:
#: REPRODUCED ON THIS PIPELINE 2026-10-17, on the clip_03 frames the camera
#: refused on residual, against the independent detector:
#:     f42   ours 447 and 457   indep 457     (10 px apart)
#:     f96   ours 412 and 423   indep 426     (11 px apart)
#:     f163  ours 496 and 507   indep 502     (11 px apart)
#: Our ladder residual on those frames ran 3-10x worse than the independent
#: detector's. I had assumed thinning made this impossible here, because LSD
#: sees one skeleton per stripe rather than two edges - **that reasoning was
#: wrong**: `cluster_1d`'s tolerance is ~10 px at this scale, so a pair 10-11 px
#: apart passes straight through it.
#:
#: The test is scale-free and uses NO POSE. Deliberately conservative at 0.30:
#: Yash measured that at 0.45 it merged a genuine line pair 22 px apart on
#: clip_01 f160 whose neighbouring gap was 53 px, destroying a real
#: observation. Raising it trades a real line for a duplicate.
SPLIT_EDGE_RATIO = 0.30

#: OFF BY DEFAULT. **The unconditioned port was measured NET NEGATIVE and
#: reverted**, per the standing rule: if a change makes a figure worse, revert
#: and rethink rather than stacking a second fix on the first.
#:
#: MEASURED end to end, all clips at stride 1, certified frames:
#:     clip_03  166 -> 173  (+7)   <- the intended target
#:     hail_mary 53 ->  66  (+13)
#:     clip_01  193 -> 183  (-10)
#:     clip_02  171 -> 124  (-47)  <- unacceptable
#:     clip_04  258 -> 258  (0), but its scale error went +0.22 -> +0.27,
#:                                over the 0.25 yd bar
#:
#: WHY IT FAILED HERE WHEN IT WORKS FOR YASH. His test fires when a gap is
#: `< ratio * min(neighbouring gaps)`, and `min(nb)` is fragile when a line is
#: MISSING: gaps of 100, 15, 100 fire even though the 15 is a real 5-yard gap,
#: because both neighbours are inflated by absent lines. clip_02 is the
#: congestion clip, so occluded lines are precisely its failure mode - it lost
#: 47 frames to genuine adjacent lines being fused. His version survives this
#: because his input is not pre-clustered; mine arrives already merged at
#: ~10 px by `cluster_1d`, so the sub-threshold gaps his test keys on are gone
#: and only ambiguous ones remain.
SPLIT_EDGE_ENABLED = False

#: Extra condition for the conditioned variant: the gap must also be small
#: against the MEDIAN gap, which one missing line cannot inflate the way
#: `min(neighbours)` can. Strictly more conservative than the ratio test alone.
SPLIT_EDGE_MEDIAN_RATIO = 0.30


def merge_split_edges(s, slopes, weights=None, ratio=SPLIT_EDGE_RATIO,
                      median_ratio=SPLIT_EDGE_MEDIAN_RATIO):
    """Fuse positions whose gap is far smaller than the real line spacing.

    `s` must be sorted. Returns `(s, slopes)`; merging is weighted so the
    fused position sits where the evidence is, not midway between two edges.

    TWO conditions, both required. The gap must be small against both
    neighbouring gaps (Yash's test) AND small against the median gap. The
    second exists because the first alone fused genuine adjacent lines
    wherever a line was missing - see `SPLIT_EDGE_ENABLED` for the 47 frames
    that cost on clip_02. A median is not inflated by one absent line.
    """
    import numpy as _np
    if len(s) < 3:
        return s, slopes
    w = _np.ones(len(s)) if weights is None else _np.asarray(weights, float)
    cur = [[float(a), float(b), float(c)] for a, b, c in zip(s, slopes, w)]
    while len(cur) >= 3:
        ys = [c[0] for c in cur]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        med = float(_np.median(gaps))
        hit = None
        for i, g in enumerate(gaps):
            nb = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps)]
            if not nb:
                continue
            if g < ratio * min(nb) and g < median_ratio * med:
                hit = i
                break
        if hit is None:
            break
        a, b = cur[hit], cur[hit + 1]
        wa, wb = a[2], b[2]
        cur = cur[:hit] + [[(a[0] * wa + b[0] * wb) / (wa + wb),
                            (a[1] * wa + b[1] * wb) / (wa + wb),
                            wa + wb]] + cur[hit + 2:]
    return (_np.array([c[0] for c in cur]), _np.array([c[1] for c in cur]))


@dataclass
class Proposal:
    """Both proposers' verdicts on one frame's line set."""

    s: np.ndarray                      # clustered transversal positions, sorted
    slopes: np.ndarray                 # image slope of each line at the column
    lat: LAT.LatticeFit | None = None
    lad: tuple | None = None           # (steps, cumulative, rms)
    n_long: int = 0
    n_dropped: int = 0                 # lines removed as ladder impostors
    agree: bool | None = None          # None when one proposer gave no answer
    steps_lat: np.ndarray | None = None
    steps_lad: np.ndarray | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.lat is not None or self.lad is not None

    def as_row(self) -> dict:
        """Flat record for a per-frame log."""
        return {
            "n_lines": int(len(self.s)),
            "n_long": int(self.n_long),
            "n_dropped": int(self.n_dropped),
            "lattice_ok": self.lat is not None,
            "ladder_ok": self.lad is not None,
            "ladder_rms": (round(float(self.lad[2]), 3)
                           if self.lad is not None else None),
            "lattice_score": (round(float(self.lat.score), 4)
                              if self.lat is not None else None),
            "agree": self.agree,
            "steps_lattice": (self.steps_lat.tolist()
                              if self.steps_lat is not None else None),
            "steps_ladder": (self.steps_lad.tolist()
                             if self.steps_lad is not None else None),
            "note": self.note,
        }


def line_positions(segs: np.ndarray, w: int, s_scale: float,
                   long_cut: float | None = None):
    """Clustered positions of a pencil along the centre column.

    Returns `(s_clustered, slopes, n_long)`.

    The transversal is the vertical line `x = w/2`, which both parents use:
    Shivam fits the lattice along it, and Yash's camera gates describe every
    line by its `(y, slope)` there. Using one column for both is what lets the
    proposers and the camera share a line set without conversion.

    MEASUREMENT TRAP, from Yash's lessons and worth repeating here because this
    function is where it would reappear: do NOT describe a line by the field
    point at the field's LATERAL CENTRE. That point moves when the camera's
    lateral position changes even though the line's image does not, so a metric
    built on it reported a p90 angle jump of 1.09 -> 13.53 deg on a change that
    cannot move a yard line by a floating-point ulp. Describe a line by where
    it actually crosses the frame.
    """
    if len(segs) == 0:
        return np.zeros(0), np.zeros(0), 0
    if long_cut is None:
        long_cut = LONG_CUT_PX * (w / 1356.0)
    L = G.seg_length(segs)
    long_segs = segs[L >= long_cut]
    n_long = len(long_segs)
    if n_long == 0:
        return np.zeros(0), np.zeros(0), 0

    ell = np.array([1.0, 0.0, -w / 2.0])
    pos, keep = LAT.transversal_positions(long_segs, ell)
    if not keep.any():
        return np.zeros(0), np.zeros(0), n_long
    pos = pos[keep]
    kept = long_segs[keep]

    order = np.argsort(pos)
    pos, kept = pos[order], kept[order]
    d = G.seg_direction(kept)
    slopes = d[:, 1] / np.where(np.abs(d[:, 0]) > 1e-9, d[:, 0], 1e-9)

    # Cluster near-duplicates ONCE, before either proposer runs, so the two are
    # scored on identical input. `tol` is one of Shivam's 960-width constants.
    sc = LAT.cluster_1d(pos, tol=7.0 * s_scale)
    # Carry a representative slope for each cluster: the nearest original.
    if len(sc) and len(pos):
        near = np.abs(sc[:, None] - pos[None, :]).argmin(axis=1)
        slopes = slopes[near]
    # Then fuse split edges, which clustering alone cannot catch - the gap is
    # larger than the clustering tolerance but far smaller than a yard spacing.
    # Weighted by segment length, so the fused line sits on the evidence.
    if SPLIT_EDGE_ENABLED and len(sc) >= 3:
        L = G.seg_length(kept)[near] if len(pos) else None
        sc, slopes = merge_split_edges(sc, slopes, weights=L)
    return sc, slopes, n_long


def compare(segs: np.ndarray, w: int, prior=None,
            s_scale: float | None = None) -> Proposal:
    """Run both proposers on one pencil and report whether they agree.

    `prior`, when given, is the downfield field-x the previous pose assigns to
    each clustered position, in the same order - it breaks ladder ties only.
    """
    if s_scale is None:
        s_scale = det_scale(w)
    s, slopes, n_long = line_positions(segs, w, s_scale)
    p = Proposal(s=s, slopes=slopes, n_long=n_long)
    if len(s) < 4:
        p.note = "only %d clustered line(s)" % len(s)
        return p

    # Proposer B first: its impostor rejection changes the line set, and the
    # comparison has to be on one set. Yash's rejection is the more
    # conservative of the two projects' and has the measured record behind it.
    lines = [(float(y), float(sl)) for y, sl in zip(s, slopes)]
    kept, n_drop = LAD.drop_impostors(lines)
    p.n_dropped = n_drop
    if n_drop:
        s = np.array([y for y, _ in kept], float)
        slopes = np.array([sl for _, sl in kept], float)
        p.s, p.slopes = s, slopes
        prior = None            # the prior was indexed to the old set

    if len(s) >= 4:
        p.lad = LAD.assign_ladder(s, prior=prior)
    if len(s) >= 5:
        f = LAT.fit(s)
        if f is not None:
            p.lat = LAT.refine(s, f)

    if p.lat is not None:
        p.steps_lat = p.lat.steps
    if p.lad is not None:
        p.steps_lad = np.asarray(p.lad[0], int)

    if p.steps_lat is not None and p.steps_lad is not None:
        if len(p.steps_lat) == len(p.steps_lad):
            p.agree = bool(np.array_equal(p.steps_lat, p.steps_lad))
        else:
            # Can happen when the lattice drops a position as an outlier; the
            # patterns are then not comparable element-wise.
            p.agree = False
            p.note = "step patterns differ in length (%d vs %d)" % (
                len(p.steps_lat), len(p.steps_lad))
    return p
