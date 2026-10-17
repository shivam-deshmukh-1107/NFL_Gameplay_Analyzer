"""Stage 4b - proposer B: the 5-yard ladder by beam search (Yash's).

PROVENANCE
----------
Ported from Yash's `camera_model.assign_ladder`, `_proj1d_residual` and
`drop_ladder_impostors`, with his measurements intact.

WHAT IT DOES
------------
Given the image positions of detected lines along the centre column, decide the
INTEGER number of 5-yard rungs between each consecutive pair, from the paint
alone. Detection misses lines but never invents one between two real ones, so
the steps are positive integers.

Scoring is the residual of a 1-D projective fit `y = (a u + b)/(c u + 1)`,
which for a fixed lateral offset is the EXACT model of a homography restricted
to the downfield axis, not an approximation. That is the same mathematics as
proposer A's `u = 1/(s - v)` linearity, in a different parameterisation.

WHY TWO PROPOSERS AT ALL
------------------------
Neither parent project had a second opinion on line identity, and identity is
where both of them were bitten:

  * Shivam measured that relabelling lines from the fitted lattice lets the fit
    move the labels to suit itself - clip_04's scrimmage-to-first-down drifted
    from 29 yd to 75.4 against a HUD still reading 29.
  * Yash measured a 5-yard identity slip on clip_06 f751 that 69 frames then
    inherited, invisible to density, err_yd and paint coverage alike, because a
    grid shifted by exactly one painted line still lands on paint.

Two independently written proposers scoring the same line set is a free
referee. See `propose.py`; whether disagreement should refuse a frame or only
downgrade it is a question to settle by MEASUREMENT, not by assumption.

THE DEGENERACY THIS CANNOT SEE
------------------------------
**The 1-D projective residual cannot separate a ladder from the same ladder
doubled - their residuals are identically 0.0.** That is the exact scene-scale
degeneracy that has already bitten this footage for real: scaling the camera
position by k and relabelling the yards by k gives pixel-identical yard lines
(max |dy| = 3e-12 px at k = 0.5/2/4), reached at 0.000 px residual, and the
solver will follow it happily to a camera 148 yd in the air.

So this module CANNOT be the thing that protects against it. Two other things
do: the `prior` argument below, which breaks the tie toward the previous pose's
spacing, and `MAX_JUMP_YD` in the camera gates. Never relax either.
"""
from __future__ import annotations

import numpy as np

# Above this 1-D projective rms (px at PX_REF_W), a detected line set does not
# describe one family of equally spaced coplanar lines and something in it is
# not paint.
LADDER_RMS_PX = 1.5

#: How much a candidate ladder is penalised, per yard, for disagreeing with the
#: spacing the PREVIOUS frame's camera implies for the same detected lines.
#:
#: WHY IT IS NEEDED. The projective residual cannot separate a ladder from the
#: same ladder doubled (see the module docstring), so `assign_ladder` was
#: separating them on the `0.05 * (sum(st) - i)` tie-break alone - a term with
#: no units and no evidence behind it. MEASURED consequence: once the near-field
#: paint was unmasked, clip_03 f150-f183 came back with an extra line and the
#: ladder flipped from all-10-yard steps to (10,10,10,5), moving the model
#: 5.3 yd from the previous frame at an unchanged 0.48 px residual. Eleven
#: frames were refused for it.
#:
#: WHY IT IS LEGITIMATE. `assign_yards` already takes the ORIGIN and DIRECTION
#: from the previous pose; this is the same prior applied to the SPACING, which
#: is the other half of the identity question. It is identity, not geometry: the
#: six camera parameters are still solved from this frame's paint alone.
#:
#: WHY IT IS NOT A LICENCE TO DRIFT. The prior only breaks ties between ladders
#: the paint cannot distinguish, and `MAX_JUMP_YD` still refuses the frame if
#: the fitted camera then disagrees with the previous one.
LADDER_PRIOR_W = 0.15

#: THIRD-STAGE impostor rejection, for goal-line views where the near-horizontal
#: family stops being the yard lines. Measured on Yash's clip_05: of 131 frames
#: whose ladder fails, 97 DO contain a clean 5-yard family, but recovering it
#: needs a median of 5 removals - outside the 2 + 1 budget of the first two
#: stages, so 61 frames were lost to the budget rather than to the evidence.
#:
#: WHY THIS IS DANGEROUS AND HOW IT IS CONSTRAINED. A subset chosen to minimise
#: ladder rms will always find something: with four lines left the projective
#: fit has three parameters and four points, so it is nearly free and "the
#: ladder got clean" is satisfied by construction. Three constraints:
#:   1. It never runs unless stages 1 and 2 have already failed.
#:   2. It is ALL-OR-NOTHING - either it reaches a ladder that is clean AND
#:      still has LADDER_MIN_SURVIVORS lines, or it returns the input
#:      untouched. It never leaves a frame half-stripped.
#:   3. LADDER_MIN_SURVIVORS is 6, not 4, so the survivor stays clearly
#:      over-determined rather than sitting at the free-fit floor.
#:
#: HONEST SIZE OF THE WIN: the subset search that motivated this found a clean
#: family in 97 of 131 failing frames; this recovers 9 of them. Do not quote the
#: 97 as the achievable number.
LADDER_EXTRA_DROPS = 6
LADDER_MIN_SURVIVORS = 6


def proj1d_residual(u: np.ndarray, y: np.ndarray) -> float:
    """Fit `y = (a u + b)/(c u + 1)` and return rms in px.

    For a fixed lateral offset a homography restricted to the downfield axis IS
    a 1-D projective map, so this is the exact model, not an approximation.
    """
    # np.column_stack rather than np.c_: identical result, no index-tricks
    # overhead, and this is hot (Yash measured 226k calls per clip).
    A = np.column_stack([u, np.ones_like(u), -y * u])
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    a, b, c = sol
    den = c * u + 1.0
    if np.any(np.abs(den) < 1e-9):
        return np.inf
    return float(np.sqrt(np.mean((y - (a * u + b) / den) ** 2)))


def assign_ladder(ys, max_step: int = 3, beam: int = 24, prior=None):
    """Integer 5-yard steps between consecutive detected lines, from the paint.

    Returns `(steps, cumulative, rms)` or None. Beam search; Yash verified it
    reproduces the exhaustive enumeration on all four anchor frames.

    `prior`, when given, is the downfield field-x the PREVIOUS pose assigns to
    each of the same lines, in the same order as sorted `ys`. It only breaks
    ties - see `LADDER_PRIOR_W`.
    """
    ys = np.sort(np.asarray(ys, float))
    n = len(ys)
    if n < 4:
        return None
    dw = None
    if prior is not None and len(prior) == n:
        pr = np.asarray(prior, float)
        if np.all(np.isfinite(pr)):
            dw = np.abs(pr - pr[0])
    states = [((), 0.0)]
    for i in range(1, n):
        nxt = []
        for steps, _ in states:
            for st_i in range(1, max_step + 1):
                st = steps + (st_i,)
                u = np.concatenate([[0], np.cumsum(st)]).astype(float)
                r = proj1d_residual(u, ys[:i + 1]) if i >= 3 else 0.0
                pen = 0.0
                if dw is not None:
                    pen = LADDER_PRIOR_W * float(
                        np.mean(np.abs(5.0 * u - dw[:i + 1])))
                nxt.append((st, r + 0.05 * (sum(st) - i) + pen))
        nxt.sort(key=lambda t: t[1])
        states = nxt[:beam]
    steps = states[0][0]
    u = np.concatenate([[0], np.cumsum(steps)]).astype(float)
    return steps, u, proj1d_residual(u, ys)


def drop_impostors(lines, max_drop: int = 2, rms_px: float = LADDER_RMS_PX):
    """Remove detected lines that cannot belong to one 5-yard ladder.

    `lines` is a list of `(y, slope)` at the centre column. Returns
    `(lines, n_dropped)`.

    ADDED by Yash 2026-09-08 after measurement: `assign_ladder` had no impostor
    rejection, and ONE stray line is enough to inflate the entire ladder. On
    clip_04 f161 the detector returned 11 lines including the yellow first-down
    marker at y=434, 18 px from the real line at y=416 where every genuine
    5-yard gap at that depth is ~100 px. The ladder came back
    (2,2,3,2,3,3,3,3,1,3) instead of all ones, the yard labels spanned roughly
    three times too much field, and - because a pinhole camera viewing a plane
    has an EXACT scale degeneracy - the solve followed it to a camera 148 yd in
    the air at zero cost in residual. Only `MAX_JUMP_YD` caught it, and the
    frame was then certified carrying a stale pose.

    Deliberately conservative. Stage 1 does nothing unless the ladder is
    already physically implausible (some pair of lines three or more rungs
    apart), and it only removes a line if doing so makes the ladder plausible
    AND lowers the residual. Leave-one-out on f161 picks the impostor
    unambiguously: dropping y=434 gives all-ones at rms 1.30, where the next
    best single drop is rms 2.24 and still has a 3-rung step.
    """
    lines = list(lines or [])
    if len(lines) < 5:
        return lines, 0
    dropped = 0

    # STAGE 1: a physically impossible rung (3+ steps between neighbours).
    for _ in range(max_drop):
        ys = np.sort(np.array([y for y, _ in lines], float))
        lad = assign_ladder(ys)
        if lad is None or max(lad[0]) <= 2:
            break
        base_max, base_rms = max(lad[0]), lad[2]
        best = None
        for i in range(len(lines)):
            sub = np.delete(ys, i)
            if len(sub) < 4:
                continue
            l2 = assign_ladder(sub)
            if l2 is None:
                continue
            key = (max(l2[0]), l2[2])
            if best is None or key < best[0]:
                best = (key, i)
        if (best is None or best[0][0] > 2 or best[0][0] >= base_max
                or best[0][1] >= base_rms):
            break
        y_drop = ys[best[1]]
        lines = [(y, sl) for y, sl in lines if y != y_drop]
        dropped += 1

    # STAGE 2: a ladder can be physically shaped (no step of 3 or more) and
    # still be nonsense, because one detected line is not on the 5-yard family
    # at all. Unmasking the near field made this common: clip_03 f150-f183 pick
    # up a turf-texture edge at y~594 whose gap to the line above it (97 px) is
    # SMALLER than the gap above that (161 px), while the previous pose puts it
    # 2.2 yd from its neighbour rather than a multiple of 5. The ladder comes
    # back (2,2,2,1) at 3.4 px rms, the model moves 5.3 yd from the previous
    # frame, and MAX_JUMP_YD refuses eleven frames.
    #
    # Deliberately narrow: it fires only when the whole set fits the projective
    # model BADLY, and only keeps a removal that makes it fit well. Note the fit
    # is nearly free once four lines remain (three parameters, four points), so
    # the evidence is the FIRST number, not the second.
    ys = np.sort(np.array([y for y, _ in lines], float))
    if len(lines) >= 5:
        lad = assign_ladder(ys)
        if lad is not None and lad[2] > rms_px:
            best = None
            for i in range(len(lines)):
                sub = np.delete(ys, i)
                l2 = assign_ladder(sub)
                if l2 is None or max(l2[0]) > 2:
                    continue
                if l2[2] <= rms_px and l2[2] < lad[2] / 3.0:
                    if best is None or l2[2] < best[0]:
                        best = (l2[2], i)
            if best is not None:
                y_drop = ys[best[1]]
                lines = [(y, sl) for y, sl in lines if y != y_drop]
                dropped += 1

    # STAGE 3 - see LADDER_EXTRA_DROPS. Only if the ladder is STILL bad, and
    # all-or-nothing.
    ys = np.sort(np.array([y for y, _ in lines], float))
    lad = assign_ladder(ys) if len(lines) >= 4 else None
    if (lad is not None and (lad[2] > rms_px or max(lad[0]) > 2)
            and len(lines) > LADDER_MIN_SURVIVORS):
        cur = list(ys)
        for _ in range(LADDER_EXTRA_DROPS):
            if len(cur) <= LADDER_MIN_SURVIVORS:
                break
            best = None
            for i in range(len(cur)):
                sub = np.sort(np.array(
                    [v for j, v in enumerate(cur) if j != i], float))
                l2 = assign_ladder(sub)
                if l2 is None or max(l2[0]) > 2:
                    continue
                if best is None or l2[2] < best[0]:
                    best = (l2[2], i)
            if best is None:
                break
            cur.pop(best[1])
            l3 = assign_ladder(np.sort(np.array(cur, float)))
            if (l3 is not None and l3[2] <= rms_px and max(l3[0]) <= 2
                    and len(cur) >= LADDER_MIN_SURVIVORS):
                keep = set(cur)
                out = [(y, sl) for y, sl in lines if y in keep]
                return out, dropped + (len(lines) - len(out))
        # never reached a clean, over-determined ladder: leave the frame alone

    return lines, dropped
