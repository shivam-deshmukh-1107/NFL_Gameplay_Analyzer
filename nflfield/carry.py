"""Stage 7c - carry the camera through refused frames by aligning the turf.

PROVENANCE
----------
Ported from Yash's `carry_pass.py`, whose limits come from his
`experiments/zoom_carry/` measurements. **Read those before changing a stop
rule.**

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
A DISPLAY-ONLY state. Nothing here certifies a frame. It never touches `tier`
for a certified frame, never writes `params`, and never advances the reference
pose the next certified frame is measured against. It writes its own fields and
is drawn in its own colour.

For each run of uncertified on-field frames next to a certified frame, the
certified frame is the KEYFRAME. Each carried frame is aligned (ECC, homography
motion, on a turf+paint mask with the HUD and players removed) **directly
against the keyframe**, not frame to frame, so error does not accumulate step
by step. Runs are carried forward from the certified frame before them and
backward from the one after; each frame keeps whichever keyframe is nearer.

WHY THIS SUCCEEDS WHERE THE WALK FAILED
---------------------------------------
This is the mechanism for the exact failure stage 6 measured on clip_01
f133-148. There, a refused frame freezes the reference - deliberately, because
advancing it produced false accepts - so by f142 the jump gate was comparing
against a pose ten frames of real camera motion old, and every frame in the run
failed on `moves model` 3-8 yd. **The carry does not go through the solve at
all**, so it does not inherit that stale reference: it aligns pixels to the
nearest certified keyframe directly.

THE STOP RULES, AND WHY EACH ONE EXISTS
---------------------------------------
A carry stops at the first frame where ANY of:

  * more than `CARRY_MAX_FRAMES` from the keyframe;
  * the painted yard lines the detector found are, by median, more than
    `CARRY_LINE_MAX_YD` from the nearest carried yard line - **the paint that
    IS visible must agree**;
  * the alignment correlation falls below `CARRY_MIN_CC` (occlusion, a cut, a
    cross-fade - where the carry was measured to slip);
  * the frame is not field.

**The carry was measured to fail silently now and then** - Yash's clip_03 f70
landed 2 yd off at a healthy correlation - which is exactly why it is drawn as
CARRIED, in its own colour, and never certified.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import camera as CAM
from .config import YARDS_PER_CELL, px_scale

#: Alignment resolution. Half size is enough: the carry is correcting tens of
#: pixels, not sub-pixel, and ECC at full resolution costs several times more.
SCALE = 0.5

CARRY_MAX_FRAMES = 30
"""NOT raised, and the measurement says why.

Longer carries measured one-sided (every hidden frame carried forward only, as
in clip_01's tail; a validator that carries from both ends never tests beyond
half a gap): clip_04 max 0.10 yd at 46-60 frames; clip_01 max 0.31 yd at 46-60
with error growing with length (median 0.07 -> 0.12 -> 0.20 yd over 1-30 /
31-45 / 46-60); clips 02/03 stop on correlation before 30 anyway. And clip_05
carried across its f42 cross-fade lands exactly one line (5.1-5.3 yd) off the
certified solve on every later frame, with the paint check reading 0.03-0.27 -
at ANY length, 17 frames inside 30 already. "Every hidden frame <= 0.5 yd"
fails, so the limit stays.
"""

CARRY_LINE_MAX_YD = 0.5
CARRY_MIN_CC = 0.85
KEY_MIN_CC = 0.8

#: ECC iteration budget and termination epsilon.
ECC_ITERS = 60
ECC_EPS = 1e-5


@dataclass
class CarryRow:
    frame: int
    carried_from: int | None = None
    carried_frames: int = 0
    cc: float | None = None
    carry_check_yd: float | None = None
    homography: list | None = None
    stopped: str | None = None

    @property
    def ok(self) -> bool:
        return self.homography is not None


def carry_mask(frame: np.ndarray, turf, hud=None) -> np.ndarray:
    """Turf and paint, with the HUD removed.

    Both turf AND paint: aligning on turf texture alone is weakly conditioned
    along a painted line, and the paint is the strongest structure on the
    field. Players are left in - ECC's correlation is robust to a modest
    fraction of moving content, and masking them out was measured to remove so
    much of the near field that the alignment lost its vertical constraint.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    t = turf.turf_mask(frame, k=3.0)
    white = cv2.inRange(hsv, np.array([0, 0, 150]),
                        np.array([179, 70, 255])) > 0
    m = t | white
    if hud is not None:
        m = m & ~hud
    return m.astype(np.uint8) * 255


def align(key_frame, cur_frame, turf, hud=None, warp_init=None):
    """ECC homography from `key_frame` to `cur_frame`. Returns `(H3, cc)`.

    The warp is initialised from the frame-to-frame motion chain when one is
    given, because ECC is a local optimiser and a cold identity start will not
    converge across a fast pan.
    """
    mk = carry_mask(key_frame, turf, hud)
    mc = carry_mask(cur_frame, turf, hud)
    gk = cv2.cvtColor(key_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gc = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gk = cv2.resize(gk * (mk > 0), None, fx=SCALE, fy=SCALE,
                    interpolation=cv2.INTER_AREA)
    gc = cv2.resize(gc * (mc > 0), None, fx=SCALE, fy=SCALE,
                    interpolation=cv2.INTER_AREA)
    gk = (gk - gk.mean()) / max(gk.std(), 1e-6)
    gc = (gc - gc.mean()) / max(gc.std(), 1e-6)

    W = np.eye(3, dtype=np.float32)
    if warp_init is not None:
        W = np.asarray(warp_init, np.float32).copy()
        # The init is in full-resolution pixels; ECC works at SCALE.
        S = np.diag([SCALE, SCALE, 1.0]).astype(np.float32)
        W = (S @ W @ np.linalg.inv(S)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    try:
        cc, W = cv2.findTransformECC(gk, gc, W, cv2.MOTION_HOMOGRAPHY, crit,
                                     None, 5)
    except cv2.error:
        return None, None
    S = np.diag([SCALE, SCALE, 1.0])
    H3 = np.linalg.inv(S) @ np.asarray(W, float) @ S
    if not np.all(np.isfinite(H3)) or abs(np.linalg.det(H3)) < 1e-12:
        return None, None
    return H3, float(cc)


def paint_disagreement_yd(H_carried, lines, W, Hgt) -> float | None:
    """Median distance, in yards, from each detected line to the nearest
    carried yard line.

    **This is the stop rule that matters**, because it is the only one that
    consults the image's own evidence rather than the alignment's self-report.
    A healthy correlation with a wrong pose is precisely the silent failure the
    carry was measured to have.
    """
    if H_carried is None or not lines:
        return None
    grid = CAM.grid_lines_image(H_carried, W, Hgt)
    if not grid:
        return None
    gy = np.array([y for _, y, _ in grid], float)
    # Local pixels-per-yard from the grid's own spacing, so the comparison is
    # in yards and not in pixels - a pixel error near the horizon is a huge
    # yard error and vice versa.
    if len(gy) >= 2:
        spacing = float(np.median(np.abs(np.diff(np.sort(gy)))))
    else:
        return None
    if spacing < 1e-6:
        return None
    px_per_yd = spacing / YARDS_PER_CELL
    det = np.array([y for y, _ in lines], float)
    d = np.abs(det[:, None] - gy[None, :]).min(axis=1)
    return float(np.median(d) / px_per_yd)


def carry_runs(rows: list) -> list:
    """[(start, stop, key_before, key_after)] of uncertified runs.

    `rows` is the walk's per-frame states in frame order. A run is bounded by
    certified frames; either bound may be absent at the clip edges.
    """
    out = []
    n = len(rows)
    i = 0
    while i < n:
        if rows[i].tier == "CERTIFIED":
            i += 1
            continue
        j = i
        while j < n and rows[j].tier != "CERTIFIED":
            j += 1
        before = i - 1 if i > 0 and rows[i - 1].tier == "CERTIFIED" else None
        after = j if j < n and rows[j].tier == "CERTIFIED" else None
        out.append((i, j, before, after))
        i = j
    return out
