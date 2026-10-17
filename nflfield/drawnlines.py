"""The game's own drawn lines: the line of scrimmage and the line to gain.

PROVENANCE
----------
Ported from Shivam's `blitzfield/pipeline.colour_lines`, with Yash's measured
hue identities (`markings.MARKER_*`) noted.

WHY THESE MATTER MORE THAN THEY LOOK
------------------------------------
Blitz paints the line of scrimmage and the line to gain directly onto the turf
in saturated colour, and the HUD states the exact yard gap between them. That
makes them the ONLY route to an absolute scale check that either parent project
has - every other instrument in both projects compares the geometry against
painted yard lines, which is to say against the same evidence the geometry was
fitted to.

Measured identities (Yash, at the line cores over the middle 40% of the frame):

    line of scrimmage    hue 26-41    S  80-145   V 116-175   yellow/gold
    line to gain         hue 178 / 1  S 140-205   V 100-185   RED
    (real painted line)  hue 87-93    S  63-140   V  96-186   pale cyan

This module FINDS them. `markings.reject_drawn_markers` keeps them OUT of the
white-paint pencil. Both are needed and they are not the same job: a drawn line
at a position that is not a multiple of 5 yards is a guaranteed outlier in the
yard ladder, and on clip_04 f161 one surviving impostor inflated the ladder and
let the solve ride the scale degeneracy to a camera 148 yd in the air.

WHICH IS WHICH IS NOT ESTABLISHED HERE
--------------------------------------
The gap is symmetric, so the scale check does not need to know which colour is
the scrimmage line and which is the line to gain. This module reports both by
colour family and makes no claim about their roles. Yash's hue measurement
above says scrimmage is yellow and the line to gain is red; that has not been
re-verified in this project, so nothing here depends on it.

ONLY `orange_yellow` AND `red` ARE GAME-DRAWN LINES
---------------------------------------------------
MEASURED 2026-10-17 and important, because the other two families look like
candidates and are not:

  * **`cyan` is PAINTED YARD LINE.** The real painted lines here are pale cyan
    at hue 87-93, which sits inside Shivam's cyan band (85-100), and they pass
    `coloured_line_masks`' saturation floor (their S is 63-140 against a floor
    of 40). On clip_02, 21 of 44 sampled frames fit a `cyan` "line" - every one
    of them is paint. Treating it as a drawn line would compare a painted yard
    line against a drawn one and call the result a down-and-distance gap.
  * **`blue`** picks up the end zone and the selection ring, which are large
    and not line-like, but survive on some frames.

`scalecheck.GAP_PAIR` therefore names the two families explicitly and the other
two are reported only so a caller can see what was in the frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .config import det_scale
from .fieldcolor import TurfModel, coloured_line_masks


def fit_drawn_lines(frame_bgr: np.ndarray, turf: TurfModel, prep=None,
                    vp_cross=None, min_area: int = 400, elong: float = 6.0,
                    ang_tol: float = 3.0) -> dict:
    """Fit the game-drawn lines. Returns {colour_name: (4,) segment}.

    Each candidate must be a large, strongly elongated blob that is PARALLEL TO
    THE YARD LINES. Without that last test the blue channel locks onto team
    jerseys instead of the line.

    `min_area` and `elong` are Shivam's, measured at `SHIVAM_REF_W`; the area
    scales by the square of the frame scale. (`ang_tol` is an angle and does
    not scale.)
    """
    s = det_scale(frame_bgr.shape[1])
    min_area = int(round(min_area * s * s))

    out = {}
    masks = (prep.colour_masks if prep is not None and prep.colour_masks is not None
             else coloured_line_masks(frame_bgr, turf))
    for name, m in masks.items():
        if prep is not None:
            m = (m & prep.region & ~prep.hud)
        m = m.astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(lbl == i)
            P = np.stack([xs, ys], 1).astype(float)
            mu = P.mean(0)
            _, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
            if S[1] < 1e-6 or S[0] / S[1] < elong:
                continue
            d = Vt[0]
            t = (P - mu) @ d
            seg = np.array([[*(mu + t.min() * d), *(mu + t.max() * d)]])
            if vp_cross is not None and G.vp_angular_error(
                    seg, np.asarray(vp_cross, float))[0] > ang_tol:
                continue
            if best is None or S[0] > best[0]:
                best = (S[0], seg[0])
        if best is not None:
            out[name] = best[1]
    return out
