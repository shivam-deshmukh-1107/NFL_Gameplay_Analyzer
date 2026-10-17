"""Stage 5 - the physical camera, and every refusal gate.

PROVENANCE
----------
Ported from Yash's `camera_model.py`, which is the most carefully evidenced
code in either parent project: nearly every constant below carries the frame
number and the measurement that set it. **Those comments are the asset. Do not
change a constant without re-measuring, and write the new measurement beside
it.**

One deliberate generalisation: `(fx, fy)` instead of a single focal, so an
anisotropically resampled picture is modelled exactly rather than excluded.
`ay = fy/fx` is derived from the picture geometry alone and is 0.999 on these
clips, so the model is numerically Yash's wherever the source is undistorted.

WHY A CAMERA AND NOT A HOMOGRAPHY
---------------------------------
Every anchor Yash's project used before 2026-09-07 was near-affine: clip_04's
mapped the whole 100-yard field into 600 px of image with a largest/smallest
5-yard gap ratio of 1.06, against 2.80 for the paint it described. Two of the
four were also 2x out in depth scale, labelling six CONSECUTIVE painted lines
0,10,20,30,40,50 when consecutive painted lines are 5 yards apart.

Re-fitting a free 8-dof homography does not prevent that, because an affine map
is a perfectly good homography and nothing in the parameterisation objects. A
pinhole camera looking at a plane CANNOT produce a foreshortening-free image of
that plane, so fitting the camera removes the failure by construction:

    H = K [ r1 | r2 | -R C ],   K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

Six parameters: focal, azimuth, elevation, roll, downfield position, height.
The camera's LATERAL position is held at field centre - yard lines are lines of
constant downfield distance and infinite laterally, so sliding the camera
sideways maps the family onto itself and they cannot constrain it. That is an
explicit assumption; it lives in the one direction the evidence does not reach,
so it cannot corrupt the downfield axis. The cross-field axis is not solved here.

ROLL COMES FROM THE PAINT, NEVER FROM A MOTION CHAIN
----------------------------------------------------
Measured on clip_01: 70 frames after the anchor, a pure ORB chain claims the
yard lines run at 23.3 deg while the paint is measurably at 0.0 deg - with 600+
inliers per frame and no warning of any kind. The dominant paint direction is a
DIRECT observation of roll from a handful of segments, so `set_roll_from_paint`
pins it every frame.

ADAPTIVE DEGREES OF FREEDOM
---------------------------
A frame showing two yard lines cannot determine six parameters, but it
determines roll very well. The solver frees only what the evidence supports and
holds the rest at the previous frame's values.

=============================================================================
THE ONE RULE THAT MATTERS MOST
=============================================================================
**A false accept is far worse than a false refusal.** Never widen a gate to
raise coverage. Certify on `solve_ok`, never on `applied` - `applied` is true
whenever ROLL alone could be updated, which is nearly every frame, and
conflating them certified 68 of 798 frames that were carrying a held-over pose.

**Never relax `MAX_JUMP_YD`.** It is the only thing that catches the exact
scene-scale degeneracy, which is invisible to every line-position metric and is
a legal solver output.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import geometry as G
from .config import FIELD_WIDTH_YD, px_scale

MID_LAT = FIELD_WIDTH_YD / 2.0

#: Image columns the per-line residual is evaluated at, as fractions of width.
SAMPLE_FRAC = (0.2, 0.5, 0.8)

# --------------------------------------------------------------------------- #
# the refusal gates
#
# Every px value is measured at config.PX_REF_W and scaled by px_scale(W).
# Every yard and degree value is scale-free and is used as written.
# --------------------------------------------------------------------------- #
MAX_FIT_RESID_PX = 4.0
"""Median per-line residual above which the frame is refused."""

OUTLIER_PX = 6.0
"""A line beyond this (or 3x the median) is dropped and the fit repeated."""

MAX_JUMP_YD = 3.0
"""Model movement, in yards, that refuses the frame. **NEVER RELAX THIS.**

It is the only guard that catches the exact scene-scale degeneracy: scaling
`(Cx, Cz)` by k and relabelling yards by k gives PIXEL-IDENTICAL yard lines
(max |dy| = 3e-12 px at k = 0.5/2/4), reached at 0.000 px residual, and the
solver will produce it. It has happened for real on this footage (clip_04
f159-163). Removing this guard took clip_04's scale_ratio p90 to 2.8.

TRIED AND REVERTED 2026-09-24: advancing the REFERENCE pose on a refused jump
(below 5 yd, good residual) to stop a stale-reference cascade. It did NOT help
the clip it was written for and it broke the invariant this guard protects -
`density_max` went 1.17 -> 1.43 on clip_01 and 1.12 -> 1.91 on clip_05 against
a documented "no frame anywhere at or above 1.5". The 14 frames it gained on
clip_01 were false accepts. **Refusing a frame AND freezing the reference is
deliberate; do not separate the two.**
"""

MAX_ROLL_DISAGREE = 3.0
"""Degrees the solved pose's yard-line direction may differ from the paint's.

THE BAR IS UNCHANGED; WHAT IT IS COMPARED AGAINST WAS FIXED (2026-09-25). The
paint's `dominant` is a LENGTH-WEIGHTED AVERAGE over every segment in the
family. The old check compared it with the angle of ONE model line, the one
nearest the centre row. On clips 01-04 the yard lines barely fan and the two
agree; in an oblique view they fan from ~25 to ~55 deg and the two are
different quantities. Measured on clip_06 f683-765, candidates scored by an
independent pixel-level paint check:

    correct solves f750-753    old 5.96-6.03 deg REFUSED   like-for-like 0.00-0.01
    correct accepted f756-764  old 0.59-2.98 deg           like-for-like 0.01-0.03
    f755 with roll set -5 deg  old 2.26 -- would PASS       like-for-like 4.70
    f755 with roll set +5 deg  old 7.74                    like-for-like 4.70

So the old check refused correct poses AND could not see a real 5 deg roll
error in the wrong direction. Use `paint_roll_disagreement` whenever segment
support is available.
"""

MIN_SOLVE_LINES = 4
"""Fewer lines than this is never certified.

With three lines `FREE_BY_N` frees only azimuth, roll and Cx; elevation and
height are inherited from the last solve, so the frame is only as right as that
solve still is. MEASURED against independent paint stripes (per-frame median,
yd): clip_01's 3-line frames 0.24 median with f208 at 1.94, f207 1.09, f213
0.89, and f188/f190 at density 0.5, f200 at 1.8 - f188 and f200 visibly cross
the paint at a different angle at full resolution. A 1-yard tick ruler measured
along the clip_01 border also puts f200's two top lines 5 yd apart where its
pose labels them 10.
"""

MIN_KEPT_AFTER_REJECTION = 4
"""If outlier rejection leaves fewer than this, the residual is not evidence.

The refit then had five free parameters against three lines - six constraints
for five unknowns, near-exact by construction. MEASURED on clip_06: f1093
reported 0.021 px while two of its four detected lines sat 12 and 40 px off the
drawn grid; f1170 reported 0.008 px with lines 31 and 74 px off; an independent
pixel check puts both grids BETWEEN the painted lines. Re-solving with fewer
free parameters was tried first and REVERTED: it admitted clip_01 f214, which
broke yard identity by dense flow and took density_max to 1.50.
"""

MAX_LOG_HEIGHT_RATE = 0.2
MAX_ELEV_RATE_DEG = 3.0
"""The camera cannot change height or elevation faster than measured.

Rates are per frame since the last certified solve. MEASURED over every
consecutive pair of certified >=4-line frames on all six of Yash's clips (1,082
pairs, gap <= 3): |d log height| per frame max 0.172 (p99 0.075-0.124),
|d elevation| per frame max 2.51 deg (p99 0.85-2.12).

Why it is needed: four parallel lines carry about five independent numbers and,
with the focal pinned, the solve has five unknowns - so a 4-line fit is nearly
exact by construction. clip_01 f206 fitted its 4 lines to 0.18 px with the
camera dropped from 8.6 to 2.0 yd and elevation from -26.6 to -16.1 deg in six
frames; its grid bunches into the banner and the paint check fails it.
MAX_JUMP_YD, which probes only the centre column, passed it.
"""

MAX_LINE_DIR_DEG = 3.0
"""Every line the fit USED must point the way its solved yard line points.

MEASURED on clip_01 f206, certified in an earlier run: its "4 lines" were yards
20/30/35 plus the flat bottom edge of a "46 YARD RETURN" banner (y=235, slope
0.001), which crosses the centre column exactly where yard 25 would. The ladder
named it 25, the fit bent the perspective to take it - the next line drawn in
the frame corner, while a painted line crosses the lower half with no grid on
it - and the median residual never saw it.

Angle between each detected line and the solved yard line at the same place, on
certified frames: median 0.06-0.10 deg, p90 0.18-0.67, on every clip. f206 read
9.2 deg. The threshold sits above clips 02/04/05's worst (1.89/0.48/2.39, which
include lines the solve did not use).
"""

MAX_YARD_ANGLE_SPREAD_DEG = 25.0
"""Projected yard lines in a real camera view of a plane fan by a few degrees.

THIS EXISTS BECAUSE A SINGLE-LINE ANGLE CHECK LIED. On clip_01 frame 240 (27
deg of real roll, one detected line) the pose had degenerated so far that the
projected grid ran nearly VERTICAL down the picture - visibly, unmistakably
wrong - while the model-versus-paint angle error read 0.03 deg, because that
metric reports only the line nearest frame centre.
"""

CHAIN_IDENTITY_MAX_YD = 10.0
CHAIN_AGREE_MAX_YD = 2.5
"""Guards for labelling lines from a motion prediction rather than the previous
pose. Not yet reachable in this project - there is no motion chain until stage
6 - and ported now so the gate sequence is complete and testable.

MEASURED on clip_06 f683-765: real camera motion reaches 2.9 yd between unique
frames; past 2.5 yd the rounding in `assign_yards` picks the NEIGHBOURING line,
the solve follows it, and a slip of 5 yd minus 2.9 yd of motion reads 2.1 yd -
under MAX_JUMP_YD. f751 slipped that way and 69 frames inherited a 5 yd offset.
`density`, `err_yd` and paint coverage are ALL blind to it, because a grid
shifted by one painted line lands on paint. Two independent instruments found
it: dense optical flow, and the ORB prediction itself.
"""

NEW_PATH_MIN_LINES = 4
"""Below this, the like-for-like roll pin and chain labelling do not apply.

With 2-3 lines the solve frees only 2-3 parameters and lands wherever the warm
start sends it, so changing the roll pin changes the ANSWER rather than the
evidence. MEASURED on clip_01's high-roll tail: the like-for-like pin let
f202/212/215/235/241 (2-3 lines, refused before on a 5-10 px residual) fit at
0.9-3.8 px - and an independent pixel check puts only 33-40% of their grid
lines on paint, one of them a too-sparse grid.
"""

#               f      az     el   roll    Cx     Cz
FREE_BY_N = {0: np.array([0, 0, 0, 0, 0, 0], bool),
             1: np.array([0, 0, 0, 0, 1, 0], bool),
             2: np.array([0, 0, 0, 1, 1, 0], bool),
             3: np.array([0, 1, 0, 1, 1, 0], bool)}
FREE_ALL = np.ones(6, bool)


def gates_for(picture_w: float) -> dict:
    """The px-valued gates, scaled to this picture. Yards/degrees are absolute."""
    s = px_scale(picture_w)
    return {"max_fit_resid_px": MAX_FIT_RESID_PX * s,
            "outlier_px": OUTLIER_PX * s,
            "px_scale": s}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def aspect_ay(W: float, Hgt: float) -> float:
    """`fy / fx` for a picture of this size holding a 4:3 render.

    Derived from the picture alone, not fitted: a 4:3 render resampled to W x H
    has horizontal scale W/W_ref and vertical scale H/(W_ref*3/4), so the ratio
    is 4H/(3W). It is 0.9990 on these clips, i.e. the model is numerically the
    single-focal one wherever the source is undistorted, and it is the whole
    reason an anisotropically stretched upload does not have to be excluded.
    """
    return (4.0 * float(Hgt)) / (3.0 * float(W))


def camera_H(p, W, Hgt, ay: float | None = None, cy_field: float = MID_LAT):
    """p = [focal_x_px, azimuth, elevation, roll, Cx_yd, Cz_yd] -> H.

    H maps (yard, lateral) -> pixel.
    """
    f, az, el, roll, cx_f, cz = p
    f = abs(f) + 1e-6
    if ay is None:
        ay = aspect_ay(W, Hgt)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = G.cross3(fwd, np.array([0.0, 0.0, 1.0]))
    nr = np.linalg.norm(right)
    if nr < 1e-9:
        return None
    right /= nr
    up_c = G.cross3(right, fwd)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.vstack([right * cr + up_c * sr, -(-right * sr + up_c * cr), fwd])
    C = np.array([cx_f, cy_field, cz])
    K = np.array([[f, 0.0, W / 2.0],
                  [0.0, f * ay, Hgt / 2.0],
                  [0.0, 0.0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def line_y_at(H, yard, xs):
    """Image y of the projected field line X=yard at image columns `xs`.

    Via the homogeneous line through two of its points, so it stays valid when
    one of them falls behind the camera.
    """
    p1 = H @ np.array([yard, 0.0, 1.0])
    p2 = H @ np.array([yard, FIELD_WIDTH_YD, 1.0])
    L = G.cross3(p1, p2)
    if abs(L[1]) < 1e-12 or not np.all(np.isfinite(L)):
        return None
    return -(L[0] * xs + L[2]) / L[1]


def pixel_to_field(H, px, py):
    """Inverse-project image point(s) through H. Raises on a singular H."""
    scalar = np.ndim(px) == 0
    px_arr = np.atleast_1d(np.asarray(px, float))
    py_arr = np.atleast_1d(np.asarray(py, float))
    inv = np.linalg.inv(np.asarray(H, float))
    P = np.column_stack([px_arr, py_arr, np.ones(len(px_arr))]) @ inv.T
    w = P[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        yard = P[:, 0] / w
        lat = P[:, 1] / w
    if scalar:
        return float(yard[0]), float(lat[0])
    return yard, lat


def grid_lines_image(H, W, Hgt, lo: float = -30.0, hi: float = 150.0,
                     step: float = 5.0):
    """(yard, y_at_centre, slope) for every 5-yard line crossing the frame."""
    xc = W / 2.0
    xs = np.array([xc - 100.0, xc, xc + 100.0])
    out = []
    for yard in np.arange(lo, hi + 1e-9, step):
        yy = line_y_at(H, float(yard), xs)
        if yy is None or not np.all(np.isfinite(yy)):
            continue
        if not (-0.3 * Hgt <= yy[1] <= 1.3 * Hgt):
            continue
        out.append((float(yard), float(yy[1]), float((yy[2] - yy[0]) / 200.0)))
    return out


def yard_angle(H, W, Hgt):
    """Image angle (deg) of the projected yard line nearest frame centre.

    Reported instead of the roll PARAMETER, which goes stale on any frame where
    the solve is rejected - exactly the frames where the truth matters most.

    Describe a line by where it CROSSES THE FRAME, never by the field point at
    the field's lateral centre: that point moves when the camera's lateral
    position changes even though the line's image does not, and a metric built
    on it reported a p90 angle jump of 1.09 -> 13.53 deg on a change that
    cannot move a yard line by a floating-point ulp.
    """
    g = grid_lines_image(H, W, Hgt)
    if not g:
        return None
    k = int(np.argmin([abs(y - Hgt / 2.0) for _, y, _ in g]))
    return float(np.degrees(np.arctan(g[k][2])))


def yard_angle_spread(H, W, Hgt):
    """Max-minus-min image angle over the projected yard lines, in degrees."""
    g = grid_lines_image(H, W, Hgt)
    if not g:
        return None
    if len(g) < 2:
        return 0.0
    a = [np.degrees(np.arctan(sl)) for _, _, sl in g]
    return float(max(a) - min(a))


# --------------------------------------------------------------------------- #
# residuals
# --------------------------------------------------------------------------- #
def per_line_resid(H, lines, yards, W):
    """Median |model y - detected y| per line, over `SAMPLE_FRAC` columns."""
    xs = np.array([f * W for f in SAMPLE_FRAC])
    xc = W / 2.0
    out = []
    for (y_c, sl), w in zip(lines, yards):
        yy = line_y_at(H, w, xs)
        out.append(1e4 if yy is None or not np.all(np.isfinite(yy))
                   else float(np.median(np.abs(yy - (y_c + sl * (xs - xc))))))
    return np.array(out)


def line_direction_errors(H, lines, yards, W):
    """Degrees between each detected line and its solved yard line. See
    MAX_LINE_DIR_DEG."""
    xs = np.array([W / 2.0 - 100.0, W / 2.0 + 100.0])
    out = []
    for (y_c, sl), w in zip(lines, yards):
        yy = line_y_at(H, float(w), xs)
        if yy is None or not np.all(np.isfinite(yy)):
            out.append(90.0)
            continue
        msl = (yy[1] - yy[0]) / 200.0
        out.append(abs(float(np.degrees(np.arctan(sl) - np.arctan(msl)))))
    return out


def paint_roll_disagreement(H, ang, support):
    """|model direction - paint direction| in degrees, LIKE FOR LIKE.

    `ang` is the length-weighted dominant angle of the paint segments in
    `support` = (midpoints Nx2, weights N). The model's yard-line direction is
    evaluated AT EACH OF THOSE MIDPOINTS and averaged with THE SAME WEIGHTS, so
    the two numbers differ only in where the direction came from.

    Scale-blind by construction - the scene-scale degeneracy preserves every
    direction - which is CORRECT for a roll check: depth scale is
    `MAX_JUMP_YD`'s job and this must not shadow it.
    """
    if ang is None or support is None:
        return None
    mid, wts = support
    mid = np.asarray(mid, float)
    wts = np.asarray(wts, float)
    if len(mid) == 0 or wts.sum() <= 0:
        return None
    try:
        inv = np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return None
    P = np.column_stack([mid, np.ones(len(mid))]) @ inv.T
    X = P[:, 0] / P[:, 2]
    Y = P[:, 1] / P[:, 2]
    Hm = np.asarray(H, float)
    a = np.column_stack([X, Y - 0.5, np.ones(len(X))]) @ Hm.T
    b = np.column_stack([X, Y + 0.5, np.ones(len(X))]) @ Hm.T
    a = a[:, :2] / a[:, 2:3]
    b = b[:, :2] / b[:, 2:3]
    md = np.degrees(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0]))
    dev = (md - ang + 90.0) % 180.0 - 90.0
    ok = np.isfinite(dev)
    if not ok.any():
        return None
    return abs(float(np.average(dev[ok], weights=wts[ok])))


def yard_shift(H_old, H_new, W, Hgt):
    """Model movement in YARDS at three depths down the centre column.

    MEASURED IN PIXELS THIS GUARD IS USELESS: probes near the horizon move
    hundreds of pixels for a fraction of a degree, and it fired on 74 of
    clip_02's 169 frames for changes invisible on screen. What it is for is
    catching a jump of a whole 5-yard line.
    """
    xs = np.array([W / 2.0, W / 2.0, W / 2.0])
    ys = np.array([float(fr) * Hgt for fr in (0.15, 0.30, 0.50)])
    try:
        a_yard, a_lat = pixel_to_field(H_old, xs, ys)
        b_yard, b_lat = pixel_to_field(H_new, xs, ys)
    except np.linalg.LinAlgError:
        return 1e6
    if not (np.all(np.isfinite(a_yard)) and np.all(np.isfinite(b_yard))):
        return 1e6
    return float(np.median(np.abs(b_yard - a_yard)))


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_camera(lines, yards, W, Hgt, base, free=None, multistart: bool = False,
               ay: float | None = None):
    """Least squares over the free subset of the six parameters.

    `cauchy` loss, not `soft_l1`: a single mis-associated line otherwise drags
    the whole fit, which is the plausible mechanism of clip_02's large
    excursion.
    """
    if ay is None:
        ay = aspect_ay(W, Hgt)
    free = FREE_ALL if free is None else free
    lines = list(lines)
    yards = np.asarray(yards, float)
    starts = []
    if base is not None:
        starts.append(np.asarray(base, float))
    if multistart or base is None:
        # THE CAMERA MUST START ON THE CORRECT SIDE OF THE FIELD IT IS LOOKING
        # AT, otherwise the solve settles into a near-horizontal, tiny-focal
        # nonsense pose: getting this sign wrong cost clip_02 a 45.6 px anchor
        # residual (elevation -1.1 deg, focal 434) where the right sign gives
        # 0.56 px.
        yard_up = yards[0] > yards[-1]
        sgn = 1.0 if yard_up else -1.0
        for f0 in (700.0, 1100.0, 1600.0, 2400.0):
            for el0 in (-0.15, -0.3, -0.5):
                for cz0 in (8.0, 18.0, 35.0):
                    starts.append(np.array([f0, 0.0 if yard_up else np.pi, el0,
                                            0.0,
                                            float(np.mean(yards)) - sgn * 40.0,
                                            cz0]))
    # A PARAMETER THAT IS NOT FREE MUST HOLD THE SAME VALUE IN EVERY START.
    # Otherwise "locked" silently becomes "chosen by multistart", and for the
    # focal that is catastrophic rather than merely untidy, because the focal
    # is NOT OBSERVABLE from yard lines: every candidate then fits equally well
    # and the winner is arbitrary.
    #
    # CAUGHT 2026-10-17 BY MEASUREMENT (docs/LESSONS). `bootstrap_pose` locks
    # the focal and asks for a multistart, whose grid assigns each candidate its
    # own f0 from (700, 1100, 1600, 2400). Every certified frame of clip_01 and
    # clip_02 came back with focal pinned at exactly 700.0 - the first grid
    # value - and therefore elevation -49.4 deg where Yash measured -28.5, and
    # height 13.0 yd where he measured 16-19. His own focal table gives
    # elevation -49.1 deg at focal 700, so the failure is visible in one number.
    #
    # The yard LINE POSITIONS were unaffected, which is exactly what makes this
    # worth a comment: the scene-scale/focal degeneracy means the grid still
    # lands on the paint, the residual is still 0.69 px, and the absolute scale
    # check still passes. Only height, elevation and anything cross-field were
    # wrong, silently.
    if base is not None and not np.all(free):
        locked = ~np.asarray(free, bool)
        b = np.asarray(base, float)
        starts = [np.where(locked, b, np.asarray(st, float)) for st in starts]

    det = np.array([[y, s] for y, s in lines], float)
    xs = np.array([f * W for f in SAMPLE_FRAC])
    xc = W / 2.0
    ydet = det[:, 0][:, None] + det[:, 1][:, None] * (xs[None, :] - xc)

    best = None
    for st in starts:
        lo = np.array([250.0, st[1] - 1.2, -1.45, -1.3, -400.0, 1.5])
        hi = np.array([9000.0, st[1] + 1.2, -0.02, 1.3, 500.0, 200.0])
        st = np.clip(st, lo + 1e-6, hi - 1e-6)

        def full(q, st=st):
            p = st.copy()
            p[free] = q
            return p

        def resid(q, st=st):
            H = camera_H(full(q, st), W, Hgt, ay)
            if H is None:
                return np.full(det.shape[0] * len(xs), 1e4)
            out = []
            for i, w in enumerate(yards):
                yy = line_y_at(H, w, xs)
                out.append(np.full(len(xs), 1e4)
                           if yy is None or not np.all(np.isfinite(yy))
                           else np.clip(yy - ydet[i], -1e4, 1e4))
            return np.concatenate(out)

        if free.sum() == 0:
            H = camera_H(st, W, Hgt, ay)
            if H is not None:
                r = per_line_resid(H, lines, yards, W)
                if best is None or np.median(r) < best[0]:
                    best = (float(np.median(r)), H, st.copy(), r)
            continue
        try:
            o = least_squares(resid, st[free], loss="cauchy", f_scale=0.75,
                              bounds=(lo[free], hi[free]),
                              max_nfev=300, xtol=1e-12, ftol=1e-12)
        except Exception:                                   # noqa: BLE001
            continue
        p = full(o.x, st)
        H = camera_H(p, W, Hgt, ay)
        if (H is None or not np.all(np.isfinite(H))
                or abs(np.linalg.det(H)) < 1e-14):
            continue
        r = per_line_resid(H, lines, yards, W)
        if best is None or np.median(r) < best[0]:
            best = (float(np.median(r)), H, p, r)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def set_roll_from_paint(p, ang_deg, W, Hgt, support=None, ay=None):
    """Pin roll so the model's yard-line direction matches the measured paint.

    SCANNED, not gradient-solved: the model's yard-line image angle is a
    DISCONTINUOUS function of roll, because the "yard line nearest frame
    centre" switches to a different line as the camera turns. `least_squares`
    stalls at those jumps - measured, it reached 14.2 deg when the paint was at
    27.0 deg and stopped. A coarse scan plus a local refine costs ~90 3x3
    matrix builds and cannot get stuck.

    With `support` the match is LIKE FOR LIKE. It matters here too, because the
    pinned pose is the `H_pred` that `MAX_JUMP_YD` measures against: MEASURED
    on clip_06 f730-753, on DUPLICATED video frames where consecutive candidate
    solves differ by 0.00 yd, the old pin moved the reference far enough that
    the jump guard read 2.19-2.85 yd - most of the 3.0 yd budget spent on
    motion that did not happen.
    """
    if ang_deg is None or p is None:
        return p
    if ay is None:
        ay = aspect_ay(W, Hgt)

    def err(q):
        pp = np.asarray(p, float).copy()
        pp[3] = q
        H = camera_H(pp, W, Hgt, ay)
        if H is None:
            return 1e4
        if support is not None:
            d = paint_roll_disagreement(H, ang_deg, support)
            return 1e4 if d is None else d
        a = yard_angle(H, W, Hgt)
        return 1e4 if a is None else abs((a - ang_deg + 90.0) % 180.0 - 90.0)

    grid = np.radians(np.arange(-70.0, 70.1, 1.5))
    vals = [err(q) for q in grid]
    k = int(np.argmin(vals))
    if vals[k] > 1e3:
        return p
    fine = np.linspace(grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)], 41)
    vf = [err(q) for q in fine]
    j = int(np.argmin(vf))
    if vf[j] > 0.5:
        return p
    out = np.asarray(p, float).copy()
    out[3] = float(fine[j])
    return out


# --------------------------------------------------------------------------- #
# labelling
# --------------------------------------------------------------------------- #
def assign_yards(H_pred, lines, W, cum=None):
    """Yard labels: SPACING from the paint's own ladder, ORIGIN and DIRECTION
    from the predicted pose.

    An earlier version read every label straight off the chain and rejected the
    frame when they disagreed with the paint's ladder. That is backwards -
    disagreement means the prediction has drifted, which is the frame that most
    needs correcting - and it cost 89 of clip_03's 182 frames.

    `cum`, when given, is the cumulative rung count from `ladder.assign_ladder`
    on the SAME sorted lines, so the ladder is not recomputed here. That is the
    merge's one change: the ladder now has a second opinion behind it
    (`propose.compare`) and recomputing it would discard that.
    """
    ys = np.sort(np.array([y for y, _ in lines], float))
    if len(ys) < 2:
        return None
    try:
        yard, lat = pixel_to_field(H_pred, np.full(len(ys), W / 2.0), ys)
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(yard)) and np.all(np.isfinite(lat))):
        return None
    w_chain = yard
    if cum is not None and len(cum) == len(ys):
        cum = np.asarray(cum, float)
    elif len(ys) >= 4:
        from . import ladder as LAD
        lad = LAD.assign_ladder(ys, prior=w_chain)
        if lad is None:
            return None
        cum = np.asarray(lad[1], float)
    else:
        steps = np.maximum(1, np.round(np.abs(np.diff(w_chain)) / 5.0)).astype(int)
        cum = np.concatenate([[0], np.cumsum(steps)]).astype(float)
    sign = -1.0 if w_chain[0] > w_chain[-1] else 1.0
    cand = sign * 5.0 * cum
    return cand + 5.0 * np.round(np.median(w_chain - cand) / 5.0)


# --------------------------------------------------------------------------- #
# the per-frame solve
# --------------------------------------------------------------------------- #
def new_info() -> dict:
    return {"applied": False, "roll_set": False, "reason": None, "n_lines": 0,
            "n_free": 0, "resid_px": None, "jump_yd": None,
            "paint_angle_deg": None, "solve_ok": False, "n_dropped": 0}


def solve_frame(H_pred, lines, ang, W, Hgt, base, lock_focal: bool = False,
                paint_support=None, cum=None, gap: int = 1, ay=None,
                H_chain=None, chain_identity: bool = False):
    """One frame. Returns `(H, params, info)`.

    `info["solve_ok"]` is the ONLY certification signal. `info["applied"]` is
    true whenever roll alone could be updated, which is nearly every frame;
    conflating the two certified 68 of 798 frames that were carrying a
    held-over pose with only roll refreshed.

    Every exit below returns `H_pred` and the UNCHANGED `base`, so a refused
    frame freezes the reference rather than advancing it. That pairing is
    deliberate - see `MAX_JUMP_YD`.

    `chain_identity` names the detected lines from the motion-chained
    prediction `H_chain` instead of from the roll-pinned previous pose, and
    then requires the solve to agree with that prediction. Geometry, the warm
    start and `MAX_JUMP_YD` are unchanged either way: **only the LABELLING uses
    the chain.**
    """
    if ay is None:
        ay = aspect_ay(W, Hgt)
    gates = gates_for(W)
    info = new_info()
    lines = list(lines or [])
    info["n_lines"] = len(lines)
    info["paint_angle_deg"] = ang

    extra_roll_guard = None
    if len(lines) < NEW_PATH_MIN_LINES:
        extra_roll_guard = paint_support      # refusal only - see NEW_PATH_MIN_LINES
        paint_support = None

    if base is None:
        info["reason"] = "no camera state"
        return H_pred, base, info

    # Roll first, from the paint, on every frame that has any - applied whether
    # or not the rest of the solve succeeds.
    base = set_roll_from_paint(base, ang, W, Hgt, support=paint_support, ay=ay)
    if ang is not None:
        H_roll = camera_H(base, W, Hgt, ay)
        if H_roll is not None:
            info["roll_set"] = True
            H_pred = H_roll
    if len(lines) < 1:
        info["reason"] = "no usable paint"
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # WHICH POSE NAMES THE LINES. Yash's `assign_yards` has always taken the
    # ORIGIN and DIRECTION from a predicted pose, and for a long time his
    # `solve_frame` overwrote the chained prediction with the roll-pinned
    # PREVIOUS pose before labelling - so the motion his chain measured was
    # discarded on every frame that had a paint angle, which is why swapping
    # the chain for something better "changed nothing" when he first tested it.
    #
    # MEASURED consequence of getting this wrong, on his clip_06 f683-765: real
    # camera motion reaches 2.9 yd between unique frames, and past 2.5 yd the
    # rounding in `assign_yards` picks the NEIGHBOURING line. The solve follows
    # it, and a slip of 5 yd minus 2.9 yd of real motion reads 2.1 yd - under
    # MAX_JUMP_YD. f751 slipped that way and 69 frames inherited a 5 yd offset,
    # invisible to density, err_yd and paint coverage alike.
    H_label = H_pred
    if chain_identity and H_chain is not None and H_pred is not None:
        moved = yard_shift(H_pred, H_chain, W, Hgt)
        info["chain_motion_yd"] = round(moved, 3)
        if moved <= CHAIN_IDENTITY_MAX_YD:
            H_label = H_chain
            info["labelled_from_chain"] = True
    yards = assign_yards(H_label, lines, W, cum=cum)
    if yards is None or len(set(yards.tolist())) != len(yards):
        info["reason"] = "yard assignment failed"
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    free = FREE_BY_N.get(len(lines), FREE_ALL).copy()
    if lock_focal:
        # FOCAL IS A GAME CONSTANT, NOT A PER-FRAME UNKNOWN. For a plane, focal
        # length and camera distance are very nearly degenerate: enlarge f and
        # pull the camera back and the image barely changes. When the camera
        # zooms in the visible yard lines bunch together, the depth baseline
        # collapses, and the pair stops being separable at all - the solve then
        # spends focal to absorb whatever else is wrong. Measured focal spread
        # WITHIN clip_04 by visible line count: 8+ lines 11%, 4-7 lines 59%,
        # <4 lines 45%, medians 1689 / 2119 / 2830. A renderer does not change
        # its field of view by 67% because fewer yard lines are in shot.
        free[0] = False
    info["n_free"] = int(free.sum())

    order = np.argsort([y for y, _ in lines])
    ls = [lines[i] for i in order]
    H, p, r = fit_camera(ls, yards, W, Hgt, base, free, ay=ay)
    if H is None:
        info["reason"] = "fit failed"
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # Outlier rejection from 4 lines up. A single stray line - a sideline, a
    # shadow edge, a mis-merged pair - otherwise fails the whole frame, and
    # residual-based refusals were 113 of the 150 frames being declined.
    n_kept = len(ls)
    fit_ls, fit_yd = ls, np.asarray(yards)
    if len(ls) >= 4:
        keep = r <= max(gates["outlier_px"], 3.0 * float(np.median(r)))
        if keep.sum() >= 3 and not keep.all():
            ls2 = [ls[i] for i in range(len(ls)) if keep[i]]
            H2, p2, r2 = fit_camera(ls2, yards[keep], W, Hgt, p, free, ay=ay)
            if H2 is not None and np.median(r2) < np.median(r[keep]):
                H, p, r = H2, p2, r2
                n_kept = int(keep.sum())
                fit_ls, fit_yd = ls2, np.asarray(yards)[keep]
    info["n_kept"] = n_kept

    # A warm start can be stale after a cut or a fast zoom and trap the solve in
    # a bad basin. If the fit is poor, pay for one cold multistart before giving
    # up - refusing a frame that is actually solvable is the more expensive
    # error here.
    if float(np.median(r)) > gates["max_fit_resid_px"] and free.sum() >= 4:
        cold = FREE_ALL.copy()
        if lock_focal:
            cold[0] = False
        H3, p3, r3 = fit_camera(ls, yards, W, Hgt,
                                base if lock_focal else None, cold,
                                multistart=not lock_focal, ay=ay)
        if H3 is not None and np.median(r3) < np.median(r):
            H, p, r = H3, p3, r3
            n_kept = len(ls)
            info["n_kept"] = n_kept
            fit_ls, fit_yd = ls, np.asarray(yards)

    # ---- GATE 1: the fit must actually fit ------------------------------- #
    med = float(np.median(r))
    info["resid_px"] = round(med, 3)
    if med > gates["max_fit_resid_px"]:
        info["reason"] = "residual %.1f px" % med
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 2: enough lines survived rejection to be evidence ---------- #
    if len(ls) >= 4 and n_kept < MIN_KEPT_AFTER_REJECTION:
        info["reason"] = ("only %d of %d lines survive outlier rejection"
                          % (n_kept, len(ls)))
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 3: a 3-line solve is not certified ------------------------- #
    if len(ls) < MIN_SOLVE_LINES:
        info["reason"] = ("only %d lines: the solve frees %d parameters and "
                          "inherits elevation and height"
                          % (len(ls), int(free.sum())))
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 4: every line used points the way its yard line points ----- #
    dirs = line_direction_errors(H, fit_ls, fit_yd, W)
    info["line_dir_max_deg"] = round(float(max(dirs)), 3) if dirs else None
    if dirs and max(dirs) > MAX_LINE_DIR_DEG:
        info["reason"] = ("a line the solve used points %.1f deg off its solved "
                          "yard line (not paint - e.g. a banner edge)"
                          % max(dirs))
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 5: the model cannot jump. NEVER RELAX THIS ----------------- #
    jump = yard_shift(H_pred, H, W, Hgt)
    info["jump_yd"] = round(jump, 2)
    if jump > MAX_JUMP_YD:
        info["reason"] = "moves model %.1f yd" % jump
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 5b: if the chain named the lines, the solve must agree ----- #
    # A NEW REFUSAL, NOT A WIDENED ONE. When the lines were named from the
    # chain, the solve must land within half a rung of the chained prediction;
    # beyond that, the line it was named after is the neighbouring one.
    # MEASURED: correct solves sit 0.00-1.28 yd from the prediction; clip_06
    # f754, whose 13 detected lines include junk at the bottom banner, solved
    # to 0.19 px on a consistent subset 4.97 yd away - one painted line off.
    # MAX_JUMP_YD passed it at 2.8 yd, because 5 yd of slip minus 2.2 yd of
    # real motion looks like 2.8 yd of motion. This guard can only remove
    # frames.
    if info.get("labelled_from_chain"):
        agree = yard_shift(H_chain, H, W, Hgt)
        info["chain_disagree_yd"] = round(agree, 3)
        if agree > CHAIN_AGREE_MAX_YD:
            info["reason"] = "disagrees with chained motion %.1f yd" % agree
            info["applied"] = info["roll_set"]
            return H_pred, base, info

    # ---- GATE 6: the solved pose must agree with the measured paint roll -- #
    # A divergence check on a signal the solve does not use. A solve's own
    # residual cannot diagnose that solve.
    if ang is not None and paint_support is not None:
        dis = paint_roll_disagreement(H, ang, paint_support)
        info["roll_disagree_deg"] = None if dis is None else round(dis, 3)
        if dis is not None and dis > MAX_ROLL_DISAGREE:
            info["reason"] = "solved pose disagrees with measured paint roll"
            info["applied"] = info["roll_set"]
            return H_pred, base, info
    elif ang is not None:
        ha = yard_angle(H, W, Hgt)
        if ha is not None and abs((ha - ang + 90.0) % 180.0 - 90.0) > MAX_ROLL_DISAGREE:
            info["reason"] = "solved pose disagrees with measured paint roll"
            info["applied"] = info["roll_set"]
            return H_pred, base, info
        if extra_roll_guard is not None:
            dis = paint_roll_disagreement(H, ang, extra_roll_guard)
            info["roll_disagree_deg"] = None if dis is None else round(dis, 3)
            if dis is not None and dis > MAX_ROLL_DISAGREE:
                info["reason"] = ("solved pose disagrees with measured paint "
                                  "roll (like for like)")
                info["applied"] = info["roll_set"]
                return H_pred, base, info

    # ---- GATE 7: the grid must not fan like no real camera --------------- #
    spread = yard_angle_spread(H, W, Hgt)
    info["angle_spread_deg"] = None if spread is None else round(spread, 2)
    if spread is not None and spread > MAX_YARD_ANGLE_SPREAD_DEG:
        info["reason"] = "projected yard lines fan %.1f deg" % spread
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    # ---- GATE 8: height and elevation cannot change faster than measured - #
    g = max(1, int(gap))
    dlog = abs(float(np.log(max(p[5], 1e-3) / max(base[5], 1e-3)))) / g
    delv = abs(float(np.degrees(p[2] - base[2]))) / g
    info["height_rate"] = round(dlog, 4)
    info["elev_rate_deg"] = round(delv, 3)
    if dlog > MAX_LOG_HEIGHT_RATE or delv > MAX_ELEV_RATE_DEG:
        info["reason"] = ("camera height/elevation changes %.2f log/%.1f deg "
                          "per frame" % (dlog, delv))
        info["applied"] = info["roll_set"]
        return H_pred, base, info

    info["applied"] = True
    info["solve_ok"] = True
    return H, p, info


# --------------------------------------------------------------------------- #
# cold start
# --------------------------------------------------------------------------- #
#: Bar a frame must clear to be used as a cold-start anchor. Yash's
#: `auto_anchor`/`bootstrap_camera_anchor` values.
ANCHOR_MIN_LINES = 6
ANCHOR_MAX_LADDER_RMS_PX = 1.0


def bootstrap_pose(lines, cum, W, Hgt, lock_focal: bool = True,
                   focal: float | None = None, ay=None):
    """Cold-start a camera from one frame's own 5-yard ladder.

    Returns `(H, p, resid)` or `(None, None, None)`.

    **YARD NAMES FROM THIS ARE ARBITRARY.** The ladder gives integer STEPS, not
    identities, so the origin is whatever this frame's first line happened to
    be. Absolute naming is stage 8.

    A BOOTSTRAPPED ANCHOR IS PROVISIONAL AND ITS OWN RESIDUAL IS NOT EVIDENCE.
    Yash's earlier version shipped anchors with a ~1.8x depth-scale error at
    sub-pixel residual, and two of them scored 6+ distinct yard lines at
    sub-pixel error, passed every single-frame check, and were wrong. Why no
    single-frame check can catch it: if the reader assigns wrong yard VALUES to
    correctly detected lines, the fit is perfect, because it is fitting to
    those same lines. **The error lives in world coordinates and is invisible
    from that frame.** Validate on the whole clip.
    """
    if ay is None:
        ay = aspect_ay(W, Hgt)
    ys = np.sort(np.array([y for y, _ in lines], float))
    if len(ys) < ANCHOR_MIN_LINES or cum is None or len(cum) != len(ys):
        return None, None, None
    order = np.argsort([y for y, _ in lines])
    ls = [lines[i] for i in order]
    yards = 5.0 * np.asarray(cum, float)

    free = FREE_ALL.copy()
    base = None
    if lock_focal:
        if focal is None:
            from .config import GAME_FOCAL_PX
            focal = GAME_FOCAL_PX * px_scale(W)
        free[0] = False
        # A start the focal can be held at; the other five are searched.
        base = np.array([focal, 0.0, -0.45, 0.0, float(np.mean(yards)) - 40.0,
                         16.0])
    H, p, r = fit_camera(ls, yards, W, Hgt, base, free,
                         multistart=True, ay=ay)
    if H is None:
        return None, None, None
    return H, p, float(np.median(r))
