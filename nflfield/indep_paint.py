"""Stage 7a - an independent painted-line detector, for grading only.

PROVENANCE
----------
Ported from Yash's `independent_paint.py`.

**THIS SHARES NO CODE WITH THE TRACKER'S DETECTOR.** No thinning, no LSD, no
`merge_collinear`, no flank test, no ladder, no `markings` at all. It finds
bright painted STRIPES - white top-hat, then Hough on the paint mask, then the
dominant family, then vanishing-point consistency - and keeps a line only if at
least `COVERAGE_MIN` of its length over turf is actually paint.

WHY A SECOND DETECTOR IS NECESSARY AND NOT PARANOIA
---------------------------------------------------
Yash's project audit originally re-detected paint with the SAME detector whose
lines the camera had been fitted to, so its error figure graded the fit against
its own inputs, and on ghost-heavy plays it saw the same ghosts. An audit that
runs the subject's own metric proves nothing.

The coverage test is what separates a yard line from everything else that is
long and bright: a yard line crosses the visible field, while digit rows,
banner edges and jerseys are short. Measured against certified grids of his
clips 01-06 (every 3rd frame), `coverage >= 0.5` keeps 86-99% of the lines a
certified grid passes through.

**IT IS A DIFFERENT DETECTOR WITH ITS OWN NOISE, NOT GROUND TRUTH.** The
grid-to-stripe-centre error it reports is an UPPER BOUND on the true error, not
an estimate of it. Use it to compare runs, never to claim an absolute accuracy.

KNOWN BLIND SPOT, inherited and real: it cannot see a grid shifted by a WHOLE
line spacing, because such a grid lands on real paint. That is what
`flowcheck.py` is for. Translucent result banners ("2 YARD RUN") also read as
turf, so it flags correct frames on banner-heavy clips - compare runs, do not
quote absolute counts there.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import px_scale

#: Minimum share of a line's over-turf length that must be paint.
COVERAGE_MIN = 0.5
#: Top-hat kernel, px at PX_REF_W. Deliberately WIDER than the widest painted
#: line seen (~20 px) so a thick near-field line is still a ridge to it. Note
#: this is a different width from the tracker's own ridge filter on purpose:
#: sharing it would share its blind spots.
TOPHAT_K = 25
TOPHAT_MIN = 18
MIN_SEG = 90         # px at PX_REF_W
GROUP_PX = 14.0      # px at PX_REF_W
ANG_WIN = 20.0       # deg around the dominant family
VP_MAX = 2.0         # deg of vanishing-point inconsistency allowed


def _odd(n: float, lo: int = 3) -> int:
    k = int(round(n))
    if k % 2 == 0:
        k += 1
    return max(k, lo)


def paint_mask(frame: np.ndarray) -> np.ndarray:
    """Bright, near-neutral, turf-surrounded pixels. Nothing else."""
    s = px_scale(frame.shape[1])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = _odd(TOPHAT_K * s)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    green = cv2.inRange(hsv, (30, 40, 25), (95, 255, 255)) > 0
    blur = _odd(41 * s)
    ctx = cv2.blur(green.astype(np.float32), (blur, blur)) > 0.55
    m = (th > TOPHAT_MIN) & (hsv[:, :, 1] < 110) & ctx
    return m.astype(np.uint8) * 255


def _hom(y: float, s: float, W: float) -> np.ndarray:
    xc = W / 2.0
    return np.cross([0.0, y - s * xc, 1.0], [float(W), y + s * (W - xc), 1.0])


def vp_filter(lines, W):
    """Keep only lines consistent with one vanishing point.

    MEASURED on ~2650 real lines across six clips: the per-line
    vanishing-point residual maxes at 0.67 deg with p99 <= 0.42, so this is a
    SAFE signal. (A detector stage built on it was tried and reverted - as a
    way to RECOVER lines it gained almost nothing and cost accuracy - but as a
    consistency filter here it is sound.)
    """
    if len(lines) < 3:
        return lines
    Ls = [_hom(y, s, W) for y, s in lines]
    angs = np.array([np.arctan(s) for _, s in lines])
    pts = np.array([[W / 2.0, y] for y, _ in lines])
    best = None
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            vp = np.cross(Ls[i], Ls[j])
            nv = np.linalg.norm(vp)
            if not np.isfinite(nv) or nv < 1e-12:
                continue
            vp = vp / nv
            if abs(vp[2]) < 1e-12:
                d = np.full(len(lines), np.arctan2(vp[1], vp[0]))
            else:
                v = vp[:2] / vp[2]
                d = np.arctan2(v[1] - pts[:, 1], v[0] - pts[:, 0])
            r = np.abs((np.degrees(angs - d) + 90.0) % 180.0 - 90.0)
            sc = float(np.sum(np.minimum(r, 3.0)))
            if best is None or sc < best[0]:
                best = (sc, r)
    if best is None:
        return lines
    keep = [l for l, r in zip(lines, best[1]) if r <= VP_MAX]
    return keep if len(keep) >= 3 else lines


def _ridge_lines_raw(frame: np.ndarray, mask=None):
    s = px_scale(frame.shape[1])
    H, W = frame.shape[:2]
    m = paint_mask(frame) if mask is None else mask
    segs = cv2.HoughLinesP(m, 1, np.pi / 720, threshold=60,
                           minLineLength=int(round(MIN_SEG * s)),
                           maxLineGap=int(round(12 * s)))
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    L = np.hypot(dx, dy)
    ang = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0
    # Only a near-horizontal-ish family can be yard lines in this game's cameras.
    ok = np.abs(ang) < 70
    segs, dx, dy, L, ang = segs[ok], dx[ok], dy[ok], L[ok], ang[ok]
    if len(segs) == 0:
        return []
    hist, edges = np.histogram(ang, bins=180, range=(-90, 90), weights=L)
    i = int(np.argmax(hist))
    pk = 0.5 * (edges[i] + edges[i + 1])
    sel = np.abs((ang - pk + 90) % 180 - 90) <= ANG_WIN
    segs, dx, dy, L = segs[sel], dx[sel], dy[sel], L[sel]
    if len(segs) == 0:
        return []
    good = np.abs(dx) > 1e-6
    segs, dx, dy, L = segs[good], dx[good], dy[good], L[good]
    if len(segs) == 0:
        return []
    slope = dy / dx
    xc = W / 2.0
    yc = segs[:, 1] + slope * (xc - segs[:, 0])
    o = np.argsort(yc)
    yc, slope, L = yc[o], slope[o], L[o]
    groups, cur = [], [0]
    for k in range(1, len(yc)):
        if yc[k] - yc[cur[-1]] <= GROUP_PX * s:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    lines = []
    for g in groups:
        g = np.array(g)
        if L[g].sum() < 1.5 * MIN_SEG * s:      # a line needs real total support
            continue
        lines.append((float(np.average(yc[g], weights=L[g])),
                      float(np.average(slope[g], weights=L[g]))))
    return vp_filter(lines, W)


def coverage(frame: np.ndarray, lines, mask=None):
    """Fraction of each line's over-turf length that lies on paint (+-3 px)."""
    s = px_scale(frame.shape[1])
    H, W = frame.shape[:2]
    if mask is None:
        mask = paint_mask(frame)
    k = _odd(7 * s)
    md = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (k, k))) > 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = _odd(41 * s)
    ctx = cv2.blur((cv2.inRange(hsv, (30, 40, 25), (95, 255, 255)) > 0
                    ).astype(np.float32), (blur, blur)) > 0.55
    out = []
    step = max(1, int(round(3 * s)))
    for y, sl in lines:
        xs = np.arange(0, W, step, dtype=float)
        ys = y + sl * (xs - W / 2.0)
        ok = (ys >= 0) & (ys < H)
        xi, yi = xs[ok].astype(int), ys[ok].astype(int)
        if not len(xi):
            out.append(None)
            continue
        c = ctx[yi, xi]
        out.append(float(md[yi[c], xi[c]].mean()) if c.sum() >= 30 else None)
    return out


def stripe_lines(frame: np.ndarray):
    """[(y at frame centre, slope)] of painted stripes passing COVERAGE_MIN."""
    m = paint_mask(frame)
    R = _ridge_lines_raw(frame, mask=m)
    if not R:
        return []
    cov = coverage(frame, R, m)
    return [l for l, c in zip(R, cov) if c is not None and c >= COVERAGE_MIN]
