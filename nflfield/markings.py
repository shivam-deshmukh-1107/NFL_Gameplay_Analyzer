"""Stage 3 - from pixels to a pencil of yard lines.

PROVENANCE AND THE ONE REAL MERGE
---------------------------------
Shivam's `blitzfield/markings.py` is the base: shared colour preparation,
ridge -> thinning -> LSD -> collinear merge -> flank test -> vanishing point.
Yash's `absolute_anchor.turf_segments` does the same job in fewer steps.

**Both parent projects invented the flank test independently** - "a painted
line has turf on both sides of it; a player silhouette, a banner border and a
shadow edge do not" - and both describe it, in their own words, as the cheapest
high-value component in the system. Yash samples 9 points at +-7 px needing
0.55 turf; Shivam samples 11 points at +-5 px needing 0.6. Shivam's is kept
because it also classifies BOUNDARY (turf on ONE side), and the sidelines and
goal lines are the only source of the down-field direction - a naive
both-sides test rejects them and leaves that pencil empty.

What is taken from Yash is `drawn_marker_frac`, and it is not a duplicate of
Shivam's low-chroma gate. Two measurements make the case:

  * **The painted yard lines here are not white.** Measured at their core over
    the middle 40% of the frame: hue 87-93, S 63-140, V 96-186 - a pale cyan,
    BGR about (165, 165, 118). A detector tuned for white paint is working
    against the data, and a chroma gate is a soft ramp, not a decision.
  * **The game's own drawn lines must be sampled ACROSS, not along.** An LSD
    segment lies on the EDGE of a line, where the colour is blended with grass.
    On clip_04 f161 the scrimmage-line impostor measures hue 38-41 at its edge
    and hue 26 at its core: outside a 15-35 hue window at the point that was
    sampled, inside it two pixels away. Yash's earlier `drop_yellow` sampled
    the segment itself and also had no red band at all, so the first-down line
    - BGR (43, 36, 183) - was structurally invisible to it.

That one surviving impostor is what inflated the ladder on clip_04 f161 and let
the solver ride the exact scale degeneracy to a camera 148 yd in the air at
zero cost in residual. The coloured lines are still DETECTED, by
`fieldcolor.coloured_line_masks`, because the HUD states the exact yard gap
between them and that is the only absolute scale check either project has -
they just never reach the white-paint pencil.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G
from .config import det_scale
from .fieldcolor import TurfModel, coloured_line_masks, white_line_response

INTERIOR, BOUNDARY = 0, 1

#: Probability above which `fieldnet` output counts as turf / as paint.
#: A plain 0.5 is meaningful here in a way no top-hat threshold ever was: the
#: net is trained with a per-pixel loss, so its output is a calibrated
#: probability rather than a filter response whose scale drifts with the frame.
#: The shipped checkpoint carries its own swept values (0.625 / 0.925) and
#: `FieldSegmenter` reads them back, so a threshold is a property of a model
#: and never a constant here.
NET_TURF_THR = 0.5
NET_LINE_THR = 0.5

# --------------------------------------------------------------------------- #
# the game's own drawn lines - Yash's measurement, see the module docstring
# --------------------------------------------------------------------------- #
MARKER_YELLOW_H = (10, 50)       # line of scrimmage: hue 26-41 at the core
MARKER_YELLOW_S = 70
MARKER_YELLOW_V = 100
MARKER_RED_H = (160, 8)          # first-down line: hue 178/1. Wraps through 0.
MARKER_RED_S = 90
MARKER_RED_V = 90
MARKER_FRAC = 0.35               # of sampled points, before a segment is dropped
MARKER_PROBE_PX = (-4.0, -2.0, 0.0, 2.0, 4.0)   # ACROSS the line, not along it


def drawn_marker_frac(hsv: np.ndarray, px: np.ndarray,
                      py: np.ndarray) -> float:
    """Fraction of the sampled points that are a DRAWN coloured marker.

    i.e. the line of scrimmage or the first-down line, rather than painted
    field marking. See the module docstring for the measured hue bands.
    """
    h, w = hsv.shape[:2]
    qx = np.clip(np.round(px).astype(int), 0, w - 1)
    qy = np.clip(np.round(py).astype(int), 0, h - 1)
    H = hsv[qy, qx, 0].astype(np.int16)
    S = hsv[qy, qx, 1].astype(np.int16)
    V = hsv[qy, qx, 2].astype(np.int16)
    yellow = ((H >= MARKER_YELLOW_H[0]) & (H <= MARKER_YELLOW_H[1]) &
              (S >= MARKER_YELLOW_S) & (V >= MARKER_YELLOW_V))
    red = (((H >= MARKER_RED_H[0]) | (H <= MARKER_RED_H[1])) &
           (S >= MARKER_RED_S) & (V >= MARKER_RED_V))
    return float((yellow | red).mean())


def reject_drawn_markers(segs: np.ndarray, frame_bgr: np.ndarray,
                         hsv: np.ndarray | None = None) -> np.ndarray:
    """Boolean keep-mask: drop segments that are the game's own drawn lines.

    The first-down line survives the flank test on its own merits - it is drawn
    on the grass, is straight, and spans the field - but it sits at a position
    that is not a multiple of 5 yards, so it is a guaranteed outlier in the
    5-yard ladder.

    MEASURED IN THIS PIPELINE 2026-10-17: **it rejects nothing, and that is the
    correct outcome, not a broken test.** On clips 1-4 it fired on 0 segments
    of every sampled frame. The reason was checked rather than assumed: the
    drawn lines are plainly present (22k-111k px per frame in
    `coloured_line_masks`), but only 1.6-1.7% of those pixels survive into
    `prep.mask`, because `white_line_response`'s low-chroma gate suppresses
    them BEFORE thinning and LSD. Yash needed this test because his detector
    ran LSD on raw greyscale with no chroma gate at all.

    Kept anyway, as a guard rather than a stage: the chroma gate is a soft ramp
    (full weight to chroma 26, zero by 52), so a strongly lit drawn line can in
    principle leak, and this is the only thing downstream of it that would
    notice. If it ever starts firing in volume, the chroma gate has moved and
    that is worth knowing.
    """
    if len(segs) == 0:
        return np.zeros(0, bool)
    if hsv is None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    t = np.linspace(0.05, 0.95, 9)
    keep = np.ones(len(segs), bool)
    for i, (x1, y1, x2, y2) in enumerate(segs):
        dx, dy = x2 - x1, y2 - y1
        n = float(np.hypot(dx, dy))
        if n < 1e-6:
            keep[i] = False
            continue
        nx, ny = -dy / n, dx / n
        px, py = x1 + t * dx, y1 + t * dy
        worst = max(drawn_marker_frac(hsv, px + k * nx, py + k * ny)
                    for k in MARKER_PROBE_PX)
        keep[i] = worst <= MARKER_FRAC
    return keep


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cc_filter(mask: np.ndarray, min_area: int = 0, max_area: int | None = None,
              max_dim: int | None = None) -> np.ndarray:
    """Keep connected components matching size constraints.

    Vectorised via fancy-indexing the label image with a per-label keep table;
    the obvious per-component Python loop is thousands of iterations on a
    speckled mask and dominated the whole frame budget.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.ones(n, bool)
    keep[0] = False                                   # background
    a = stats[:, cv2.CC_STAT_AREA]
    if min_area:
        keep &= a >= min_area
    if max_area is not None:
        keep &= a <= max_area
    if max_dim is not None:
        keep &= np.maximum(stats[:, cv2.CC_STAT_WIDTH],
                           stats[:, cv2.CC_STAT_HEIGHT]) <= max_dim
    return keep[lbl]


def _odd(n: float, lo: int = 3) -> int:
    """Nearest odd integer >= lo. Morphology kernels must be odd-sized."""
    k = int(round(n))
    if k % 2 == 0:
        k += 1
    return max(k, lo)


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold of a float array in [0, 1], returned in the same units."""
    if values.size < 64:
        return 0.25
    v = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


# --------------------------------------------------------------------------- #
# shared per-frame preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class FramePrep:
    """All the colour-domain work for one frame, computed once.

    Built once per frame and passed everywhere. With a net in play the prep is
    a forward pass rather than a handful of array ops, and Shivam measured the
    dataset builder constructing one twice per frame - the bulk of the cost of
    a nine-frame voting window.
    """

    frame: np.ndarray
    turf_mask: np.ndarray
    region: np.ndarray
    occluder: np.ndarray
    resp: np.ndarray
    mask: np.ndarray
    hud: np.ndarray
    region_closed: np.ndarray | None = None  # region + drawn lines; boundary only
    tick_mask: np.ndarray | None = None      # thin, colour-derived; hash marks only
    off_turf: np.ndarray | None = None       # Mahalanobis > 3; shared
    colour_masks: dict | None = None         # drawn-line hues, keyed by colour
    hsv: np.ndarray | None = None            # shared with the marker rejection
    det_s: float = 1.0                       # config.det_scale(picture width)

    @staticmethod
    def build(frame: np.ndarray, turf: TurfModel, hud: np.ndarray | None = None,
              k: float = 3.0, open_ksize: int = 11, min_area: int = 150,
              dilate: int = 5, thr_lo: float = 0.15, thr_hi: float = 0.60,
              speckle_area: int = 22, net=None) -> "FramePrep":
        """Colour-domain preprocessing, optionally with a learned front end.

        `net` is a `fieldnet.FieldSegmenter`. When given it supplies the turf
        mask and the line response in place of the Mahalanobis cut and the
        top-hat; with `net=None` this is exactly the colour pipeline, which
        stays the fallback whenever no weights are trained.

        **A new capture gets the colour path until it has its own rows.** This
        is a standing rule, not a preference: a 0.48 M-parameter net learns
        "turf" as a COLOUR, and three separate captures have broken it - not
        degraded, collapsed - at 3-5% and then 4-10% P(turf) against the colour
        model's 62-88%.

        The occluder mask is deliberately NOT taken from the net. Players are
        marked `ignore` when the dataset is built - a moving body is turf in
        some frames of the voting window and not in others - so the net has no
        reason to carve them out and is free to predict the field straight
        through them. That is the right answer for the tint and the wrong one
        for "where must the line search not look", so that question keeps its
        own answer from the colour mask.
        """
        # Every pixel constant below was measured at SHIVAM_REF_W; lengths
        # scale by `s`, areas by `s * s`. See config.det_scale.
        s = det_scale(frame.shape[1])
        open_ksize = _odd(open_ksize * s)
        min_area = int(round(min_area * s * s))
        dilate = max(1, int(round(dilate * s)))
        speckle_area = int(round(speckle_area * s * s))
        ridge_ksize = _odd(11 * s)
        close_k, open_k = _odd(25 * s), _odd(7 * s)

        if hud is None:
            hud = np.zeros(frame.shape[:2], bool)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maha = turf.mahalanobis(frame)
        tm_colour = (maha < k) & ~hud
        off_turf = maha > 3.0
        p_line = None
        if net is not None:
            from .fieldcolor import normalise_to_canon
            # tm_colour, not the raw Mahalanobis cut: it already excludes the
            # HUD, and the HUD's luminance has no business in a median over
            # turf pixels.
            src = (normalise_to_canon(frame, turf, tm_colour)
                   if getattr(net, "normalise", False) else frame)
            p_turf, p_line = net.predict(src)
            tm = (p_turf >= getattr(net, "turf_thr", NET_TURF_THR)) & ~hud
        else:
            tm = tm_colour

        def _close(mask_bool):
            mm = mask_bool.astype(np.uint8)
            mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_k, close_k)))
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (open_k, open_k)))
            return cc_filter(mm, min_area=int(mm.size * 0.02))

        # The plain turf region. Kept free of the game-drawn colour: anything
        # consuming `region` as "where the field is" - notably the
        # scrimmage/first-down line fitter - degrades badly if transient banner
        # colour is allowed to widen it.
        region = _close(tm)

        # A second, extended region used ONLY for boundary tracing. The
        # coloured scrimmage line reads as "not turf" and carves a channel
        # across the field, which then surfaces as a cross-field region edge
        # and is mistaken for a goal line. Only line-like coloured components
        # are folded in: taking every saturated pixel also swallows the light
        # out-of-bounds apron and erases the sideline.
        colour_masks = coloured_line_masks(frame, turf, off_turf=off_turf)
        painted = np.zeros(tm.shape, bool)
        for _name, cmask in colour_masks.items():
            nn, ll, stt, _ = cv2.connectedComponentsWithStats(
                cmask.astype(np.uint8), 8)
            a_ = stt[:, cv2.CC_STAT_AREA]
            w_ = stt[:, cv2.CC_STAT_WIDTH]
            h_ = stt[:, cv2.CC_STAT_HEIGHT]
            elong = np.maximum(w_, h_) / np.maximum(np.minimum(w_, h_), 1)
            keep = (a_ >= 300 * s * s) & (elong >= 4.0)
            keep[0] = False
            painted |= keep[ll]
        region_closed = _close(tm | (painted & ~hud))

        # Occluders: large off-turf blobs. The opening is wider than any painted
        # line, so markings drop out of this mask while solid objects survive.
        solid = cv2.morphologyEx((~tm_colour).astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(
                                     cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)))
        occ = cc_filter(solid, min_area=min_area).astype(np.uint8)
        if dilate:
            occ = cv2.dilate(occ, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))
        occ = occ.astype(bool)

        valid = region & ~occ & ~hud

        # The colour ridge mask, ALWAYS - even with a net, because the tick
        # mask has to come from it. Adaptive threshold: turf has a mottled
        # texture whose top-hat response varies frame to frame. A fixed cut
        # either floods the mask with speckle - which makes LSD's density test
        # reject every rectangle and silently return ZERO segments - or misses
        # faint distant lines.
        resp_c = white_line_response(frame, ksize=ridge_ksize, valid=valid) * valid
        thr = float(np.clip(_otsu(resp_c[valid]) if valid.any() else 0.25,
                            thr_lo, thr_hi))
        tick_mask = cc_filter((resp_c >= thr).astype(np.uint8),
                              min_area=speckle_area).astype(np.uint8)

        if p_line is not None:
            resp = p_line * valid
            mask = cc_filter(
                (resp >= getattr(net, "line_thr", NET_LINE_THR)).astype(np.uint8),
                min_area=speckle_area).astype(np.uint8)
        else:
            resp, mask = resp_c, tick_mask
        return FramePrep(frame, tm, region, occ, resp, mask, hud,
                         region_closed, tick_mask, off_turf, colour_masks, hsv,
                         det_s=s)


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def detect_segments(mask: np.ndarray, min_len: float = 18.0,
                    s: float = 1.0) -> np.ndarray:
    """Thin the mask to centrelines and run LSD on the skeleton.

    Thinning first, so LSD sees ONE line per painted stripe rather than its two
    edges. That matters more than it sounds: an LSD segment on an edge sits
    where the colour is blended with grass, which is the measurement that broke
    Yash's first drawn-marker test.
    """
    thin = cv2.ximgproc.thinning(mask * 255,
                                 thinningType=cv2.ximgproc.THINNING_GUOHALL)
    img = cv2.dilate(thin, np.ones((2, 2), np.uint8))
    # REFINE_STD, NOT REFINE_ADV. On a thinned skeleton the advanced NFA
    # refinement rejects almost every genuine line (0 vs 47 long segments on a
    # sample frame) and runs ~2000x slower (64 s vs 0.03 s per frame).
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img)[0]
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4), np.float64)
    segs = np.asarray(lines, np.float64).reshape(-1, 4)
    return segs[G.seg_length(segs) >= min_len * s]


def merge_collinear(segs: np.ndarray, angle_tol_deg: float = 2.0,
                    perp_tol: float = 4.0, gap_tol: float = 80.0,
                    s: float = 1.0) -> np.ndarray:
    """Fuse fragments of the same painted line (players break lines in two).

    `gap_tol` is deliberately finite and this runs BEFORE the vanishing point
    is known, so a yard line crossed by three players can break further apart
    than 80 px and stay broken.

    TRIED AND REVERTED: merging instead by shared transversal crossing once the
    pencil is known. It recovered NO extra lines - it only consolidated
    duplicates of lines already found, clip_04's pencil going 36 -> 24 segments
    with the tracked count unchanged at 15 - and a fixed 6 px tolerance fuses
    genuinely distinct yard lines where they crowd near the horizon, which
    broke clip_04's yard scale to 75.4 yd against a HUD still reading 29. A
    scale-aware tolerance could be made safe, but with no measured recall
    benefit there is nothing to buy.
    """
    if len(segs) == 0:
        return segs
    perp_tol, gap_tol = perp_tol * s, gap_tol * s
    segs = segs[np.argsort(-G.seg_length(segs))]
    n = len(segs)
    used = np.zeros(n, bool)
    # Hoisted out of the O(n^2) pair loop. LSD returns a few hundred segments
    # per frame, so recomputing a direction and a length per PAIR was the
    # single largest cost in a render.
    D = G.seg_direction(segs)
    LN = G.seg_to_line(segs)
    LEN = G.seg_length(segs)
    PTS = segs.reshape(n, 2, 2)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    out = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        di = D[i]
        # Both cheap rejections for every remaining candidate at once; only the
        # survivors reach the gap test, which has to stay sequential because it
        # depends on the group built so far.
        free = np.flatnonzero(~used[i + 1:]) + i + 1
        if len(free):
            aligned = np.abs(D[free] @ di) >= cos_tol
            free = free[aligned]
        if len(free):
            perp = np.abs(PTS[free] @ LN[i, :2] + LN[i, 2]).max(axis=1)
            free = free[perp <= perp_tol]

        ti = PTS[i] @ di
        lo_k, hi_k = int(np.argmin(ti)), int(np.argmax(ti))
        t_lo, t_hi = ti[lo_k], ti[hi_k]
        p_lo, p_hi = PTS[i][lo_k], PTS[i][hi_k]
        covered = float(LEN[i])
        for j in free:
            tj = PTS[j] @ di
            j_lo, j_hi = int(np.argmin(tj)), int(np.argmax(tj))
            n_lo, n_hi = min(t_lo, tj[j_lo]), max(t_hi, tj[j_hi])
            if (n_hi - n_lo) - (covered + float(LEN[j])) > gap_tol:
                continue
            if tj[j_lo] < t_lo:
                t_lo, p_lo = tj[j_lo], PTS[j][j_lo]
            if tj[j_hi] > t_hi:
                t_hi, p_hi = tj[j_hi], PTS[j][j_hi]
            covered += float(LEN[j])
            used[j] = True
        out.append([p_lo[0], p_lo[1], p_hi[0], p_hi[1]])
    return np.asarray(out, np.float64)


def classify_flanks(segs: np.ndarray, prep: FramePrep, offset: float = 5.0,
                    n_samples: int = 11, min_frac: float = 0.6,
                    boundary_min_len: float = 50.0) -> np.ndarray:
    """Label each segment INTERIOR / BOUNDARY, or -1 to reject.

    **THE FLANK TEST**, invented independently in both parent projects and
    described by both as the cheapest high-value component in the system.
    Painted markings are turf-flanked by construction, so requiring turf on
    both sides removes almost all structural false positives with no temporal
    model at all.

    Sidelines and goal lines are the deliberate exception: turf on one side,
    the apron or end zone on the other. They are the main source of the
    down-field vanishing point, so rejecting them - as a naive both-sides test
    does - is what leaves the second pencil empty. Yash's version has no
    BOUNDARY class and that is the one thing it gives up.
    """
    n = len(segs)
    if n == 0:
        return np.zeros(0, np.int8)
    h, w = prep.frame.shape[:2]
    tm = prep.turf_mask
    lengths = G.seg_length(segs)
    out = np.full(n, -1, np.int8)
    offset, boundary_min_len = offset * prep.det_s, boundary_min_len * prep.det_s

    for i, s in enumerate(segs):
        p0, p1 = s[:2], s[2:]
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        nrm = np.array([-d[1], d[0]])
        ts = np.linspace(0.15, 0.85, n_samples)[:, None]
        base = p0 + ts * (p1 - p0)
        frac = []
        for sgn in (+1.0, -1.0):
            q = np.round(base + sgn * offset * nrm).astype(int)
            inb = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            frac.append(float(tm[q[inb, 1], q[inb, 0]].mean()) if inb.sum() >= 3
                        else 0.0)
        lo, hi = min(frac), max(frac)
        if lo >= min_frac:
            out[i] = INTERIOR
        elif hi >= min_frac and lo <= 1.0 - min_frac and lengths[i] >= boundary_min_len:
            out[i] = BOUNDARY
    return out


# --------------------------------------------------------------------------- #
# field boundary (sidelines, goal lines)
# --------------------------------------------------------------------------- #
def boundary_segments(prep: FramePrep, min_len: float = 60.0,
                      border_margin: int = 6, eps: float = 3.0,
                      axis_tol_deg: float = 2.0) -> np.ndarray:
    """Straight edges of the field region - the sidelines and goal lines.

    Blitz does not draw the sideline as a thin white stripe: the entire
    out-of-bounds apron is light coloured, so the field edge is a REGION
    BOUNDARY, not a ridge. A top-hat ridge filter is blind to it, which is why
    the down-field pencil stays empty if only painted lines are sought.

    Edges on the image border, on the HUD, or within `axis_tol_deg` of exactly
    horizontal/vertical are discarded - transient HUD banners are axis-aligned
    rectangles and field lines almost never are.
    """
    src = prep.region_closed if prep.region_closed is not None else prep.region
    m = src.astype(np.uint8)
    h, w = m.shape
    s = prep.det_s
    min_len, eps = min_len * s, eps * s
    border_margin = max(1, int(round(border_margin * s)))
    hud_k = _odd(11 * s)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hud_d = cv2.dilate(prep.hud.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hud_k, hud_k)))
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 0.02 * m.size:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        for i in range(len(ap)):
            p0, p1 = ap[i], ap[(i + 1) % len(ap)]
            d = p1 - p0
            L = float(np.hypot(*d))
            if L < min_len:
                continue
            ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 90.0
            if min(ang, 90.0 - ang) < axis_tol_deg:
                continue
            ts = np.linspace(0.0, 1.0, 15)[:, None]
            P = p0 + ts * d
            near_border = ((P[:, 0] < border_margin) | (P[:, 0] > w - 1 - border_margin)
                           | (P[:, 1] < border_margin) | (P[:, 1] > h - 1 - border_margin))
            if near_border.mean() > 0.3:
                continue
            q = np.round(P).astype(int)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            if hud_d[q[:, 1], q[:, 0]].mean() > 0.2:
                continue
            out.append([p0[0], p0[1], p1[0], p1[1]])
    return np.asarray(out, np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# coverage - is this segment actually ON paint along its whole length?
# --------------------------------------------------------------------------- #
#: Minimum share of a segment's USABLE length that must sit on the line mask.
#:
#: MEASURED MOTIVATION (docs/LESSONS, "clip_03's 13 residual refusals"). On the
#: 13 clip_03 frames the camera refused on residual, the independently written
#: stripe detector fitted a camera to 0.19-2.99 px - inside the 4.0 px gate -
#: where our pencil failed at 6-22 px, and our ladder residual ran 3-10x worse
#: than its on identical frames (f42 8.62 vs 1.16, f96 15.03 vs 2.62, f166 8.68
#: vs 0.65). The difference that stood out is that the referee keeps a line
#: only if at least half its over-turf length is really paint, and our pencil
#: had only a LENGTH cut - so a long segment that is mostly off paint survived.
#:
#: WHY THIS IS NOT A COPY OF THE REFEREE. `indep_paint` measures coverage
#: against its own wide top-hat mask, dilated, sampled every 3 px across the
#: full frame width. This measures against `prep.mask` - the thinned,
#: Otsu-thresholded line mask the pencil was actually built from - over the
#: segment's own extent, and it excludes occluded and off-field samples from
#: the denominator rather than the numerator. Different mask, different
#: sampling, different denominator. Keeping them independent is the point: the
#: referee must not end up grading its own code.
COVERAGE_MIN = None
"""OFF BY DEFAULT pending an end-to-end run. Pass coverage_min=0.35 to enable.

LOCAL RESULT on the 13 clip_03 target frames: 4 better, 1 WORSE (f96 20.75
against 15.03), and f42 - the frame that motivated the whole investigation -
unchanged at 13 lines / 8.62 rms despite 59 segments dropped. Not a clean win,
and the precedent is pointed: the split-edge merge looked better locally (6 of
13 improved) and cost 47 certified frames on clip_02 end to end. Enabling this
requires that full run.

Minimum share of a candidate line's ON-FIELD length that must be on paint.

**MEASURED, AND THIS IS THE FIX THAT WORKS.** On clip_03 f42, probing each
candidate across the frame width:

    confirmed painted line y=457   0.88 of its length on the line mask
    our false line        y=447   0.03

A real yard line crosses the visible field; a spurious detection on turf
texture does not, however convincing the short segment that produced it looks.

MY FIRST VERSION OF THIS TEST WAS VACUOUS, and the reason is the whole point.
It sampled along the SEGMENT'S OWN EXTENT, where a short spurious segment
scores 1.0 by construction - it dropped 0 of 13 target frames, with coverage
p10 1.00. `indep_paint.coverage` works because it samples across the WHOLE
FRAME WIDTH. Same idea, different denominator, opposite outcome.

Independence from the referee is preserved: it measures Hough lines against its
own 25 px top-hat with its own dilation, this measures our candidates against
`prep.mask` - the thresholded ridge the pencil was built from - restricted to
the field region. Different mask, different lines, different restriction.

The threshold is deliberately well below the 0.88 a good line scores: a yard
line genuinely occluded by players, or running off the frame edge, legitimately
covers less, and the point is to reject 0.03 rather than to demand 0.9.
"""

#: Dilation of the line mask before sampling, px at SHIVAM_REF_W. The mask is
#: a thresholded ridge response, so a fitted line may sit a pixel or two to the
#: side of its peak without being wrong.
COVERAGE_TOL_PX = 3.0
COVERAGE_MIN_SAMPLES = 12


def line_coverage(segs: np.ndarray, prep: FramePrep,
                  step_px: float = 3.0) -> np.ndarray:
    """Share of each line's on-field length that lies on the line mask.

    Each segment is extended to a full LINE and sampled across the frame width,
    not merely over its own extent - see `COVERAGE_MIN`. Samples outside the
    field region are excluded from the denominator, so a line that leaves the
    visible field is judged only where it could have been seen.

    Returns NaN where too few samples were usable. A caller must treat NaN as
    "no opinion": refusing an unjudgeable line would silently drop lines in
    heavy traffic, the opposite of what this is for.
    """
    n = len(segs)
    if n == 0:
        return np.zeros(0, float)
    s = prep.det_s
    h, w = prep.frame.shape[:2]
    xc = w / 2.0

    k = _odd(COVERAGE_TOL_PX * 2.0 * s)
    paint = cv2.dilate(prep.mask.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (k, k))) > 0
    onfield = prep.region & ~prep.hud

    xs = np.arange(0.0, float(w), max(1.0, step_px * s))
    out = np.full(n, np.nan)
    for i, seg in enumerate(segs):
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        if abs(dx) < 1e-6:
            continue
        sl = (y2 - y1) / dx
        y_at = y1 + sl * (xc - x1)
        ys = y_at + sl * (xs - xc)
        q = np.round(ys).astype(int)
        inb = (q >= 0) & (q < h)
        if inb.sum() < COVERAGE_MIN_SAMPLES:
            continue
        qy, qx = q[inb], np.round(xs[inb]).astype(int)
        use = onfield[qy, qx]
        if use.sum() < COVERAGE_MIN_SAMPLES:
            continue
        out[i] = float(paint[qy[use], qx[use]].mean())
    return out


# --------------------------------------------------------------------------- #
# split edges - is the span BETWEEN two close lines paint, or turf?
# --------------------------------------------------------------------------- #
#: Widest separation, in px at SHIVAM_REF_W, at which two lines can still be
#: the two edges of ONE painted stripe.
#:
#: MEASURED: painted lines here are ~8 px wide at the top of the picture and
#: 15-20 px at the bottom (Yash). Observed split pairs sit 10-13 px apart - his
#: clip_04 f190 at y=884.5/897.4, ours on clip_03 at 447/457, 412/423 and
#: 496/507. The cap sits above the widest stripe and far below any real 5-yard
#: gap, which is ~100-185 px at the depths where stripes are that wide.
SPLIT_MAX_SEP_PX = 26.0

#: Two edges of one stripe are parallel to within detection noise.
SPLIT_ANGLE_TOL_DEG = 2.0

#: Share of the midline between a pair that must look like PAINT rather than
#: turf before they are fused.
#:
#: WHY A PHYSICAL TEST AND NOT A GAP-SIZE TEST. A gap-ratio test was ported
#: from Yash, measured NET NEGATIVE here and reverted - clip_02 lost 47
#: certified frames - and a median-gap variant was then shown not to help:
#: with gaps of 100, 15, 100 the median IS 100, so it fires on a real adjacent
#: pair whose neighbours are merely missing. **From gap sizes alone a real pair
#: and a split edge are indistinguishable**, so no tuning of a ratio separates
#: them. See docs/LESSONS.md.
#:
#: The physical difference is unambiguous: a split edge is ONE stripe, so the
#: span between its halves is paint; a real adjacent pair has turf between
#: them. This is the flank test turned inward - `classify_flanks` asks whether
#: there is turf on BOTH SIDES, this asks whether there is turf BETWEEN.
SPLIT_PAINT_MIN = 0.70
"""**OFF BY DEFAULT: the premise is DISPROVED.** `fuse_edges` defaults False.

The tight pairs on clip_03 are NOT the two edges of one stripe. Measured on
f42, probing across the frame at each y:

    confirmed painted line y=457   L 166.0   chroma 17.0   in line mask 0.88
    our extra line        y=447   L  89.5   chroma 19.0   in line mask 0.03
    between them          y=452   L  94.5   chroma 18.4   in line mask 0.02
    plain turf                     L  86-100 chroma 18-19  in line mask 0.00-0.03

y=447 has TURF luminance and is essentially absent from the line mask: it is a
false line on bare turf, not a second edge of the stripe at 457. The remedy is
to REJECT it, which `COVERAGE_MIN` now does - not to fuse it.

The `paint_like` predicate below was also measurably wrong, and that is the
more useful finding: **chroma does not discriminate paint from turf in this
game.** Paint reads chroma 17 against turf's 18.6-19.4, and Blitz's pale-cyan
paint has nearly turf CHROMATICITY (Mahalanobis 1.48 against turf's 1.21-1.38),
so `~turf_mask` fires only 0.10 on a real painted line. The 26.0 chroma value
was borrowed from `white_line_response`, where it is a SOFT RAMP multiplying an
already-isolating top-hat rather than a hard gate paint must pass. Luminance is
the discriminating channel here (166 against 86-100), not chroma.

Kept wired in and inert, with the measurement, in case a genuinely split stripe
turns up on footage with thicker paint.
"""

#: Chroma below which a pixel is near-neutral, i.e. paint-coloured rather than
#: turf- or jersey-coloured. The same value `white_line_response` gates on, for
#: the same reason: Blitz paint is near-neutral (a, b ~ 128) while turf sits at
#: (110, 126) and the game's drawn markers are saturated.
SPLIT_CHROMA_MAX = 26.0


def paint_like(frame_bgr, turf_mask):
    """Pixels that look like painted marking: not turf, and near-neutral.

    Deliberately NOT `prep.mask`. That is the thresholded top-hat ridge
    response, and a stripe WIDER than the top-hat kernel responds only at its
    two EDGES - which is exactly why split edges exist. A test for "is the
    stripe interior in the line mask" would therefore answer no for precisely
    the stripes it needs to catch.
    """
    from .fieldcolor import chroma
    return (~turf_mask) & (chroma(frame_bgr) <= SPLIT_CHROMA_MAX)


def _line_at(seg, xc):
    """(y at column xc, slope) for a segment, or None if vertical."""
    x1, y1, x2, y2 = seg
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return None
    sl = (y2 - y1) / dx
    return float(y1 + sl * (xc - x1)), float(sl)


def fuse_split_edges(segs, prep, max_sep=SPLIT_MAX_SEP_PX,
                     angle_tol=SPLIT_ANGLE_TOL_DEG,
                     paint_min=SPLIT_PAINT_MIN, n_samples=25):
    """Fuse pairs of segments that are the two edges of one painted stripe.

    Returns `(segs, n_fused)`. A pair is fused only when all three hold:
      * their centre-column separation is at most `max_sep` - an ABSOLUTE
        bound, which is what the reverted gap-ratio test lacked;
      * they are parallel to within `angle_tol`;
      * at least `paint_min` of the midline between them is paint-like.

    The third condition does the work, and it is why a real adjacent pair near
    the horizon survives: two genuine yard lines have turf between them however
    close together perspective puts them.
    """
    n = len(segs)
    if n < 2:
        return segs, 0
    h, w = prep.frame.shape[:2]
    xc = w / 2.0
    s = prep.det_s
    max_sep = max_sep * s

    pl = paint_like(prep.frame, prep.turf_mask)
    lengths = G.seg_length(segs)

    desc = []
    for i, seg in enumerate(segs):
        la = _line_at(seg, xc)
        if la is not None:
            desc.append((la[0], la[1], i))
    order = sorted(desc, key=lambda d: d[0])

    fused_into, used, n_fused = {}, set(), 0
    t = np.linspace(0.12, 0.88, n_samples)
    for k in range(len(order) - 1):
        ya, sla, ia = order[k]
        yb, slb, ib = order[k + 1]
        if ia in used or ib in used:
            continue
        if (yb - ya) > max_sep:
            continue
        if abs(np.degrees(np.arctan(sla) - np.arctan(slb))) > angle_tol:
            continue
        xs_lo = max(min(segs[ia][0], segs[ia][2]),
                    min(segs[ib][0], segs[ib][2]))
        xs_hi = min(max(segs[ia][0], segs[ia][2]),
                    max(segs[ib][0], segs[ib][2]))
        if xs_hi - xs_lo < 20.0 * s:
            continue
        xs = xs_lo + t * (xs_hi - xs_lo)
        ymid = 0.5 * ((ya + sla * (xs - xc)) + (yb + slb * (xs - xc)))
        qx = np.round(xs).astype(int)
        qy = np.round(ymid).astype(int)
        inb = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
        if inb.sum() < 8:
            continue
        if float(pl[qy[inb], qx[inb]].mean()) < paint_min:
            continue
        wa, wb = float(lengths[ia]), float(lengths[ib])
        y_new = (ya * wa + yb * wb) / (wa + wb)
        sl_new = (sla * wa + slb * wb) / (wa + wb)
        x_lo = min(min(segs[ia][0], segs[ia][2]), min(segs[ib][0], segs[ib][2]))
        x_hi = max(max(segs[ia][0], segs[ia][2]), max(segs[ib][0], segs[ib][2]))
        fused_into[ia] = [x_lo, y_new + sl_new * (x_lo - xc),
                          x_hi, y_new + sl_new * (x_hi - xc)]
        used.add(ia)
        used.add(ib)
        n_fused += 1

    out = []
    for i in range(n):
        if i in fused_into:
            out.append(fused_into[i])
        elif i in used:
            continue
        else:
            out.append([float(x) for x in segs[i]])
    return np.asarray(out, np.float64).reshape(-1, 4), n_fused


# --------------------------------------------------------------------------- #
# per-frame driver
# --------------------------------------------------------------------------- #
@dataclass
class FrameMarkings:
    """What one frame yields: the yard-line pencil and the field boundary."""

    frame_index: int
    segs: np.ndarray                 # (N,4) surviving segments
    label: np.ndarray                # 0 = yard-line pencil, -1 = other
    flank: np.ndarray = field(       # INTERIOR / BOUNDARY per surviving segment
        default_factory=lambda: np.zeros(0, np.int8))
    boundary: np.ndarray = field(    # (K,4) field-region edges
        default_factory=lambda: np.zeros((0, 4)))
    vp_cross: object = None          # vanishing point of the yard-line pencil
    n_raw: int = 0                   # LSD segments before merging
    n_drawn_rejected: int = 0        # dropped as the game's own coloured lines
    coverage: np.ndarray = field(    # share of each survivor's length on paint
        default_factory=lambda: np.zeros(0, float))
    n_low_coverage: int = 0          # dropped for sitting mostly off paint
    n_fused_edges: int = 0           # pairs fused as two edges of one stripe

    @property
    def pencil(self) -> np.ndarray:
        """The yard-line pencil: segments RANSAC accepted into the family."""
        return self.segs[self.label == 0] if len(self.segs) else self.segs


def analyse_frame(frame: np.ndarray, turf: TurfModel, hud=None,
                  frame_index: int = -1, rng=None, net=None,
                  prep: FramePrep | None = None,
                  coverage_min: float | None = COVERAGE_MIN,
                  fuse_edges: bool = False) -> FrameMarkings:
    """Extract the yard-line pencil and the field boundary from one frame.

    Only the CROSS-FIELD pencil is recovered here. A second pencil for the
    down-field direction was fitted in both parent projects on the way to a
    focal length and a metric width; that is not recoverable from this footage
    (see geometry.py), and forcing the hash columns through a tracked
    down-field vanishing point was measured to make them WORSE - ticks-on-the-
    line went 8/10/14/13 to 0/1/13/9 across the four clips, because a few
    degrees of vanishing-point error rotates a column off the marks it was
    fitted to. The locally fitted columns were already right.
    """
    rng = rng or np.random.default_rng(0)
    if prep is None:
        prep = FramePrep.build(frame, turf, hud=hud, net=net)

    raw = detect_segments(prep.mask, s=prep.det_s)
    segs = merge_collinear(raw, s=prep.det_s)

    # The game's own drawn lines, before the flank test: they pass it on their
    # own merits, and they are a guaranteed outlier in a 5-yard ladder.
    n_drawn = 0
    if len(segs):
        keep_drawn = reject_drawn_markers(segs, frame, prep.hsv)
        n_drawn = int((~keep_drawn).sum())
        segs = segs[keep_drawn]

    flank = classify_flanks(segs, prep) if len(segs) else np.zeros(0, np.int8)
    keep = flank >= 0
    segs, flank = segs[keep], flank[keep]

    # Split edges: two edges of one painted stripe arriving as two lines.
    # The test is physical - is there turf BETWEEN them? - not statistical.
    n_fused = 0
    if fuse_edges and len(segs) >= 2:
        segs, n_fused = fuse_split_edges(segs, prep)
        if n_fused:
            # Recompute flanks for the fused set rather than mapping the
            # old labels across a merge: cheap, and cannot go stale.
            flank = classify_flanks(segs, prep)
            keep2 = flank >= 0
            segs, flank = segs[keep2], flank[keep2]

    # Coverage: a long segment that is mostly NOT on paint is not a yard line.
    # NaN means unjudgeable and is kept - see `segment_coverage`.
    n_low_cov = 0
    cov = np.zeros(0, float)
    if coverage_min is not None and len(segs):
        cov = line_coverage(segs, prep)
        ok = ~(np.isfinite(cov) & (cov < coverage_min))
        n_low_cov = int((~ok).sum())
        segs, flank, cov = segs[ok], flank[ok], cov[ok]

    label = np.full(len(segs), -1, np.int8)
    v1 = None
    if len(segs) >= 3:
        v1, in1, _ = G.ransac_vp(segs, tau_deg=1.5, iters=700, rng=rng)
        if v1 is not None:
            label[in1] = 0

    bnd = boundary_segments(prep)
    return FrameMarkings(frame_index=frame_index, segs=segs, label=label,
                         flank=flank, boundary=bnd, vp_cross=v1,
                         n_raw=len(raw), n_drawn_rejected=n_drawn,
                         coverage=cov, n_low_coverage=n_low_cov,
                         n_fused_edges=n_fused)


# --------------------------------------------------------------------------- #
# paint direction - the tracker's roll observation
# --------------------------------------------------------------------------- #
#: How far from the measured dominant angle a segment may sit and still count
#: as part of the same painted family.
#:
#: WAS 5.0 in Yash's project, raised to 20.0 on 2026-09-24. Perspective FANS
#: projected yard lines - `camera.MAX_YARD_ANGLE_SPREAD_DEG` already allows 25
#: deg of spread for exactly that reason - so in a low, close, oblique view the
#: real lines span more than 5 deg and the tighter window discards them.
#: Measured on his clip_06, frames 100-1547 every 25th: ladder-pass rate 15.5%
#: at 5.0 against 22.4% at 20.0, with the median line count going 3 -> 4.
#:
#: **THIS IS NOT AN ANGLE PRIOR.** The banned `expected_angle_deg` fed a
#: POSE-derived angle into the detector and created a positive feedback loop
#: (5 lines at 0-4 deg of pose error, 0 lines at 6+). `dominant` here is
#: measured from the surviving segments themselves, so nothing about the pose
#: reaches the detector and a drifted pose still cannot blind it.
PAINT_ANGLE_TOL_DEG = 20.0


def paint_direction(segs: np.ndarray, angtol: float = PAINT_ANGLE_TOL_DEG):
    """`(dominant_angle_deg, support)` for a pencil of painted segments.

    `support` is `(midpoints Nx2, lengths N)`: exactly the segments that entered
    the length-weighted dominant angle, with exactly those weights.
    `camera.paint_roll_disagreement` needs it to compare the model with the
    paint AT THE SAME PLACES - comparing one model line against a fanned
    average refused correct poses and could not see a real 5 deg roll error in
    the wrong direction.

    Returns `(None, None)` when there is no measurable family.
    """
    segs = np.asarray(segs, float)
    if len(segs) == 0:
        return None, None
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    L = np.hypot(dx, dy)
    if L.sum() <= 0:
        return None, None
    ang = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0
    hist, edges = np.histogram(ang, bins=180, range=(-90, 90), weights=L)
    i = int(np.argmax(hist))
    peak = 0.5 * (edges[i] + edges[i + 1])
    d = (ang - peak + 90.0) % 180.0 - 90.0
    sel = np.abs(d) <= angtol
    if sel.sum() == 0:
        return None, None
    dominant = float(np.average(peak + d[sel], weights=L[sel]))
    support = (np.stack([(segs[sel, 0] + segs[sel, 2]) / 2.0,
                         (segs[sel, 1] + segs[sel, 3]) / 2.0], axis=1),
               L[sel].copy())
    return dominant, support
