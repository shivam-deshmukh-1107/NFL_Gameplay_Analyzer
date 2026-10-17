"""The absolute-scale referee: measured yard gap vs the gap the game prints.

PROVENANCE
----------
Shivam's `scripts/valid_scale.py`, restructured so the measurement is a library
function and the verdict is a script. Yash's project has no equivalent: every
instrument in it compares the geometry against painted yard lines, which is to
say against the same evidence the geometry was fitted to.

WHY THIS IS THE STRONGEST CHECK EITHER PROJECT HAS
--------------------------------------------------
Blitz prints down-and-distance, so the yard gap between the line of scrimmage
and the line to gain is KNOWN. This measures that gap through the fitted
geometry and compares. Three things make it genuinely independent:

  1. The HUD is masked out before any geometry runs (`hudmask`), so the number
     cannot reach the thing being graded.
  2. The two drawn lines are found by COLOUR (`drawnlines`), separately from
     the white painted yard lines the lattice is fitted to.
  3. The quantity checked is a LENGTH IN YARDS, not a residual. Fit quality
     proves nothing - a homography fitted to 4 points has zero residual by
     construction - and both parent projects have a list of metrics that looked
     fine while the geometry was wrong. This one cannot flatter itself.

IT CATCHES WHAT PAINT-BASED METRICS STRUCTURALLY CANNOT. A grid mis-scaled by a
whole line spacing still lands on real paint and scores well on every
paint-fit measure in either project. Measured in Shivam's work: relabelling
lines from the fitted lattice drifted clip_04's scrimmage-to-first-down from
29 yd to **75.4** against a HUD still reading 29, and this check is the only
reason that was caught.

NO CAMERA IS INVOLVED
---------------------
The gap is read off the lattice `(v, alpha, beta)` directly, so this referee
works before any camera exists - which is why it is built first. Once the
camera lands (MERGE_PLAN.md Phase 1 step 3) the same measurement runs through
the certified homography instead, and that will be the first time Yash's camera
meets an absolute external number.

WHAT IT CANNOT DO
-----------------
It measures a DISTANCE, so it is blind to where that distance sits on the
field. It cannot say which yard line is the 40. Absolute identity needs the
"ON THE N" reading (`hudtext`) plus a direction cue, and that is Phase 4.

READ THE IQR, NOT ONLY THE MEDIAN. A tight spread means the scale is recovered
the same way on every frame; a wide one means frames disagree and the median
merely happens to land in the right place. Shivam's clip_03 passes on neither
count (IQR 4.34) because its pre-snap framing pushes the coloured lines partly
off-frame, and that is a real limitation of the instrument, not of the geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import drawnlines, gating, hudmask, ingest, lattice as LAT, markings, propose
from .config import YARDS_PER_CELL, VideoSpec

#: The pair of drawn lines whose separation the HUD states. The gap is
#: symmetric, so which is the scrimmage line and which is the line to gain does
#: not matter here - see drawnlines.py.
GAP_PAIR = ("orange_yellow", "red")


@dataclass
class ScaleRow:
    """One frame's contribution to the referee."""
    frame: int
    gap_yd: float | None = None
    n_lines: int = 0
    lattice_score: float | None = None
    agree: bool | None = None          # did the two proposers agree this frame
    found: tuple = ()                  # which drawn-line colours were fitted
    note: str = ""


@dataclass
class ScaleResult:
    clip: str
    stride: int
    rows: list = field(default_factory=list)

    @property
    def gaps(self) -> np.ndarray:
        return np.asarray([r.gap_yd for r in self.rows if r.gap_yd is not None])

    def summary(self, truth: float | None = None) -> dict:
        g = self.gaps
        out = {
            "clip": self.clip,
            "stride": self.stride,
            "n_frames": len(self.rows),
            "n_measured": int(len(g)),
            "median_yd": float(np.median(g)) if len(g) else None,
            "iqr_yd": (float(np.percentile(g, 75) - np.percentile(g, 25))
                       if len(g) > 3 else None),
            "p10_yd": float(np.percentile(g, 10)) if len(g) > 3 else None,
            "p90_yd": float(np.percentile(g, 90)) if len(g) > 3 else None,
            "truth_yd": truth,
        }
        if truth is not None and len(g):
            out["error_yd"] = out["median_yd"] - truth
            # Per-frame error distribution, not only the error of the median:
            # a median that lands on the truth while individual frames scatter
            # is a different situation from frames that agree.
            out["abs_error_median_yd"] = float(np.median(np.abs(g - truth)))
            out["abs_error_p90_yd"] = float(np.percentile(np.abs(g - truth), 90))
        return out


def gap_through_lattice(fit: LAT.LatticeFit, segs: dict, w: int,
                        pair=GAP_PAIR) -> tuple[float | None, tuple]:
    """Yard gap between two drawn lines, measured through a lattice fit.

    Each drawn line is intersected with the same transversal the lattice was
    fitted on, mapped to a FRACTIONAL cell index, and the difference scaled by
    `YARDS_PER_CELL`. Fractional on purpose: the drawn lines sit wherever the
    ball is, not on the 5-yard family, so rounding would destroy the quantity
    being measured.
    """
    ell = np.array([1.0, 0.0, -w / 2.0])
    idx = {}
    for name, seg in segs.items():
        s, keep = LAT.transversal_positions(np.asarray(seg, float)[None], ell)
        if not keep.any():
            continue
        d = s[0] - fit.v
        if abs(d) < 1e-9 or abs(fit.alpha) < 1e-15:
            continue
        u = 1.0 / d
        v = (u - fit.beta) / fit.alpha
        if np.isfinite(v):
            idx[name] = float(v)
    found = tuple(sorted(idx))
    a, b = pair
    if a in idx and b in idx:
        return abs(idx[a] - idx[b]) * YARDS_PER_CELL, found
    return None, found


def measure(spec: VideoSpec, stride: int = 2, net=None,
            collect_glyphs: bool = False) -> tuple[ScaleResult, list, dict]:
    """Run the referee over one video.

    Returns `(result, glyph_lines, geo)`. The last two are only populated when
    `collect_glyphs` is set, and exist so `scripts/read_hud.py` can reuse this
    one pass to build the HUD glyph book - the geometric distances it needs are
    exactly what this function already computes.

    NO TRACKER. The lattice is fitted cold on every frame, independently. That
    is noisier than Shivam's version, which offers the cold fit to a tracker
    and measures whatever lock the tracker keeps, and it is the right choice
    here: with no tracker built yet, a per-frame fit tests the lattice itself
    rather than a tracker's memory of it. Expect a wider IQR than his numbers
    for that reason alone.
    """
    payload, pic = ingest.load(spec)
    turf, gate = gating.load(spec)
    hud = hudmask.load(spec)
    spans = gating.gameplay_spans(gate) or [(0, payload["meta"]["nb_frames"] or 0)]

    res = ScaleResult(clip=spec.vid, stride=stride)
    glyph_lines, geo = [], {}
    rng = np.random.default_rng(0)

    for lo, hi in spans:
        with ingest.FrameSource(spec, pic, stride=stride, start=lo, stop=hi) as src:
            for i, fr in src:
                row = ScaleRow(frame=int(i))
                prep = markings.FramePrep.build(fr, turf, hud=hud, net=net)
                fm = markings.analyse_frame(fr, turf, hud=hud, frame_index=i,
                                            rng=rng, prep=prep)
                if collect_glyphs:
                    for gl in hudtext_lines(fr):
                        glyph_lines.append(((spec.order, int(i)), gl))

                p = propose.compare(fm.pencil, pic.w)
                row.n_lines = int(len(p.s))
                row.agree = p.agree
                if p.lat is None:
                    row.note = p.note or "no lattice"
                    res.rows.append(row)
                    continue
                row.lattice_score = float(p.lat.score)

                drawn = drawnlines.fit_drawn_lines(fr, turf, prep=prep,
                                                   vp_cross=fm.vp_cross)
                gap, found = gap_through_lattice(p.lat, drawn, pic.w)
                row.found = found
                row.gap_yd = gap
                if gap is None:
                    row.note = "drawn pair incomplete"
                else:
                    geo[(spec.order, int(i))] = gap
                res.rows.append(row)
    return res, glyph_lines, geo


def hudtext_lines(frame):
    """Indirection so `measure` need not import hudtext unless asked."""
    from . import hudtext
    return hudtext.text_lines(frame)


# --------------------------------------------------------------------------- #
# the same referee, through the camera instead of the lattice
# --------------------------------------------------------------------------- #
def gap_through_camera(H, segs: dict, w: int, pair=GAP_PAIR):
    """Yard gap between two drawn lines, measured through a camera homography.

    **This is the measurement MERGE_PLAN.md Phase 1 step 3 calls the single
    most valuable one available**, and the reason is worth stating plainly:
    Yash's camera has never been checked against an absolute external number.
    Every instrument in his project - `independent_paint`, `paint_check`,
    `flow_check`, the project audit's `density` and `err_yd` - compares the
    pose against PAINTED YARD LINES, which is to say against the same kind of
    evidence the pose was fitted to. A grid shifted by a whole line spacing
    lands on paint and satisfies all of them; so does a grid at the wrong
    metric scale, as long as it is self-consistent.

    This does not. It takes a length in yards that the game itself states, from
    a HUD that was masked out before any geometry ran, between two lines found
    by colour rather than by the white-paint detector.

    Cleaner than the lattice route, too: no `(v, alpha, beta)` and so none of
    their jitter - the drawn line's crossing of the centre column is
    inverse-projected straight to a field coordinate.
    """
    import numpy as np

    from . import camera as CAM
    from . import geometry as GEO

    xc = w / 2.0
    ell = np.array([1.0, 0.0, -xc])
    yard = {}
    for name, seg in segs.items():
        L = GEO.seg_to_line(np.asarray(seg, float)[None])[0]
        P = np.cross(L, ell)
        if abs(P[2]) < 1e-9:
            continue
        y_at = P[1] / P[2]
        try:
            yd, lat = CAM.pixel_to_field(H, xc, float(y_at))
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(yd):
            yard[name] = float(yd)
    found = tuple(sorted(yard))
    a, b = pair
    if a in yard and b in yard:
        return abs(yard[a] - yard[b]), found
    return None, found
