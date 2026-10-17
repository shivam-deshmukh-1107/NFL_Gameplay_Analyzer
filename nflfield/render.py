"""Stage 10 - draw what the pipeline certified, and nothing it did not.

PROVENANCE
----------
Overlay semantics are Yash's, because they encode a rule rather than a style:
**never draw a grid from a pose that was not validated.** Colour says which
tier a frame is in, and the banner says why when nothing is drawn.

    green   CERTIFIED  camera solved from this frame's own paint, all 8 gates
    blue    CARRIED    ECC turf alignment from a certified keyframe, validated
                       against the visible paint, bounded at 30 frames
    amber   TRACKED    pose carried through measured motion; NO camera claim
    nothing REFUSED    the evidence is not there; the banner names the gate

The encode is Shivam's: raw BGR piped to libx264 at CRF 18 with the original
audio muxed back, rather than `cv2.VideoWriter`, whose MPEG-4 Part 2 default he
measured as far larger and softer.

=============================================================================
WHY NO YARD NUMBERS ARE PRINTED
=============================================================================
Absolute yard identity is not established. The lattice and the camera both
place the lines beautifully and neither can say WHICH line is the 40: painted
numbers are mirror-symmetric, so a "40" sits at yard 40 and at yard 60, and
only the end zone, the direction chevrons or the midfield logo break that tie.
Our HUD reader can read the absolute position in principle and cannot learn
enough digits from five single-play clips (2 of 10).

So the overlay prints **no yard numbers at all** and captions the fact. Yash's
project made the same choice for the same reason, and the reasoning is worth
keeping verbatim: showing a confident "40" the system cannot stand behind would
be the most misleading thing it could do.

WHAT TO LOOK FOR WHEN WATCHING
------------------------------
Both parent projects state that looking at the overlay catches defects that
pass every numeric test - a grid drawn vertically down the picture once
reported 0.03 deg of angle error. So:

  * do the green lines sit ON the painted lines, at every depth?
  * does the spacing look like one grid line per painted line, not every other?
  * at a tier change, does the geometry jump?
  * where nothing is drawn, is the banner's reason plausible?
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

from . import camera as CAM
from .config import YARDS_PER_CELL

#: BGR, deliberately high-contrast against turf and distinguishable when the
#: video is compressed. One colour per tier and no gradients: a viewer must be
#: able to tell the tier from a single frame.
TIER_COLOUR = {
    "CERTIFIED": (80, 255, 80),      # green
    "CARRIED": (255, 170, 60),       # blue (BGR)
    "TRACKED": (40, 190, 255),       # amber
    "REFUSED": None,                 # nothing is drawn
}
TIER_LABEL = {
    "CERTIFIED": "TRACKING",
    "CARRIED": "CARRIED - NOT CERTIFIED",
    "TRACKED": "PREDICTED - NOT CERTIFIED",
    "REFUSED": "NOT TRACKED",
}

CAPTION = "no yard numbers: absolute identity not established"


def _scale(w: int) -> float:
    """Drawing sizes scale with the picture so a 1444-wide clip is not thinner."""
    return max(0.6, w / 1356.0)


def draw_grid(img, H, tier: str, thickness: int | None = None,
              half_step: bool = False) -> int:
    """Draw the projected 5-yard grid. Returns the number of lines drawn.

    Lines are drawn edge to edge from the camera, not from the detected
    segments: once a pose has passed its gates the position of a yard line no
    detector happened to report this frame is determined, not guessed. If the
    pose is wrong, drawing the full grid is what makes it obvious - which is
    the point of looking.
    """
    if H is None:
        return 0
    colour = TIER_COLOUR.get(tier)
    if colour is None:
        return 0
    h, w = img.shape[:2]
    s = _scale(w)
    if thickness is None:
        thickness = max(1, int(round(2 * s)))
    xs = np.array([0.0, float(w)])
    n = 0
    for yard, _y, _sl in CAM.grid_lines_image(H, w, h):
        yy = CAM.line_y_at(H, float(yard), xs)
        if yy is None or not np.all(np.isfinite(yy)):
            continue
        p0 = (0, int(round(yy[0])))
        p1 = (int(w), int(round(yy[1])))
        cv2.line(img, p0, p1, colour, thickness, cv2.LINE_AA)
        n += 1
    if half_step:
        # The half-step control: lines at yard + 2.5, which must fall BETWEEN
        # the painted lines. If they land on paint, the grid is half-scaled -
        # a failure no distance metric can see. Off by default; a diagnostic.
        for yard, _y, _sl in CAM.grid_lines_image(H, w, h):
            yy = CAM.line_y_at(H, float(yard) + YARDS_PER_CELL / 2.0, xs)
            if yy is None or not np.all(np.isfinite(yy)):
                continue
            cv2.line(img, (0, int(round(yy[0]))), (int(w), int(round(yy[1]))),
                     (200, 200, 200), 1, cv2.LINE_AA)
    return n


def draw_banner(img, row: dict, clip: str) -> None:
    """Status line: tier, evidence, and - when refused - which gate and why."""
    h, w = img.shape[:2]
    s = _scale(w)
    tier = row.get("tier", "REFUSED")
    colour = TIER_COLOUR.get(tier) or (170, 170, 170)

    bits = ["f%d" % row.get("frame", -1), TIER_LABEL.get(tier, tier)]
    if row.get("n_lines") is not None:
        bits.append("%d lines" % row["n_lines"])
    if tier == "CERTIFIED" and row.get("resid_px") is not None:
        bits.append("resid %.2f px" % row["resid_px"])
    if tier == "CARRIED":
        if row.get("carried_from") is not None:
            bits.append("from f%d (+%d)" % (row["carried_from"],
                                            row.get("carried_frames", 0)))
        if row.get("carry_check_yd") is not None:
            bits.append("paint %.2f yd" % row["carry_check_yd"])
    if tier == "TRACKED" and row.get("track_age"):
        bits.append("age %d" % row["track_age"])
    if tier == "REFUSED" and row.get("reason"):
        bits.append(str(row["reason"])[:60])
    text = "  |  ".join(bits)

    fs = 0.52 * s
    th = max(1, int(round(1.2 * s)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(round(6 * s))
    y0 = int(round(8 * s))
    box = img[y0:y0 + tht + 2 * pad, 0:min(w, tw + 2 * pad)]
    if box.size:
        box[:] = (box * 0.35).astype(img.dtype)        # darken, do not blank
    cv2.putText(img, text, (pad, y0 + pad + tht),
                cv2.FONT_HERSHEY_SIMPLEX, fs, colour, th, cv2.LINE_AA)

    cap = "%s  -  %s" % (clip, CAPTION)
    (cw, cht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX,
                                   0.42 * s, max(1, int(round(s))))
    y1 = h - int(round(10 * s))
    box2 = img[max(0, y1 - cht - pad):min(h, y1 + pad), 0:min(w, cw + 2 * pad)]
    if box2.size:
        box2[:] = (box2 * 0.35).astype(img.dtype)
    cv2.putText(img, cap, (pad, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s,
                (210, 210, 210), max(1, int(round(s))), cv2.LINE_AA)


def draw_drawn_lines(img, drawn: dict, pair=("orange_yellow", "red")) -> None:
    """Mark the game's own scrimmage / first-down lines, if they were fitted.

    Drawn thin and dashed-looking so they cannot be mistaken for the grid: they
    are evidence the pipeline MEASURES (the HUD states their separation) rather
    than geometry it produced.
    """
    if not drawn:
        return
    h, w = img.shape[:2]
    s = _scale(w)
    for name in pair:
        seg = drawn.get(name)
        if seg is None or len(seg) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in seg]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 (255, 255, 255), max(1, int(round(s))), cv2.LINE_AA)


def frame_overlay(frame, row: dict, clip: str, drawn: dict | None = None,
                  half_step: bool = False):
    """Composite one annotated frame. `frame` is modified in place and returned."""
    tier = row.get("tier", "REFUSED")
    H = None
    if row.get("homography"):
        H = np.asarray(row["homography"], float).reshape(3, 3)
    # NOTHING GEOMETRIC IS DRAWN ON A REFUSED FRAME. A failed frame still
    # carries raw detections, and on a replay wipe or a menu those are turf
    # texture and banner edges - one frame in Shivam's project drew 73 "yard
    # lines" across a zoomed shot before this rule existed.
    if tier != "REFUSED":
        draw_grid(frame, H, tier, half_step=half_step)
        if drawn:
            draw_drawn_lines(frame, drawn)
    draw_banner(frame, row, clip)
    return frame
