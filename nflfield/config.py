"""Project-wide paths, measured constants and the video registry.

PROVENANCE
----------
This project merges two independent field trackers (see ../MERGE_PLAN.md):

  * `Field_Tracker/` (Shivam Deshmukh) - lattice tracker: comb-score yard
    lattice in horizon-relative inverse coordinates, learned/colour front end,
    unsupervised gating, HUD text reader, parallel render.
  * `Yash_Field_Tracker/easely-done/` (Yash Kamerkar) - camera tracker:
    6-parameter pinhole solve per frame, named refusal gates, measured metric
    field template, independent referees.

Every constant below carries its measurement and which project measured it.
Do not change one without re-measuring, and write the new measurement beside it.


THE PICTURE FRAME - a deliberate change from both parents
---------------------------------------------------------
Both parents normalise the frame, differently, and neither convention survives
the merge unmodified:

  * Shivam resized the active picture into a fixed 960x720 canonical buffer.
    Clean for the net, but it rescales every pixel distance by 0.708 on these
    clips, and painted lines here are 1-3 px wide at native resolution.
  * Yash cropped the pillarbox and kept NATIVE resolution, because his camera
    gates are pixel thresholds measured at picture width 1356 and his focal
    constant scales as `1511 * W/1356`.

This project works in the **picture frame**: the active content box at its own
native resolution, with an exact invertible transform back to source pixels.

Why that way round:
  1. Clips 1-4 are natively 1356 px wide, so the picture frame IS the frame
     every camera constant was measured in. Those gates hold exactly, and our
     numbers on clips 1-4 stay directly comparable to Yash's committed results -
     which is the only free validation either project has.
  2. Thin paint survives. A 0.708x downscale does not help a 1 px line.
  3. Other sources are handled by scaling, not by resampling: `px_scale(W)`
     below extends to the residual gates the same rule Yash already applies to
     the focal length.
  4. The learned front end is the only component that needs a fixed input size,
     and it can resize internally and upsample its masks - one module's problem
     instead of the whole pipeline's.

The cost is that pixel-valued constants must be written as "measured at
PX_REF_W" and scaled at use. `px_scale()` is the single place that happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"           # source videos
WORK = ROOT / "work"           # per-video intermediates; rebuildable
OUT = ROOT / "out"             # deliverables only: rendered video + templates


# --------------------------------------------------------------------------- #
# the field, in yards
# --------------------------------------------------------------------------- #
YARDS_PER_CELL = 5.0
"""Painted yard lines are 5 yd apart.

Measured independently by both parents: Yash on clip_04 f190 (goal line at
y=258, then 307/360/439/571, with the painted "10" beside 360 and the "20"
beside 571 and exactly one line between them); Shivam against the game's own
down-and-distance to +-0.1 yd.
"""

FIELD_LENGTH_YD = 100.0
"""Goal line to goal line.

Shivam, from 1,814 HUD "ON THE N" readings with a hard ceiling at exactly 50
and no value above it, plus the football convention that N counts from the
nearer goal line.
"""

END_ZONE_YD = 10.0
"""End zone depth.

Yash, `experiments/archive/field_layout.py`: the painted white end line sits at
X = 110.91 / 110.99 / 110.94 on three independent clip maps with the goal line
at 100 - one number, three clips. An earlier 8.00 yd reading on clip_02 was an
artefact of a single navy column falling below a 45% threshold.
"""

FIELD_WIDTH_YD = 53.3333
"""Sideline to sideline. **ASSUMED, NOT MEASURED - the weakest number here.**

Yash's retraction #9 states it plainly: `GAME_FOCAL_PX` rests solely on this
assumption, so every cross-field yard value inherits it. His own measurement of
the painted layout came out at 52.954 yd for sideline-to-sideline, i.e. 0.7%
short of the assumed value, which is consistent but not independent (the
layout was placed through a focal calibrated on the assumption).

Two non-circular tests exist and neither has been run: the painted border band
width (measured 2.0 yd, against 1.5 in Yash's `lateral.py` and 1.8 in its
docstring - two instruments 25-30% apart on one physical length), and Shivam's
selection-ring circular-points rectification, which needs no focal length at
all. See MERGE_PLAN.md Phase 5.
"""

# Lateral layout, measured by Yash in experiments/archive/field_layout.py from a
# merged top-down map of clips 02/03/04. The RATIOS are what was measured; the
# yard values inherit FIELD_WIDTH_YD above. Per-clip spread in the last column.
HASH_SEPARATION_YD = 7.165        # per-clip spread 0.076-0.084
NUMBER_ROW_SEPARATION_YD = 22.733  # per-clip spread 0.019-0.138
BORDER_BAND_YD = 2.011            # painted out-of-bounds band, field side 1.987
"""Yash's layout measurement. Regulation is 6.167 (hash) and 27.333 (numbers);
Blitz does not reproduce the real field, which is why these are measured rather
than looked up. The repo's older 7.42 hash figure was peak-to-peak, not
half-maximum band centres, and biased the hash midpoint the cross-field offset
hangs on.
"""


# --------------------------------------------------------------------------- #
# the camera
# --------------------------------------------------------------------------- #
GAME_FOCAL_PX = 1511.0
"""One focal length for the whole game, in pixels at PX_REF_W.

Yash, 2026-09-09. The focal is NOT observable from yard lines: pin it anywhere
from 700 to 4200 px, refit the other five parameters, and the residual is
bit-for-bit identical while camera height moves 18.7 -> 7.0 yd. A frame with 11
lines is exactly as uninformative as one with three.

So it is calibrated on a POSITION, not a direction: the hash rows' midpoint is
the field's centre by symmetry and the painted out-of-bounds border is its edge,
making midpoint-to-border half the field width - a 26.67 yd baseline. Clips
02/03/04 calibrated in isolation give 1521/1497/1512, agreeing to 1%.

It is also flat across clip_04's apparent zoom (1502/1521/1512 across thirds):
**the Blitz camera dollies, it does not zoom.** Do not re-estimate per clip.

INHERITS FIELD_WIDTH_YD's assumption - see retraction #9 there.
"""

PX_REF_W = 1356
"""Picture width every pixel-valued constant in this project was measured at.

Clips 1-4 are natively 1356 px wide, so on them `px_scale() == 1.0` and every
measured gate is used at exactly the value it was measured at.
"""


def px_scale(picture_w: float) -> float:
    """Scale factor for a pixel-valued constant measured at `PX_REF_W`.

    The focal length is in pixels and so scales with the picture; Yash's
    `clip_focal` has always done this. Residual and tolerance gates are also
    pixel distances in the same image, so they scale by the same factor. This
    is the single place that conversion happens.

    THE ARGUMENT IS THE PICTURE WIDTH, NEVER THE CONTAINER WIDTH. Handing it a
    1920 pillarbox container instead of the 1443 picture inside gave a 33%
    focal error on clip_05 and announced itself to nobody (Yash, content_box).
    """
    return float(picture_w) / PX_REF_W


def focal_px(picture_w: float, picture_h: float) -> tuple[float, float]:
    """(fx, fy) for a picture of this size, in that picture's own pixels.

    TWO FOCALS, NOT ONE - a deliberate generalisation of Yash's single-`f`
    camera. A pinhole K with one focal is only correct when the picture frame is
    an ISOTROPIC scaling of the game's render. That holds for the 4:3 clips
    (clip_04 measures anisotropy 1.0009) and fails for an anisotropically
    stretched upload, where Shivam measured a 1746x982 active area on what is a
    4:3 render. With (fx, fy) the model is exact in both cases and identical to
    the single-focal form whenever the source is undistorted, so nothing that
    was measured moves.
    """
    fx = GAME_FOCAL_PX * (float(picture_w) / PX_REF_W)
    # The reference picture is 4:3, so its height reference is PX_REF_W * 3/4.
    fy = GAME_FOCAL_PX * (float(picture_h) / (PX_REF_W * 3.0 / 4.0))
    return fx, fy


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
#: Width Shivam's detection constants were measured at - his canonical buffer,
#: NOT PX_REF_W. Mixing the two reference widths silently is the exact trap this
#: project is most likely to fall into, so every constant says which it uses.
SHIVAM_REF_W = 960


def det_scale(picture_w: float) -> float:
    """Factor converting a DETECTION constant measured at 960 to this picture.

    The camera gates were measured at `PX_REF_W` (1356) and the detection
    constants at `SHIVAM_REF_W` (960). Two reference widths in one pipeline is
    an accident waiting to happen, so there are two named conversions and no
    bare numbers.

    MEASURED CONSEQUENCE OF GETTING THIS WRONG, 2026-10-17. Carried over
    unscaled, `merge_collinear`'s 80 px gap tolerance is ~57 px of real
    tolerance at 1356 and ~53 px at 1444, so fragments of one painted line that
    a player broke apart never rejoin. On the dark-graded Hail Mary capture
    (turf L of 19 against the clips' 100, so paint has low contrast and
    fragments more) the long-line count read 36 per frame against Shivam's
    documented 10 - the extra 26 were pieces of lines already counted. Nothing
    errored: the pencil just filled with fragments, each individually long
    enough to pass the length cut, all of them heading for the 5-yard ladder.

    Lengths and offsets scale by this factor; AREAS scale by its square.
    """
    return float(picture_w) / SHIVAM_REF_W

LONG_CUT_PX = 110.0 * (PX_REF_W / SHIVAM_REF_W)     # = 155.4 px at PX_REF_W
"""Separates full yard lines from 1-yard tick marks, in px at PX_REF_W.

Shivam measured 110 px on his 960-wide canonical buffer; the converted value is
what this project uses, because it works at the native picture width.

CAUGHT 2026-10-17 BY MEASUREMENT, and recorded because the failure was silent.
Carried over as a flat 110 px it admits short segments as full yard lines: on
clip_04 f0 the long-line count reads 20 against 15 at the correct cut, and on
clips 1-4 the per-frame medians read 12/14/9/16 against Shivam's documented
10/12/9/14. Nothing errors - the pencil just quietly gains members, which then
reach the ladder.

The underlying measurement is unchanged and still Shivam's: the length
distribution here is strongly bimodal, so the exact cut is not critical - but
it has to be in the right units.
"""

NET_INPUT_W = 960
NET_INPUT_H = 720
"""Input size the learned front end was trained at (Shivam's canonical buffer).

The net is the only component that needs a fixed input, so it resizes into this
box itself and upsamples its masks back to the picture frame. Nothing else in
the pipeline sees this size.
"""


# --------------------------------------------------------------------------- #
# the video registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoSpec:
    """One input video.

    `build` is the game build, which matters because the turf colour, the HUD
    layout and the painted-number style differ between them, and a texture map
    or a trained net is only valid within one build. It is recorded, never
    guessed: `None` means not established.
    """

    order: int
    vid: str              # slug; names the work directory and the output file
    filename: str
    note: str
    build: str | None = None

    @property
    def path(self) -> Path:
        return DATA / self.filename

    @property
    def workdir(self) -> Path:
        return WORK / f"{self.order:02d}_{self.vid}"

    @property
    def video_out(self) -> Path:
        return OUT / f"{self.vid}_field.mp4"


VIDEOS: list[VideoSpec] = [
    # Clips 1-4 are the four hand-picked test clips, each chosen to exercise one
    # camera behaviour. Both parent projects have committed results on these
    # exact files (byte-identical in both repositories), which makes them the
    # validation set for the merge.
    #
    # NAMING TRAP, and it cost Yash's project days: clip 4 "Camera Swing" is the
    # LEAST rotating clip - its swing is a lateral pan, max 3.0 deg of tilt and
    # zero frames above 5. **Clip 1 is the rotation clip**, at 33 deg. Any
    # rotation test must use clip 1.
    VideoSpec(1, "clip1_kickoff", "Clip_1 Kickoff return not QB.mp4",
              "kickoff return, wide camera, non-QB view; THE ROTATION CLIP (33 deg)",
              build="A"),
    VideoSpec(2, "clip2_congestion", "Clip_2 Congestion Contact Test.mp4",
              "heavy player congestion / contact; occlusion test",
              build="A"),
    VideoSpec(3, "clip3_qb", "Clip_3 Easiest QB Test.mp4",
              "clean QB view, easiest case; tight pre-snap framing",
              build="A"),
    VideoSpec(4, "clip4_swing", "Clip_4 Camera Swing and Distance Change Test.mp4",
              "lateral pan + dolly; NOT a roll test (3.0 deg max)",
              build="A"),
    # A different capture with a dark, desaturated colour grade: turf at
    # Lab (a,b) = (119.3, 132.9) against the clips' (110, 126). It broke
    # Shivam's trained net completely (4-10% P(turf) against the colour model's
    # 65-88%) until it had its own auto-labelled rows - the third time that
    # happened, hence the standing rule in fieldcolor/fieldnet.
    #
    # Build NOT established: the grade differs, but nothing here has checked
    # whether the HUD layout and painted numbers match build A.
    VideoSpec(5, "hail_mary_left", "1_Hail_Mary_Left_Side.mp4",
              "single play, dark desaturated grade; new-capture generalisation test",
              build=None),
]

BY_ORDER = {v.order: v for v in VIDEOS}
BY_VID = {v.vid: v for v in VIDEOS}


def get_video(key: int | str) -> VideoSpec:
    """Look a video up by processing order or by slug."""
    if isinstance(key, int) or (isinstance(key, str) and str(key).isdigit()):
        return BY_ORDER[int(key)]
    return BY_VID[str(key)]
