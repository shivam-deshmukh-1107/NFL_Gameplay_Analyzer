"""Stage 6a - camera motion between consecutive frames.

PROVENANCE
----------
Ported from Shivam's `blitzfield/motion.py`. **This replaces Yash's ORB chain
outright**, and the decision is his own measurement, not a preference:

  * His retraction #11 records that the ORB HUD mask was computed and never
    passed to `detectAndCompute`, so the chain was matching the SCOREBOARD and
    returning identity homographies. A chain of identities FREEZES rather than
    drifting, which is how "23 deg of silent rotation drift" came to be
    measured and then withdrawn.
  * Even with the mask passed, cross-clip frames still match at 214-296
    inliers because the rendered field is identical between clips, so his
    `CHAIN_BROKEN` state is unreachable on this footage.
  * Measured head to head on the same 80 consecutive frames, LK with
    forward-backward checking gives **~45% less frame-to-frame jitter
    (1.06 px vs 1.94 px)**, cleaner scale (sd 0.0032 vs 0.0051) and cleaner
    rotation (sd 0.08 vs 0.27 deg). Those last two matter because they feed
    thresholds of the same order.

TWO DETAILS THAT ARE LOAD-BEARING
---------------------------------
* **Features are restricted to the turf.** The HUD never moves regardless of
  the camera, so tracks on it drag the estimate toward zero. Learning that
  region beats a hardcoded UI rectangle: a fixed mask tuned to one capture's
  layout discarded 36% of the actual turf on this footage.
* **Forward-backward consistency rejects tracks that slid along a line.** The
  field is full of long parallel edges, so the aperture problem is the dominant
  error mode here, and descriptor matching has no equivalent guard.

WHAT MOTION IS AND IS NOT USED FOR
----------------------------------
It predicts WHERE things are, so it may label which detected line is which
yard, and it may carry a pose into the TRACKED tier. **It never supplies
geometry to a certified frame**, and it never supplies roll: roll comes from
the paint, every frame, because a chain cannot observe it.

Classification is for reporting and for cut detection. Raw parameters are
median-filtered BEFORE thresholding, rather than classifying each frame and
voting on labels: smoothing the continuous signal keeps magnitude information
that a vote throws away, and a median stays robust to the occasional bad
RANSAC fit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import det_scale

# Thresholds, all measured by Shivam at SHIVAM_REF_W and scaled per picture.
STATIC_PX = 0.8          # median displacement below this is a held camera
CUT_PX = 45.0            # this much displacement AND few tracks means a cut
# Scale-free: a ratio and an angle.
ZOOM_EPS = 0.0035        # |scale - 1| above this is a dolly
ROT_EPS_DEG = 0.35       # rotation above this is a camera roll
CUT_TRACKS = 30

#: What each label implies for anything consuming the geometry downstream.
MEANING = {
    "static": "camera held; one calibration covers the whole segment",
    "pan": "camera translates; line positions shift but pixels-per-yard holds",
    "zoom_in": "scale grows; pixels-per-yard changes, recalibrate per frame",
    "zoom_out": "scale shrinks; pixels-per-yard changes, recalibrate per frame",
    "pan+zoom": "translation and scale change together; hardest case",
    "rotate": "camera rolls with little translation; line slopes change",
    "cut": "correspondence lost; hard boundary, do not track across",
    "unknown": "estimation failed (too few tracks)",
}


@dataclass
class Motion:
    label: str = "unknown"
    dx: float = 0.0
    dy: float = 0.0
    zoom: float = 1.0        # >1 moving in
    rot_deg: float = 0.0
    n_tracks: int = 0
    speed: float = 0.0       # median displacement magnitude, px/frame
    speed_s: float = 0.0     # smoothed, the value actually classified
    zoom_s: float = 1.0
    rot_s: float = 0.0
    #: 3x3 similarity taking PREVIOUS-frame pixels to THIS frame's pixels;
    #: None when estimation failed. Geometry propagation uses this and must
    #: treat None as "no information", never as identity.
    H: object = None

    @property
    def ok(self) -> bool:
        return self.H is not None and self.label not in ("cut", "unknown")


def classify(speed: float, zoom: float, rot_deg: float, n_tracks: int,
             s: float = 1.0) -> str:
    """Label one already-smoothed motion estimate."""
    # A cut needs BOTH a huge apparent jump and a collapse in correspondence.
    # Speed alone misfires on a genuinely fast pan, which still tracks well.
    if speed > CUT_PX * s and n_tracks < CUT_TRACKS:
        return "cut"
    moving = speed > STATIC_PX * s
    zooming = abs(zoom - 1.0) > ZOOM_EPS
    rotating = abs(rot_deg) > ROT_EPS_DEG
    if not (moving or zooming or rotating):
        return "static"
    if zooming and moving:
        return "pan+zoom"
    if zooming:
        return "zoom_in" if zoom > 1.0 else "zoom_out"
    if rotating and not moving:
        return "rotate"
    return "pan"


class MotionEstimator:
    """Frame-to-frame similarity transform, classified from smoothed signals."""

    def __init__(self, smooth: int = 5, max_corners: int = 400,
                 picture_w: int = 1356):
        self.prev = None
        self.max_corners = max_corners
        self.s = det_scale(picture_w)
        self.hist: deque = deque(maxlen=max(1, smooth))

    def reset(self):
        self.prev = None
        self.hist.clear()

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> Motion:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = self.s
        m = None
        if mask is not None:
            k = max(3, int(round(9 * s)) | 1)
            m = cv2.erode(mask.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if self.prev is None:
            self.prev = gray
            return Motion()

        min_dist = max(2.0, 8.0 * s)
        p0 = cv2.goodFeaturesToTrack(self.prev, self.max_corners, 0.01,
                                     min_dist, mask=m)
        if p0 is None or len(p0) < 12:
            self.prev = gray
            return self._finish(Motion())

        win = max(7, int(round(21 * s)) | 1)
        p1, stt, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None,
                                              winSize=(win, win), maxLevel=3)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None,
                                              winSize=(win, win), maxLevel=3)
        ok = (stt.ravel() == 1) & (stb.ravel() == 1)
        # Forward-backward consistency: the dominant error mode here is a track
        # sliding ALONG a painted line, and nothing else catches it.
        ok &= np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < 1.5 * s
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        self.prev = gray
        if len(a) < 12:
            return self._finish(Motion())

        M, inl = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5 * s)
        if M is None:
            return self._finish(Motion())
        H = np.eye(3)
        H[:2, :] = M
        mo = Motion(
            H=H,
            dx=float(M[0, 2]), dy=float(M[1, 2]),
            zoom=float(np.hypot(M[0, 0], M[0, 1])),
            rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            n_tracks=int(inl.sum()) if inl is not None else 0,
            speed=float(np.median(np.linalg.norm(b - a, axis=1))),
        )
        return self._finish(mo)

    def _finish(self, mo: Motion) -> Motion:
        """Median-filter the parameters, then classify from smoothed values."""
        self.hist.append(mo)
        live = [h for h in self.hist if h.n_tracks > 0]
        if not live:
            mo.label = "unknown"
            return mo
        mo.speed_s = float(np.median([h.speed for h in live]))
        mo.zoom_s = float(np.median([h.zoom for h in live]))
        mo.rot_s = float(np.median([h.rot_deg for h in live]))
        tracks = int(np.median([h.n_tracks for h in live]))
        mo.label = ("unknown" if mo.n_tracks == 0
                    else classify(mo.speed_s, mo.zoom_s, mo.rot_s, tracks, self.s))
        return mo


def invert(H) -> np.ndarray | None:
    """The reverse of a motion homography, for walking backwards.

    A motion `H` maps previous-frame pixels to this frame's. Walking from frame
    t back to t-1 needs the inverse, so a backward walk is the same code with
    this applied - it is not a separate estimator, and that matters: an
    independently estimated backward motion would not compose exactly with the
    forward one, and the two walks would disagree about where the anchor is.
    """
    if H is None:
        return None
    try:
        return np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return None


def chain_pose(H_pose, H_motion) -> np.ndarray | None:
    """Carry a camera pose through one frame of image-space motion.

    `H_pose` maps (yard, lateral) -> previous-frame pixels and `H_motion` maps
    previous-frame pixels -> this frame's, so the composition maps the field
    straight into this frame. Used ONLY to predict - to label which line is
    which yard, and to carry a TRACKED pose - never to certify.
    """
    if H_pose is None or H_motion is None:
        return None
    out = np.asarray(H_motion, float) @ np.asarray(H_pose, float)
    if not np.all(np.isfinite(out)) or abs(np.linalg.det(out)) < 1e-14:
        return None
    return out
