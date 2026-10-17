"""Stage 1a - colour models for turf, painted markings and the drawn game lines.

PROVENANCE
----------
Ported whole from Shivam's `blitzfield/fieldcolor.py`. This replaces Yash's
fixed HSV ranges (`absolute_anchor.TURF_LOWER/TURF_UPPER`), which already
needed a per-clip special case (`DARK_LIT_MAX/MIN`) to cope with one capture
whose lower field fades to a median V of 2/255. A model refitted per video
needs no such case; whether that special case can actually be deleted is a
falsifiable test on clip_07 once we have it (MERGE_PLAN.md Phase 2).

Everything here is learned from the video itself - no hand-tuned hue ranges.
The one structural prior is that the central-lower part of a gameplay frame is
nearly always turf, which is enough to bootstrap.

Working colour space is CIELAB:
  * `L`      separates bright paint from dark turf
  * `(a, b)` is chromaticity, largely invariant to the stadium lighting ramp,
             so turf in shadow still scores as turf

Measured turf chromaticity on the three captures seen between the two parent
projects: teal (110, 126) on these clips, green (102, 151) on the Midway
capture, desaturated (119, 133) on the Hail Mary clip. Painted lines are
near-neutral (a, b ~ 128) in all three. That chroma gap is what lifts paint off
the field without a hue threshold.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# turf
# --------------------------------------------------------------------------- #
@dataclass
class TurfModel:
    """Robust 2-D Gaussian over turf chromaticity (a, b), plus an L range."""

    mean: list          # [a, b]
    cov: list           # 2x2
    l_lo: float
    l_hi: float
    n_pixels: int

    # ---- construction ---------------------------------------------------- #
    @staticmethod
    def fit(frames, band=(0.55, 0.86), xband=(0.25, 0.75),
            max_pixels: int = 120_000, seed: int = 0) -> "TurfModel":
        """Fit from a list of picture-frame BGR frames.

        A horizontal band across the lower middle of the frame is the bootstrap
        sample. That band is *mostly* turf during play - but a full-length
        capture is substantially menus, play-select screens and title cards, so
        the sample can be nearly half contaminated.

        A TRIMMED MEAN CANNOT SURVIVE THAT. Measured on the long captures it
        dragged the b-channel mean from ~126 to ~140 and inflated the standard
        deviation four-fold, after which the 3-sigma gate accepted almost every
        colour and the gameplay gate lost all discrimination.

        Mode seeking fixes it: turf is by far the densest single colour in that
        band, so take the chromaticity histogram peak and grow a Gaussian around
        it, re-estimating a few times. O(n), deterministic, and unaffected by
        contamination sitting anywhere else in colour space. (MinCovDet solves
        the same problem and is far too slow at hundreds of thousands of pixels.)
        """
        rng = np.random.default_rng(seed)
        samples = []
        for f in frames:
            h, w = f.shape[:2]
            roi = f[int(h * band[0]):int(h * band[1]),
                    int(w * xband[0]):int(w * xband[1])]
            samples.append(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3))
        lab = np.concatenate(samples).astype(np.float64)
        if len(lab) > max_pixels:
            lab = lab[rng.choice(len(lab), max_pixels, replace=False)]
        ab = lab[:, 1:]

        # 1. densest chromaticity bin
        H, xe, ye = np.histogram2d(ab[:, 0], ab[:, 1], bins=64,
                                   range=[[0, 255], [0, 255]])
        H = cv2.GaussianBlur(H, (5, 5), 0)
        pk = np.unravel_index(int(np.argmax(H)), H.shape)
        mean = np.array([(xe[pk[0]] + xe[pk[0] + 1]) / 2,
                         (ye[pk[1]] + ye[pk[1] + 1]) / 2])

        # 2. grow a Gaussian around it
        radius = 14.0
        support = np.linalg.norm(ab - mean, axis=1) <= radius
        cov = np.eye(2) * 9.0
        for _ in range(4):
            if support.sum() < 64:
                break
            mean = ab[support].mean(axis=0)
            cov = np.cov(ab[support].T) + np.eye(2) * 1e-3
            d = ab - mean
            P = np.linalg.inv(cov)
            m2 = (d[:, 0] ** 2 * P[0, 0] + 2 * d[:, 0] * d[:, 1] * P[0, 1]
                  + d[:, 1] ** 2 * P[1, 1])
            support = m2 <= 9.0                      # 3 sigma

        cov = cov + np.eye(2) * 1e-3                 # regularised
        L = lab[support, 0] if support.sum() >= 64 else lab[:, 0]
        return TurfModel(mean.tolist(), cov.tolist(),
                         float(np.percentile(L, 1)), float(np.percentile(L, 99)),
                         int(support.sum()))

    # ---- evaluation ------------------------------------------------------ #
    def mahalanobis(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel Mahalanobis distance of chromaticity from the turf model.

        The most expensive thing in the colour front end. It was being computed
        three times per frame before Shivam hoisted it onto the shared frame
        preparation; keep it computed once and passed around.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:].astype(np.float64)
        d = ab - np.asarray(self.mean)
        P = np.linalg.inv(np.asarray(self.cov))
        m = (d[..., 0] ** 2 * P[0, 0] + 2 * d[..., 0] * d[..., 1] * P[0, 1]
             + d[..., 1] ** 2 * P[1, 1])
        return np.sqrt(np.maximum(m, 0.0))

    def turf_mask(self, frame_bgr: np.ndarray, k: float = 3.0) -> np.ndarray:
        """Boolean mask of pixels whose chromaticity matches turf."""
        return self.mahalanobis(frame_bgr) < k

    # ---- io -------------------------------------------------------------- #
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "TurfModel":
        return TurfModel(**json.loads(s))


# --------------------------------------------------------------------------- #
# colour normalisation - WIRED IN AND INERT BY DEFAULT
# --------------------------------------------------------------------------- #
#: Where turf is put, in 8-bit Lab (a, b, L). Arbitrary but fixed: a reference
#: point near the middle of the three captures seen so far.
CANON_TURF = (110.0, 136.0, 120.0)


def normalise_to_canon(frame_bgr: np.ndarray, turf: "TurfModel",
                       turf_mask: np.ndarray | None = None,
                       canon=CANON_TURF) -> np.ndarray:
    """Map this video's turf colour onto a canonical one.

    **MEASURED TO WORK IN PRINCIPLE AND NOT PROMOTED - DO NOT ENABLE WITHOUT
    RE-RUNNING THE REPLICATES.** Shivam took this through three attempts and
    five seeds:

      * Leave-one-capture-out, held-out turf IoU 0.120 -> 0.889 (the normalised
        arm's WORST epoch beat the raw arm's BEST). The benefit is real.
      * But the production gate on the clips defeated it every time. Five
        replicate runs varying `--seed`: clip_04 line count median -10.3%,
        range -13.5 to -2.7, sd 4.6, against a 10% bar - 2 of 5 inside it.
        The honest statement is not "normalisation costs 12% of clip_04's
        lines" but "normalisation is a coin flip on clip_04", which is a
        stronger reason not to ship it than a consistent loss would be.
      * Seed 0 alone passed every criterion and was the best of five samples
        from a distribution whose median fails. Promoting on it would have
        shipped a lucky draw. **This is why one run is not a verdict.**

    Two details worth keeping. Chromaticity is a translation, so the turf-to-
    paint gap the net actually needs is preserved exactly. Luminance is NOT a
    translation: the captures differ in exposure as much as in hue (turf L of
    97, 47 and 24), and an additive shift large enough to align them drove
    near-white paint into the 255 ceiling - 28.9% of painted pixels clipped on
    one capture, 66.9% on another - after which the net trained on flat white
    where line structure should be and clip_04's lattice tipped 17 yd. The
    gamma curve below pins both endpoints, is monotone on [0, 255], and
    measured 0.0% clipping.

    Nothing here is hand-tuned to a video: every quantity comes from the turf
    model or from the frame. It runs only when a loaded checkpoint carries
    `normalise: True`.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] += canon[0] - float(turf.mean[0])
    lab[:, :, 2] += canon[1] - float(turf.mean[1])
    if turf_mask is None:
        turf_mask = turf.mahalanobis(frame_bgr) < 3.0
    if int(turf_mask.sum()) > 500:
        med = float(np.median(lab[:, :, 0][turf_mask]))
        med = min(max(med, 1.0), 254.0)
        gamma = np.log(canon[2] / 255.0) / np.log(med / 255.0)
        lab[:, :, 0] = 255.0 * np.power(lab[:, :, 0] / 255.0, gamma)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# marking channels
# --------------------------------------------------------------------------- #
def chroma(frame_bgr: np.ndarray) -> np.ndarray:
    """Distance from the neutral axis in Lab. Paint is low, turf and logos high."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)


def white_line_response(frame_bgr: np.ndarray, ksize: int = 11,
                        chroma_max: float = 26.0,
                        valid: np.ndarray | None = None,
                        pct: float = 99.5) -> np.ndarray:
    """Ridge response for thin bright *neutral* lines - the painted markings.

    A white top-hat on L isolates structures thinner than `ksize` that are
    brighter than their surroundings. Multiplying by a soft low-chroma gate
    suppresses the game's own drawn scrimmage/first-down lines and the team
    logos, which are handled separately by `coloured_line_masks`.

    NORMALISATION IS BY A HIGH PERCENTILE INSIDE THE FIELD REGION, NEVER BY THE
    GLOBAL MAX. One specular highlight - a chrome helmet - or an opaque HUD
    banner otherwise rescales the whole frame and drives the real field lines
    below any sensible threshold.

    Returns float32 in [0, 1]. Both parents use a top-hat here; Yash's
    `independent_paint` uses a wider kernel (25 px, wider than the widest
    painted line seen at ~20 px) on purpose, because it is a referee and must
    not share this module's blind spots.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    top = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, ker).astype(np.float32)

    pool = top[valid] if valid is not None and valid.any() else top
    scale = float(np.percentile(pool, pct))
    if scale < 1.0:                       # essentially no ridge structure
        scale = max(float(pool.max()), 1.0)
    top = np.clip(top / scale, 0.0, 1.0)

    gate = np.clip(1.0 - (chroma(frame_bgr) - chroma_max) / chroma_max, 0.0, 1.0)
    return (top * gate).astype(np.float32)


def coloured_line_masks(frame_bgr: np.ndarray, turf: TurfModel,
                        sat_min: float = 40.0,
                        off_turf: np.ndarray | None = None) -> dict:
    """Isolate the game-drawn scrimmage / first-down lines by hue.

    Blitz paints the line of scrimmage and the line to gain directly onto the
    turf in saturated colour. These are enormously useful - the HUD states the
    exact yard gap between them, which is the only absolute scale check either
    parent project has - so they are kept as their own channel rather than
    discarded as noise.

    They must ALSO be excluded from the white-paint pencil. Yash's
    `drop_ladder_impostors` records what happens otherwise: on clip_04 f161 the
    yellow first-down marker sat 18 px from a real yard line where every
    genuine 5-yard gap at that depth is ~100 px, the ladder came back
    (2,2,3,2,3,3,3,3,1,3) instead of all ones, and - because a pinhole camera
    viewing a plane has an exact scale degeneracy - the solve followed it to a
    camera 148 yd in the air at zero cost in residual.

    Returns a dict of boolean masks keyed by colour family.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1], hsv[:, :, 2]
    strong = (S > sat_min) & (V > 60)
    if off_turf is None:
        off_turf = turf.mahalanobis(frame_bgr) > 3.0

    def band(lo, hi):
        m = ((H >= lo) & (H <= hi)) if lo <= hi else ((H >= lo) | (H <= hi))
        return m & strong & off_turf

    return {
        "red": band(170, 10),        # wraps 0
        "orange_yellow": band(11, 34),
        "cyan": band(85, 100),
        "blue": band(101, 130),
    }
