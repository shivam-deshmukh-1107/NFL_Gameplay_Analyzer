"""Stage 2 - learn the static HUD overlay map.

PROVENANCE
----------
Ported from Shivam's `blitzfield/hudmask.py`. This replaces Yash's
`absolute_anchor.hud_mask_frame`, which finds the HUD by a fixed row fraction
(`_HUD_GRAPHIC_ROW_FRAC = 0.30`) plus a per-frame adjustment - a reasonable
heuristic, but one that assumes a layout, and the two game builds between the
parent projects do not share one.

Blitz draws its scorebug, turbo/credits banners and capture watermark at fixed
screen positions with pixel-identical artwork every frame. Two signals separate
that from the game world:

  * **persistence** - the pixel is off-turf in nearly every sampled frame;
  * **stability**   - its colour barely changes over time.

The playing surface fails the first test as soon as the camera moves, and the
out-of-bounds apron fails the second, so only the overlay survives both.

WHY THE MASK MATTERS MORE THAN IT LOOKS
---------------------------------------
Three separate failures in the parent projects trace to HUD pixels reaching a
stage that should not have seen them:

  1. Boundary tracing. The rectangular edge of a HUD banner is
     indistinguishable from a real field boundary, so without the mask a
     scorebug edge becomes a "sideline".
  2. The yard-line pencil. Yash's clip_01 f206 was certified on "4 lines" of
     which one was the flat bottom edge of a result banner (y=235, slope
     0.001) crossing the centre column exactly where yard 25 would be; the
     ladder named it 25, the fit bent the perspective to take it, and the
     median residual never saw it. That is what `MAX_LINE_DIR_DEG` now catches,
     but the mask is the cheaper fix.
  3. Motion estimation. The HUD never moves regardless of the camera, so
     tracked features on it drag the motion estimate toward zero. Worse, Yash's
     ORB chain had its HUD mask computed and never passed to
     `detectAndCompute`, so it matched the scoreboard and returned identity
     homographies - a chain of identities FREEZES rather than drifting, which
     is how "23 deg of silent rotation drift" came to be measured and then
     retracted.

Sampling is restricted to frames the gate called gameplay, and the stability
threshold is derived per video rather than fixed. Both are necessary rather
than incidental - see `learn`.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .config import VideoSpec
from .fieldcolor import TurfModel
from .ingest import FrameSource, Picture


def learn(spec: VideoSpec, pic: Picture, turf: TurfModel,
          n_samples: int = 160, persist_min: float = 0.85,
          std_max: float | None = None, std_pct: float = 25.0,
          std_bounds=(6.0, 30.0), min_area: int = 60,
          dilate: int = 3, segments=None) -> tuple[np.ndarray, dict]:
    """Return `(hud_mask, stats)`.

    `segments` restricts sampling to gameplay spans. THIS MATTERS: on a
    full-length capture, menus and stats screens redraw the whole frame, so a
    scorebug pixel swings wildly between play and menu and fails the stability
    test - which is why sampling the entire video found no HUD at all.
    """
    cap = cv2.VideoCapture(str(spec.path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    spans = [(s, e) for s, e in (segments or [(0, total)]) if e > s]
    if not spans:
        spans = [(0, total)]
    length = sum(e - s for s, e in spans)
    stride = max(1, length // max(1, n_samples))

    off_sum = l_sum = l_sq = None
    n = 0
    for s, e in spans:
        with FrameSource(spec, pic, stride=stride, start=s, stop=e) as src:
            for _, f in src:
                off = (~turf.turf_mask(f, 3.0)).astype(np.float32)
                L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                if off_sum is None:
                    off_sum = np.zeros_like(off)
                    l_sum = np.zeros_like(L)
                    l_sq = np.zeros_like(L)
                off_sum += off
                l_sum += L
                l_sq += L * L
                n += 1
    if n == 0:
        raise RuntimeError("no frames for " + spec.vid)

    persistence = off_sum / n
    mean = l_sum / n
    std = np.sqrt(np.maximum(l_sq / n - mean * mean, 0.0))

    # STABILITY IS JUDGED RELATIVE TO THIS SOURCE. A fixed cut does not
    # transfer: on one upload the overlay sits at std < 1 while on a more
    # heavily re-encoded one its 10th percentile is already 18, so an absolute
    # threshold of 14 finds a clean HUD in the first and nothing at all in the
    # second. Take instead the most stable quarter of the persistent pixels.
    hi = persistence >= persist_min
    if std_max is None:
        std_max = (float(np.clip(np.percentile(std[hi], std_pct), *std_bounds))
                   if hi.any() else std_bounds[0])
    hud = hi & (std <= std_max)
    hud = cv2.morphologyEx(hud.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # drop specks
    nl, lbl, stats_cc, _ = cv2.connectedComponentsWithStats(hud, 8)
    keep = stats_cc[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    hud = keep[lbl].astype(np.uint8)
    if dilate:
        hud = cv2.dilate(hud, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))
    hud = hud.astype(bool)

    stats = {
        "n_samples": n,
        "stride": stride,
        "restricted_to_gameplay": segments is not None,
        "hud_fraction": float(hud.mean()),
        "persist_min": persist_min,
        "std_max": float(std_max),
    }
    return hud, stats


def run(spec: VideoSpec, pic: Picture, turf: TurfModel, **kw) -> dict:
    hud, stats = learn(spec, pic, turf, **kw)
    d = spec.workdir / "stage2"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "hud_mask.png"), hud.astype(np.uint8) * 255)
    (d / "hud.json").write_text(json.dumps(stats, indent=2))
    return stats


def load(spec: VideoSpec) -> np.ndarray:
    p = spec.workdir / "stage2" / "hud_mask.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(
            "stage 2 has not been run for %s - run scripts/prepare.py %d"
            % (spec.vid, spec.order))
    return m > 127
