"""Stage 0 - find the real picture inside the capture, and pin the transform.

PROVENANCE
----------
Ported from Shivam's `blitzfield/ingest.py` (active-picture detection from
sampled frames, invertible transform, strided frame source), with Yash's
`content_box.py` reasoning folded in and two changes:

  * The analysis frame is now the picture at NATIVE resolution, not a fixed
    960x720 resize - see config.py, "THE PICTURE FRAME".
  * `picture_to_source` therefore carries only a translation (the crop offset),
    so it is exact rather than a resampling, and `anisotropy` is recorded for
    the camera model to absorb as (fx, fy) instead of being a reason to exclude
    a source.

WHY THIS STAGE IS NOT OPTIONAL
------------------------------
Two things broke on pillarbox bars in Yash's project and neither announced
itself (content_box.py):

  1. The green field-visibility gate failed 213 of clip_05's 221 frames,
     because the bars are not green and dragged the measured green fraction
     from 0.631 down to 0.449 - a gate meant to reject menu screens was
     rejecting gameplay.
  2. `clip_focal` was handed the 1920 container width instead of the 1443
     picture width and returned 2139 px instead of 1608 px, a 33% error
     straight into every cross-field number.

So: **no module opens a video directly.** Everything goes through `FrameSource`,
which hands out picture-frame crops and nothing else.

DELIBERATELY CONSERVATIVE. On a capture with no bars this returns the full
frame and the pipeline is identical to having no crop at all. A wrong crop is
far worse than no crop - it silently moves the principal point and rescales the
focal - so the bars must be dark across the WHOLE sampled video before anything
is cropped, and a crop that would remove more than `MAX_CROP_FRAC` of either
axis is refused as implausible.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import VideoSpec, focal_px, px_scale

#: A crop may not remove more than this fraction of either axis. A 16:9
#: container holding 4:3 content loses 25% of its width; 2.35:1 would lose 43%.
#: Past half the axis we are not looking at pillarbox any more.
MAX_CROP_FRAC = 0.50


# --------------------------------------------------------------------------- #
# container metadata
# --------------------------------------------------------------------------- #
def ffprobe(path: Path) -> dict:
    """Video stream key fields, as plain Python types.

    ffprobe rather than OpenCV because OpenCV's `CAP_PROP_FRAME_COUNT` is a
    container estimate on some of these files, and the real frame count matters
    for the sampling below.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,"
        "codec_name,pix_fmt,sample_aspect_ratio,display_aspect_ratio",
        "-of", "json", str(path),
    ]
    raw = json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                    check=True).stdout)
    s = raw["streams"][0]

    def _rate(v):
        if not v or "/" not in str(v):
            return float(v) if v else 0.0
        n, d = str(v).split("/")
        return float(n) / float(d) if float(d) else 0.0

    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": _rate(s.get("r_frame_rate")),
        "avg_fps": _rate(s.get("avg_frame_rate")),
        "nb_frames": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "duration": float(s["duration"]) if s.get("duration") else None,
        "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
        "sar": s.get("sample_aspect_ratio"),
        "dar": s.get("display_aspect_ratio"),
    }


# --------------------------------------------------------------------------- #
# the active picture
# --------------------------------------------------------------------------- #
@dataclass
class Picture:
    """The active picture inside the capture, and what follows from its size."""

    x: int
    y: int
    w: int
    h: int
    aspect: float            # w / h
    anisotropy: float        # aspect / (4/3); 1.0 == an undistorted 4:3 render
    cropped: bool            # False when the whole frame is the picture
    n_samples: int
    luma_threshold: int
    refused_crop: str | None = None   # why a detected crop was not applied

    @property
    def px_scale(self) -> float:
        """Scale for pixel constants measured at `config.PX_REF_W`."""
        return px_scale(self.w)

    @property
    def focal(self) -> tuple[float, float]:
        """(fx, fy) in this picture's own pixels."""
        return focal_px(self.w, self.h)

    def to_source(self) -> np.ndarray:
        """3x3 picture-frame pixel -> source-frame pixel. A pure translation."""
        return np.array([[1.0, 0.0, float(self.x)],
                         [0.0, 1.0, float(self.y)],
                         [0.0, 0.0, 1.0]], np.float64)

    def from_source(self) -> np.ndarray:
        return np.array([[1.0, 0.0, -float(self.x)],
                         [0.0, 1.0, -float(self.y)],
                         [0.0, 0.0, 1.0]], np.float64)


def detect_picture(path: Path, n_samples: int = 64, luma_threshold: int = 18,
                   coverage: float = 0.004) -> Picture:
    """Locate the non-black picture region.

    The per-pixel MAXIMUM luma over frames spread across the whole video, not a
    single frame: any row or column belonging to the picture carries bright
    content in at least one sample, while true padding stays black in all of
    them. A single frame can be a fade, a dark menu or a black cut, and on this
    footage all three occur.

    `coverage` guards against codec noise in the bars - a row or column needs
    this fraction of its pixels above threshold before it counts as picture.

    THIS MEASURES BLACK BARS AND NOTHING ELSE. It has no opinion about turf,
    paint or cameras, and it cannot help a capture that is merely dark: a dark
    field still has bright paint on it, so its columns are not black. (Yash's
    clip_07 fades its lower field to a median V of 2/255 and is still not
    cropped by this, correctly.)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open " + str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip the outer 5%: intros and outros are often full black, and a sample
    # that is black everywhere contributes nothing to a per-pixel maximum.
    lo, hi = int(total * 0.05), int(total * 0.95)
    if hi <= lo:
        lo, hi = 0, max(total - 1, 0)
    idxs = np.unique(np.linspace(lo, hi, max(n_samples, 1)).astype(int))

    acc = np.zeros((H, W), np.uint8)
    got = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        np.maximum(acc, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), out=acc)
        got += 1
    cap.release()
    if got == 0:
        raise RuntimeError("no frames decoded from " + str(path))

    hot = acc > luma_threshold
    row_ok = hot.mean(axis=1) > coverage
    col_ok = hot.mean(axis=0) > coverage

    refused = None
    if not row_ok.any() or not col_ok.any():
        x, y, w, h = 0, 0, W, H
        refused = "no row or column above the luma threshold"
    else:
        ys, xs = np.flatnonzero(row_ok), np.flatnonzero(col_ok)
        x, y = int(xs[0]), int(ys[0])
        w, h = int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)
        # Refuse an implausible crop rather than apply it. Getting this wrong
        # moves the principal point and rescales the focal, in silence.
        if w < (1.0 - MAX_CROP_FRAC) * W or h < (1.0 - MAX_CROP_FRAC) * H:
            refused = ("crop %dx%d would remove more than %.0f%% of an axis"
                       % (w, h, 100 * MAX_CROP_FRAC))
            x, y, w, h = 0, 0, W, H

    aspect = w / h
    return Picture(x=x, y=y, w=w, h=h, aspect=aspect,
                   anisotropy=aspect / (4.0 / 3.0),
                   cropped=(w != W or h != H),
                   n_samples=got, luma_threshold=luma_threshold,
                   refused_crop=refused)


# --------------------------------------------------------------------------- #
# frame iteration
# --------------------------------------------------------------------------- #
class FrameSource:
    """Iterate a video, yielding picture-frame BGR crops.

    The crop is a view into the decoded frame, so it costs nothing beyond the
    decode. Callers that mutate the frame must copy it.
    """

    SEEK_THRESHOLD = 12
    """Stride above which seeking beats decoding every frame.

    Measured by Shivam on this footage: sequential decode runs at ~220 fps
    while a seek plus read costs about as much as decoding 10 frames. Below the
    threshold sequential wins; above it, seeking avoids decoding frames that
    are discarded anyway.
    """

    def __init__(self, spec: VideoSpec, pic: Picture, stride: int = 1,
                 start: int = 0, stop: int | None = None,
                 seek: bool | None = None):
        self.spec, self.pic, self.stride = spec, pic, stride
        self.start, self.stop = start, stop
        self.seek = (stride >= self.SEEK_THRESHOLD) if seek is None else seek
        self.cap = cv2.VideoCapture(str(spec.path))
        if not self.cap.isOpened():
            raise RuntimeError("cannot open " + str(spec.path))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cap.release()

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        p = self.pic
        return frame[p.y:p.y + p.h, p.x:p.x + p.w]

    def __iter__(self):
        stop = self.stop if self.stop is not None else self.total
        if self.seek:
            for i in range(self.start, stop, self.stride):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield i, self._crop(frame)
            return
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        i = self.start
        while i < stop:
            ok, frame = self.cap.read()
            if not ok:
                break
            if (i - self.start) % self.stride == 0:
                yield i, self._crop(frame)
            i += 1


def open_clip(spec: VideoSpec, **kw) -> FrameSource:
    """The only sanctioned way to read frames. Loads stage 0 from disk."""
    _, pic = load(spec)
    return FrameSource(spec, pic, **kw)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def load(spec: VideoSpec) -> tuple[dict, Picture]:
    """Read back a persisted stage 0 result."""
    p = spec.workdir / "stage0" / "source.json"
    if not p.exists():
        raise FileNotFoundError(
            "stage 0 has not been run for %s - run scripts/prepare.py %d"
            % (spec.vid, spec.order))
    payload = json.loads(p.read_text())
    return payload, Picture(**payload["picture"])


def run(spec: VideoSpec, n_samples: int = 64) -> dict:
    """Execute stage 0 for one video and persist `stage0/source.json`."""
    meta = ffprobe(spec.path)
    pic = detect_picture(spec.path, n_samples=n_samples)
    fx, fy = pic.focal

    d = spec.workdir / "stage0"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": {"order": spec.order, "vid": spec.vid,
                  "filename": spec.filename, "note": spec.note,
                  "build": spec.build},
        "meta": meta,
        "picture": asdict(pic),
        "derived": {
            # Written down so a wrong one is visible in the artifact rather than
            # only in a number three stages downstream.
            "px_scale": pic.px_scale,
            "focal_x_px": fx,
            "focal_y_px": fy,
        },
        "picture_to_source": pic.to_source().tolist(),
        "source_to_picture": pic.from_source().tolist(),
    }
    (d / "source.json").write_text(json.dumps(payload, indent=2))
    return payload
