"""Stage 6b - one decode pass per clip, producing per-frame observations.

WHY THIS EXISTS
---------------
Stage 5's walk decoded, detected and solved in one sequential pass. That is
fine for a forward-only walk and impossible for anything else, and three things
now need the frames in a different order or more than once:

  1. **The bidirectional walk.** An anchor in the middle of a play has to be
     walked outward in both directions, so frames before it are visited after
     frames after it.
  2. **Anchor selection.** Choosing a good anchor means looking at candidates
     across the play BEFORE committing to one, which cannot be done while
     walking past them.
  3. **Judging a gate change.** Re-running a walk with one threshold moved
     should not cost another full detection pass. Both parent projects judge
     every change against a frozen baseline, and at ~0.5 s/frame a re-walk of
     five clips is nine minutes of recomputing identical detections.

So detection is separated from tracking. An `Observation` is everything the
walk needs about a frame and nothing about any pose: the lines, the ladder, the
paint direction, the motion to the previous frame, and the drawn-line segments.
It is small - a few hundred floats - so a whole clip's observations sit in
memory and can be cached to disk.

**NOTHING HERE DEPENDS ON A POSE.** That is the property that makes the cache
sound: no observation can change because a walk went differently, so a re-walk
on cached observations is exactly a re-walk, not an approximation of one. The
one thing deliberately left out is the drawn-line GAP, which needs a pose to
measure; the drawn-line segments are stored instead and the gap is computed
later from whatever pose the walk certifies.

Motion is estimated in frame order during this pass and stored per frame as
"previous -> this". A backward walk inverts it rather than re-estimating, so
the two directions compose exactly and cannot disagree about where the anchor
is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import (drawnlines, gating, hudmask, ingest, markings, motion as MO,
               propose)
from .config import VideoSpec


@dataclass
class Observation:
    """What one frame says, independently of any camera pose."""

    frame: int
    #: (y at centre column, slope) per surviving line, sorted by y.
    lines: list = field(default_factory=list)
    #: Cumulative 5-yard rung count per line, from the ladder proposer.
    cum: list = field(default_factory=list)
    ladder_rms: float | None = None
    lattice_score: float | None = None
    agree: bool | None = None
    n_long: int = 0
    n_dropped: int = 0
    #: Length-weighted dominant paint angle, and where it was measured.
    paint_angle_deg: float | None = None
    support_mid: list = field(default_factory=list)
    support_wt: list = field(default_factory=list)
    #: Motion from the PREVIOUS observed frame to this one, or None.
    motion_H: list | None = None
    motion_label: str = "unknown"
    motion_speed: float = 0.0
    #: Game-drawn line segments by colour family, for the scale referee.
    drawn: dict = field(default_factory=dict)
    turf_frac: float = 0.0
    field_state: str = "field"
    #: Index of the play this frame belongs to. A play boundary is a real
    #: discontinuity - a gating span edge or a detected cut - and the walk must
    #: not track across one, so it is recorded here rather than re-derived.
    #: Re-deriving it from `motion_H is None` would be wrong: estimation also
    #: fails mid-play when too few features survive, which is not a boundary.
    play: int = 0

    @property
    def support(self):
        if not len(self.support_mid):
            return None
        return (np.asarray(self.support_mid, float),
                np.asarray(self.support_wt, float))

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def anchor_grade(self, px_scale: float, min_lines: int,
                     max_rms: float) -> bool:
        """Is this frame fit to cold-start a camera?

        Three conditions, all Yash's: enough lines to be over-determined, a
        ladder that fits the 1-D projective model tightly, and NO GAPS - every
        rung exactly one. The last is what keeps a bootstrapped anchor from
        inheriting a doubled depth scale, because a ladder with a 2-step gap is
        consistent with a family at twice the spacing.
        """
        if len(self.lines) < min_lines or self.ladder_rms is None:
            return False
        if self.ladder_rms > max_rms * px_scale:
            return False
        steps = np.diff(np.asarray(self.cum, float))
        return bool(len(steps)) and bool(np.all(steps == 1))


def observe(spec: VideoSpec, stride: int = 1, net=None) -> tuple[list, object]:
    """One decode pass. Returns `(observations, picture)`.

    Observations are in frame order, and `motion_H` on each is the motion from
    the PREVIOUS entry in this list - which is the previous *observed* frame,
    so a stride above 1 means the motion spans that stride. The walk is told
    the stride so it can reason about rates per real frame.
    """
    payload, pic = ingest.load(spec)
    turf, gate = gating.load(spec)
    hud = hudmask.load(spec)
    spans = gating.gameplay_spans(gate) or [(0, payload["meta"]["nb_frames"] or 0)]

    est = MO.MotionEstimator(picture_w=pic.w)
    rng = np.random.default_rng(0)
    out = []
    play = -1

    for lo, hi in spans:
        play += 1
        # A span boundary is a discontinuity in the footage, so the motion
        # estimator starts fresh rather than reporting a jump across it.
        est.reset()
        with ingest.FrameSource(spec, pic, stride=stride, start=lo, stop=hi) as src:
            for i, fr in src:
                prep = markings.FramePrep.build(fr, turf, hud=hud, net=net)
                fm = markings.analyse_frame(fr, turf, hud=hud, frame_index=i,
                                            rng=rng, prep=prep)
                pencil = fm.pencil
                ang, support = markings.paint_direction(pencil)
                p = propose.compare(pencil, pic.w)
                mo = est.update(fr, prep.region)
                drawn = drawnlines.fit_drawn_lines(fr, turf, prep=prep,
                                                   vp_cross=fm.vp_cross)
                tf = float(prep.turf_mask.mean())

                ob = Observation(
                    frame=int(i),
                    lines=[(float(y), float(s)) for y, s in zip(p.s, p.slopes)],
                    cum=([float(x) for x in p.lad[1]] if p.lad is not None else []),
                    ladder_rms=(float(p.lad[2]) if p.lad is not None else None),
                    lattice_score=(float(p.lat.score) if p.lat is not None else None),
                    agree=p.agree,
                    n_long=int(p.n_long),
                    n_dropped=int(p.n_dropped),
                    paint_angle_deg=(None if ang is None else float(ang)),
                    support_mid=([] if support is None
                                 else support[0].tolist()),
                    support_wt=([] if support is None else support[1].tolist()),
                    motion_H=(None if mo.H is None
                              else np.asarray(mo.H, float).ravel().tolist()),
                    motion_label=mo.label,
                    motion_speed=float(mo.speed),
                    drawn={k: [float(x) for x in v] for k, v in drawn.items()},
                    turf_frac=tf,
                    field_state=gating.field_state(tf, p.lat is not None),
                    play=play,
                )
                # A hard cut inside a gating span is also a boundary: the
                # motion estimator reports it, and tracking across it would
                # carry a pose over discontinuous footage.
                if mo.label == "cut" and out:
                    play += 1
                    ob.play = play
                    ob.motion_H = None
                out.append(ob)
    return out, pic


def motion_matrix(ob: Observation):
    """`ob.motion_H` as a 3x3, or None."""
    if ob.motion_H is None:
        return None
    H = np.asarray(ob.motion_H, float).reshape(3, 3)
    return H if np.all(np.isfinite(H)) else None


# --------------------------------------------------------------------------- #
# persistence - what makes the cache worth having
# --------------------------------------------------------------------------- #
#: Observations are written gzipped because the support arrays dominate the
#: size and compress well (clip_04: ~2 MB raw, ~400 kB gzipped).
import gzip                                             # noqa: E402
import json                                             # noqa: E402
from dataclasses import asdict                          # noqa: E402


def cache_path(spec: VideoSpec):
    return spec.workdir / "stage6" / "observations.json.gz"


def save(spec: VideoSpec, obs: list, stride: int) -> None:
    """Persist observations so a re-walk costs nothing but the walk.

    IN-MEMORY CACHING IS NOT ENOUGH, and that was an oversight worth recording:
    the whole argument for separating detection from tracking is that re-walking
    with a threshold moved should be cheap, and it is - 8 s against 94 s of
    detection on clip_02 - but only WITHIN one process. A fresh process re-ran
    the full detection pass, so the first experiment that needed two walks of
    the same clip would have paid for detection twice.
    """
    p = cache_path(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clip": spec.vid, "stride": stride,
               "n": len(obs), "obs": [asdict(o) for o in obs]}
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def load(spec: VideoSpec, stride: int = 1):
    """Read cached observations, or None if absent or for a different stride."""
    p = cache_path(spec)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:                                    # noqa: BLE001
        return None
    if payload.get("stride") != stride:
        return None
    out = []
    for d in payload["obs"]:
        d["drawn"] = {k: list(v) for k, v in (d.get("drawn") or {}).items()}
        out.append(Observation(**d))
    return out


def observe_cached(spec: VideoSpec, stride: int = 1, net=None,
                   refresh: bool = False):
    """`observe`, via the on-disk cache when one matches.

    `refresh` forces a re-observation, which is what to use after any change to
    detection, the proposers or the motion estimator - the cache is only valid
    for the code that produced it, and nothing here versions that. A stale
    cache after a detector change would silently measure the old detector.
    """
    _, pic = ingest.load(spec)
    if not refresh:
        hit = load(spec, stride)
        if hit is not None:
            return hit, pic, True
    obs, pic = observe(spec, stride=stride, net=net)
    save(spec, obs, stride)
    return obs, pic, False
