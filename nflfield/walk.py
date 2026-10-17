"""Stage 6c - the walk: anchor a play, then track outward in both directions.

PROVENANCE
----------
The spine of Yash's `field_chain_tracker.py` and `auto_anchor.py`, with
Shivam's motion estimator in place of the ORB chain, over the observation cache
from `observe.py`.

WHAT CHANGED FROM STAGE 5, AND WHY
----------------------------------
Stage 5 walked forward from the first anchor-grade frame it met. Measured, that
cost real coverage for a structural reason rather than an evidential one:

    clip_02   frames 0-10 refused "no anchor yet", anchored at f11
    clip_04   frames 0-11 refused "no anchor yet", anchored at f12

Twelve of clip_04's fourteen refusals were that, and the frames were perfectly
solvable - the walk simply had not started yet. Yash does not pay this: his
anchors sit MID-CLIP (clip_01 f51, clip_02 f17, clip_03 f6, clip_04 f0) and he
tracks outward both ways.

So the walk now: picks an anchor per play, then runs the same stepping code
forward to the end and backward to the start. Backward steps use the INVERSE of
the stored forward motion rather than a separately estimated backward motion -
if the two were estimated independently they would not compose exactly, and the
two halves of the walk would disagree about where the anchor is.

HOW THE ANCHOR IS CHOSEN, AND WHY NOT BY RESULT
-----------------------------------------------
By per-frame anchor quality only: most lines, then lowest ladder residual,
among frames with >= `ANCHOR_MIN_LINES`, a tight ladder, and NO GAPS.

It is tempting to try several anchors and keep whichever yields the most
certified frames. **That would be circular** - it optimises the number being
reported - so it is not done. Yash's `auto_anchor` avoids the circularity
properly: it tries candidates and accepts one only if the whole play passes a
gate measured with `independent_paint`, a detector the solve never sees. We do
not have that detector yet, so the honest position is a deterministic
per-frame choice plus the standing warning that **a bootstrapped anchor's own
residual is not evidence of its depth scale** - two of Yash's scored 6+
distinct yard lines at sub-pixel error and were ~1.8x out. Only a whole-clip
measurement catches that, and it belongs with the referees.

THE TIERS
---------
    CERTIFIED  camera solved from this frame's paint, every gate passed
    TRACKED    pose carried through measured camera motion; NO camera claim,
               bounded by `MAX_TRACK_AGE` frames and stopped at a cut
    REFUSED    the evidence is not there; `reason` names the gate

CARRIED (ECC turf alignment from a certified keyframe, Yash's `carry_pass`) is
stage 7 and is still absent, so coverage here remains a floor.

**TRACKED IS NEVER CERTIFIED AND NEVER ADVANCES THE REFERENCE.** It is drawn
differently, counted separately, and a TRACKED frame cannot seed the pose that
a later CERTIFIED frame is measured against - otherwise a chain of predictions
would launder itself into a certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import camera, motion as MO, observe as OB, scalecheck
from .config import VideoSpec

#: Frames a TRACKED pose may be carried from the last certified one.
#:
#: Shivam's value, and his reasoning: prediction is for bridging a dropout - a
#: player crossing the lines, a frame of motion blur - which lasts a handful of
#: frames. At his old 45 it was also bridging the START OF A PLAY-SELECT
#: SCREEN: three quarters of a second of overlay drawn over a menu before the
#: state expired. Twelve frames is 0.2 s at 60 fps and still covers every real
#: dropout.
MAX_TRACK_AGE = 12


@dataclass
class FrameState:
    frame: int
    tier: str = "REFUSED"
    reason: str | None = None
    play: int = 0
    direction: str = "fwd"            # which half of the walk produced this
    n_lines: int = 0
    n_long: int = 0
    n_dropped: int = 0
    agree: bool | None = None
    resid_px: float | None = None
    jump_yd: float | None = None
    roll_disagree_deg: float | None = None
    line_dir_max_deg: float | None = None
    angle_spread_deg: float | None = None
    height_rate: float | None = None
    elev_rate_deg: float | None = None
    paint_angle_deg: float | None = None
    motion_label: str = "unknown"
    labelled_from_chain: bool = False
    chain_disagree_yd: float | None = None
    track_age: int = 0                # frames since the last certified pose
    params: list | None = None
    homography: list | None = None
    anchor: bool = False
    gap: int = 0
    gap_yd: float | None = None
    gap_found: tuple = ()

    @property
    def certified(self) -> bool:
        return self.tier == "CERTIFIED"


@dataclass
class WalkResult:
    clip: str
    n_frames: int = 0
    anchors: list = field(default_factory=list)
    rows: list = field(default_factory=list)

    def ordered(self) -> list:
        return sorted(self.rows, key=lambda r: r.frame)

    def counts(self) -> dict:
        c = {"CERTIFIED": 0, "TRACKED": 0, "REFUSED": 0}
        for r in self.rows:
            c[r.tier] = c.get(r.tier, 0) + 1
        return c

    def scale_summary(self) -> dict:
        """The absolute-scale referee, over CERTIFIED frames only.

        Only certified frames, on purpose: the number is meaningful only if the
        pose it was measured through passed every gate. A figure from a TRACKED
        pose would be a measurement of the motion estimator.
        """
        g = np.asarray([r.gap_yd for r in self.rows
                        if r.certified and r.gap_yd is not None])
        if not len(g):
            return {"n": 0, "median_yd": None, "iqr_yd": None}
        return {"n": int(len(g)),
                "median_yd": float(np.median(g)),
                "iqr_yd": (float(np.percentile(g, 75) - np.percentile(g, 25))
                           if len(g) > 3 else None),
                "p10_yd": float(np.percentile(g, 10)) if len(g) > 3 else None,
                "p90_yd": float(np.percentile(g, 90)) if len(g) > 3 else None}

    def reasons(self) -> dict:
        out = {}
        for r in self.rows:
            if r.tier == "CERTIFIED":
                continue
            msg = r.reason or "no reason recorded"
            key = msg
            for tok in ("residual", "moves model", "only", "a line the solve",
                        "camera height/elevation", "projected yard lines",
                        "solved pose disagrees", "yard assignment",
                        "no usable paint", "fit failed", "no camera state",
                        "too few lines", "no anchor", "disagrees with chained",
                        "proposers disagree", "track age", "not field"):
                if msg.startswith(tok):
                    key = tok
                    break
            key = "%s [%s]" % (key, r.tier) if r.tier == "TRACKED" else key
            out[key] = out.get(key, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# anchor choice
# --------------------------------------------------------------------------- #
def pick_anchor(obs: list, px_scale: float,
                min_lines: int = camera.ANCHOR_MIN_LINES,
                max_rms: float = camera.ANCHOR_MAX_LADDER_RMS_PX,
                anchor_at: str | None = None):
    """Index into `obs` of the best anchor-grade frame, or None.

    Deterministic and per-frame: most lines first, then lowest ladder
    residual, then earliest frame. Never scored by what the walk goes on to
    produce - see the module docstring.
    """
    grade = [k for k, ob in enumerate(obs)
             if ob.anchor_grade(px_scale, min_lines, max_rms)]
    if not grade:
        return None
    if anchor_at == "first":
        return grade[0]
    if anchor_at == "last":
        return grade[-1]
    best = None
    for k in grade:
        ob = obs[k]
        key = (-ob.n_lines, ob.ladder_rms, ob.frame)
        if best is None or key < best[0]:
            best = (key, k)
    return best[1]


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #
def _state_from(ob, play: int, direction: str) -> FrameState:
    return FrameState(frame=ob.frame, play=play, direction=direction,
                      n_lines=ob.n_lines, n_long=ob.n_long,
                      n_dropped=ob.n_dropped, agree=ob.agree,
                      paint_angle_deg=(None if ob.paint_angle_deg is None
                                       else round(ob.paint_angle_deg, 3)),
                      motion_label=ob.motion_label)


def _attach_gap(st: FrameState, H, ob, W: int) -> None:
    drawn = {k: np.asarray(v, float) for k, v in (ob.drawn or {}).items()}
    gap, found = scalecheck.gap_through_camera(H, drawn, W)
    st.gap_found = found
    st.gap_yd = None if gap is None else round(float(gap), 3)


def _step(obs: list, order: list, k0: int, base0, H0, pic, play: int,
          direction: str, lock_focal: bool, require_agree: bool,
          max_track_age: int, rows: dict, stride: int) -> None:
    """Walk one direction from the anchor. `order` is indices in walk order.

    `rows` is written in place, keyed by frame index, so the two directions
    merge without either having to know about the other.
    """
    W, Hgt = pic.w, pic.h
    ay = camera.aspect_ay(W, Hgt)
    base, H_ref = base0, H0
    last_cert_frame = obs[k0].frame
    track_H, track_age = None, 0
    #: Motion accumulated since the last certified frame. Resets to the
    #: certified pose on every certification, so it is never stale by more
    #: than the current refused run.
    chain_acc = None

    for pos, k in enumerate(order):
        ob = obs[k]
        st = _state_from(ob, play, direction)

        # The motion that carries the previous visited frame onto this one.
        # Forward: the motion stored on THIS frame (prev -> this).
        # Backward: the inverse of the motion stored on the frame we came
        # from, which in file order is the one after this.
        if direction == "fwd":
            M = OB.motion_matrix(ob)
        else:
            prev_k = order[pos - 1] if pos else k0
            M = MO.invert(OB.motion_matrix(obs[prev_k]))

        if not ob.lines or not len(ob.cum):
            st.reason = "too few lines (%d) for a ladder" % ob.n_lines
            track_H, track_age = None, 0
            rows[ob.frame] = st
            continue
        if require_agree and ob.agree is False:
            st.reason = "proposers disagree"
            rows[ob.frame] = st
            continue

        lines = [(float(y), float(s)) for y, s in ob.lines]
        cum = np.asarray(ob.cum, float)

        # THE CHAIN MUST ACCUMULATE ACROSS THE WHOLE GAP, NOT ONE FRAME.
        #
        # CAUGHT 2026-10-17 by the flow referee (docs/LESSONS). Written as
        # `chain_pose(H_ref, M)`, this applies ONE frame of motion to the
        # reference - and the reference is frozen on every refusal, so after a
        # refused run it is N frames stale while M covers only the last one.
        # The prediction is then wrong by N-1 frames of camera motion, the
        # correct solve lands far from it, and CHAIN_AGREE_MAX_YD refuses a
        # good pose.
        #
        # MEASURED on clip_01: six frames (f142, f144-148) that stage 5
        # certified were refused this way, and BOTH independent referees say
        # those poses were right - 24 of 24 flow pairs consistent, paint error
        # 0.082 yd median. I had credited the refusals to "the gates working".
        # They were my bug.
        #
        # `chain_acc` is the same accumulated pose the TRACKED tier propagates,
        # so the chain and the TRACKED pose cannot disagree about where the
        # camera is - they are one quantity with one update rule.
        chain_src = chain_acc if chain_acc is not None else H_ref
        H_chain = MO.chain_pose(chain_src, M) if M is not None else None

        gap_frames = max(1, abs(int(ob.frame) - int(last_cert_frame)))
        st.gap = gap_frames
        H, pnew, info = camera.solve_frame(
            H_ref, lines, ob.paint_angle_deg, W, Hgt, base,
            lock_focal=lock_focal, paint_support=ob.support, cum=cum,
            gap=gap_frames, ay=ay, H_chain=H_chain,
            chain_identity=H_chain is not None)

        st.resid_px = info.get("resid_px")
        st.jump_yd = info.get("jump_yd")
        st.roll_disagree_deg = info.get("roll_disagree_deg")
        st.line_dir_max_deg = info.get("line_dir_max_deg")
        st.angle_spread_deg = info.get("angle_spread_deg")
        st.height_rate = info.get("height_rate")
        st.elev_rate_deg = info.get("elev_rate_deg")
        st.labelled_from_chain = bool(info.get("labelled_from_chain"))
        st.chain_disagree_yd = info.get("chain_disagree_yd")

        if info.get("solve_ok"):
            st.tier = "CERTIFIED"
            st.params = [float(x) for x in pnew]
            st.homography = [float(x) for x in np.asarray(H).ravel()]
            _attach_gap(st, H, ob, W)
            base, H_ref = pnew, H
            last_cert_frame = ob.frame
            track_H, track_age = None, 0
            chain_acc = H          # fresh: the chain restarts from evidence
            rows[ob.frame] = st
            continue

        # ---- refused by the camera: can it be TRACKED? ------------------ #
        st.reason = info.get("reason")
        # The reference is FROZEN, not advanced. solve_frame already returned
        # the unchanged base; this is the explicit statement of it.
        base, H_ref = base, H_ref

        if ob.motion_label == "cut" or M is None:
            # Never track across a discontinuity, and never treat a missing
            # motion estimate as identity - that is a claim, not an absence.
            track_H, track_age, chain_acc = None, 0, None
            rows[ob.frame] = st
            continue
        prop = H_chain
        if prop is None:
            track_H, track_age, chain_acc = None, 0, None
            rows[ob.frame] = st
            continue
        # One accumulated pose serves both the chain prediction and the
        # TRACKED tier; see the note at `chain_src`.
        chain_acc = prop
        track_age += 1
        if track_age > max_track_age:
            st.reason = "track age %d > %d (%s)" % (track_age, max_track_age,
                                                    st.reason or "refused")
            track_H = None
            rows[ob.frame] = st
            continue
        track_H = prop
        st.tier = "TRACKED"
        st.track_age = track_age
        st.homography = [float(x) for x in np.asarray(prop).ravel()]
        # No `params`: a propagated pose has no solved parameters, and writing
        # the stale ones there would invite someone to read them as solved.
        rows[ob.frame] = st


def walk(spec: VideoSpec, stride: int = 1, lock_focal: bool = True,
         net=None, require_agree: bool = False, obs=None, pic=None,
         max_track_age: int = MAX_TRACK_AGE,
         anchor_at: str | None = None) -> WalkResult:
    """Observe (or reuse observations), then anchor and track every play.

    `anchor_at` overrides anchor selection, for experiments only:
    "first" / "last" take the earliest / latest anchor-grade frame in each
    play instead of the best one. It exists to separate walk DIRECTION from
    position in the clip, which the default (best-quality, usually mid-play)
    choice confounds - anchoring mid-play means "walked backward" and "early in
    the clip" are the same set of frames.
    """
    if obs is None or pic is None:
        obs, pic = OB.observe(spec, stride=stride, net=net)
    res = WalkResult(clip=spec.vid, n_frames=len(obs))
    rows: dict = {}

    plays = {}
    for k, ob in enumerate(obs):
        plays.setdefault(ob.play, []).append(k)

    W, Hgt = pic.w, pic.h
    ay = camera.aspect_ay(W, Hgt)

    for play, idxs in sorted(plays.items()):
        sub = [obs[k] for k in idxs]
        a_local = pick_anchor(sub, pic.px_scale, anchor_at=anchor_at)
        if a_local is None:
            for k in idxs:
                st = _state_from(obs[k], play, "none")
                st.reason = ("no anchor-grade frame in this play (best %d "
                             "lines)" % max((o.n_lines for o in sub),
                                            default=0))
                rows[obs[k].frame] = st
            continue

        k0 = idxs[a_local]
        ob0 = obs[k0]
        H0, p0, r0 = camera.bootstrap_pose(
            [(float(y), float(s)) for y, s in ob0.lines],
            np.asarray(ob0.cum, float), W, Hgt,
            lock_focal=lock_focal, ay=ay)
        if (H0 is None or r0 is None
                or r0 > camera.MAX_FIT_RESID_PX * pic.px_scale):
            for k in idxs:
                st = _state_from(obs[k], play, "none")
                st.reason = ("anchor candidate f%d failed to bootstrap "
                             "(resid %s)" % (ob0.frame,
                                             "-" if r0 is None else "%.2f" % r0))
                rows[obs[k].frame] = st
            continue

        st0 = _state_from(ob0, play, "anchor")
        st0.tier = "CERTIFIED"
        st0.anchor = True
        st0.resid_px = round(r0, 3)
        st0.params = [float(x) for x in p0]
        st0.homography = [float(x) for x in H0.ravel()]
        _attach_gap(st0, H0, ob0, W)
        rows[ob0.frame] = st0
        res.anchors.append({"play": play, "frame": int(ob0.frame),
                            "resid_px": round(r0, 3),
                            "n_lines": ob0.n_lines,
                            "ladder_rms": ob0.ladder_rms})

        fwd = idxs[a_local + 1:]
        back = list(reversed(idxs[:a_local]))
        for order, direction in ((fwd, "fwd"), (back, "back")):
            if order:
                _step(obs, order, k0, p0, H0, pic, play, direction,
                      lock_focal, require_agree, max_track_age, rows, stride)

    res.rows = list(rows.values())
    return res


def to_json(res: WalkResult) -> dict:
    return {"clip": res.clip,
            "n_frames": res.n_frames,
            "anchors": res.anchors,
            "counts": res.counts(),
            "refusal_reasons": res.reasons(),
            "scale": res.scale_summary(),
            "frames": [asdict(r) for r in res.ordered()]}
