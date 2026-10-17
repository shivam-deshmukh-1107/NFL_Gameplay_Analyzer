"""Stage 1b - shot segmentation, gameplay gating and the per-frame field test.

PROVENANCE
----------
Ported from Shivam's `blitzfield/gating.py`, plus the per-frame "is this a
field" decision that lived in his `render.py`. This replaces Yash's
`field_chain_tracker.GREEN_MIN` (a flat 0.50 on masked green) and its
`DARK_LIT_MAX/MIN` band, and it feeds his play-segmentation
(`play_segments`): the merged pipeline splits plays wherever the gate says the
field is gone for `PLAY_BREAK_MIN_FRAMES`.

WHY GATE AT ALL
---------------
The long captures are mostly *not* gameplay: attract mode, team select,
replays, cutscenes and strategy screens. Roughly a quarter of the Midway
capture has no turf on screen at all and another quarter is the play-select
menu. Feeding those to the geometry stages produces confident nonsense - on one
frame Shivam's renderer drew 73 "yard lines" across a replay wipe, because a
failed frame still carries raw detections and on a wipe those are turf texture
and banner edges.

The gate is unsupervised. Per sampled frame, measure the fraction of pixels
matching the learned turf model; on a real capture that statistic is strongly
bimodal and a 2-component 1-D Gaussian mixture finds the split by itself.
Absolute floors keep it sane on inputs that are *all* gameplay - the four
hand-cut clips - where there is no split to find.

TRIED AND REJECTED: MASK FRAGMENTATION
--------------------------------------
Asking whether the turf is *in one piece* looked decisive and is recorded here
so nobody rebuilds it. On wide shots it separates cleanly - the menu grid
leaves 16-18 fragments with the largest holding 6-7% of the mask, against 2-3
fragments holding 57-97% for gameplay, an order of magnitude of clearance.

**But it was measured entirely on wide shots.** At close range a painted yard
line is tens of pixels across, so THE LINES THEMSELVES cut the field into
bands: zoomed gameplay scores 0.25 against menus reaching 0.54. The statistic
inverts exactly where it matters, and it cost 1,447 frames of the Midway
capture their highlight. It is not used. (MERGE_PLAN.md credited this test to
the gate in error; this is the correction.)

What is used instead is two independent kinds of evidence - turf coverage at or
above the measured valley, OR a solved lattice, since a lattice needs two yard
lines lying on turf and so proves a field by itself. That second clause is not
a nicety: without it, 441 frames had yard lines drawn across an unhighlighted
frame, which is a contradiction on its face.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .config import VideoSpec
from .fieldcolor import TurfModel
from .ingest import FrameSource, Picture

#: A frame with less turf than this is never gameplay; more than this always is.
#: Floors, not thresholds - they only bound where the mixture may put the split.
#:
#: MEASURED 2026-10-17, this project, all five sources: `turf_frac_lower` never
#: drops below 0.673 on any sampled frame of any of them, because all five are
#: entirely gameplay. (Yash measured the same thing a different way: every one
#: of the 873 frames in clips 1-4 has a masked green fraction >= 0.625.)
#:
#: TURF_CEIL EARNS ITS KEEP ON CLIP 4. There is no real split to find there, but
#: the dolly swings coverage from 0.673 to 0.941 and the mixture fits that as
#: two components, returning a threshold the clamp pulls back to 0.550. Zero
#: samples then fall below it, so the 100%-gameplay result is genuine and not
#: rescued by run smoothing - which is the thing worth checking, because
#: smoothing a wrong flag back to the right answer would hide the fault.
TURF_FLOOR = 0.18
TURF_CEIL = 0.55

#: Turf coverage at or above which a single frame is a playing surface.
#: MEASURED on the Midway capture (65,608 frames), where coverage is trimodal
#: and the valley is unambiguous:
#:     0.000-0.025  14,078 frames  no turf (attract, splash, crowd)  lattice 0.1%
#:     0.425-0.500  12,300 frames  the play-select menu              lattice 6.5%
#:     0.500-0.525     820 frames  the valley                        lattice 33%
#:     0.525 and up                the field                         lattice 46-69%
FIELD_COVERAGE_MIN = 0.52

#: Between this and TURF_CEIL with no lattice the answer is genuinely
#: unavailable - a corner-of-the-field view and turf behind a menu occupy the
#: same range - and those frames (3,418 on the Midway capture) are left
#: undecided rather than guessed.
FIELD_COVERAGE_UNSURE = 0.15

#: Consecutive non-field frames that end a play. Yash's value.
PLAY_BREAK_MIN_FRAMES = 8


@dataclass
class Segment:
    """A contiguous run of frames of one kind."""
    start: int
    stop: int          # exclusive, in source frame indices
    kind: str          # "gameplay" | "other"
    mean_turf: float


def frame_features(frame: np.ndarray, turf: TurfModel) -> dict:
    """Cheap per-frame descriptors used for gating and cut detection."""
    m = turf.turf_mask(frame, k=3.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    h = frame.shape[0]
    return {
        "turf_frac": float(m.mean()),
        "turf_frac_lower": float(m[int(h * 0.4):].mean()),
        "mean_v": float(gray.mean()),
        "edge_density": float((cv2.Canny(gray, 60, 160) > 0).mean()),
        "hist": hist.flatten(),
    }


def field_state(turf_frac: float, lattice_ok: bool) -> str:
    """Per-frame field decision: "field" | "unsure" | "no_field".

    Two independent kinds of evidence, because EXTENT AND GEOMETRY ARE SEPARATE
    CLAIMS. The turf mask stays reliable at zoom levels where the yard lattice
    is simply underdetermined - clip_01's end-of-play close-ups show a plain
    field with one yard line on it - so gating the field claim on the lattice
    throws away the claim that was still good.
    """
    if turf_frac >= FIELD_COVERAGE_MIN or lattice_ok:
        return "field"
    if turf_frac < FIELD_COVERAGE_UNSURE:
        return "no_field"
    return "unsure"


def _auto_threshold(values: np.ndarray) -> float:
    """Split a 1-D distribution into low/high modes, with guard rails.

    If the mixture is degenerate - one true mode, i.e. the whole video is
    gameplay, which is the case for all four clips - the components collapse
    together and we fall back to the floor rather than inventing a split.
    """
    v = values.reshape(-1, 1)
    if len(v) < 16 or np.ptp(values) < 0.05:
        return TURF_FLOOR
    gm = GaussianMixture(2, n_init=4, random_state=0).fit(v)
    lo, hi = np.sort(gm.means_.ravel())
    if hi - lo < 0.12:                      # poorly separated -> no real split
        return TURF_FLOOR
    grid = np.linspace(lo, hi, 256).reshape(-1, 1)
    p = gm.predict_proba(grid)
    lo_comp = int(np.argmin(gm.means_.ravel()))
    cross = int(np.argmin(np.abs(p[:, lo_comp] - 0.5)))
    return float(np.clip(grid[cross, 0], TURF_FLOOR, TURF_CEIL))


def _smooth_binary(flags: np.ndarray, min_run: int) -> np.ndarray:
    """Remove runs shorter than `min_run` so segments are usable shots."""
    out = flags.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_run and (i > 0 or j < n):
            out[i:j] = out[i - 1] if i > 0 else out[j]
        i = j
    return out


def run(spec: VideoSpec, pic: Picture, stride: int = 5,
        cut_threshold: float = 0.45, min_run_seconds: float = 0.5) -> dict:
    """Execute stage 1 and persist `stage1/turf_model.json` + `gating.json`."""
    # ---- pass 1: learn the turf model from a spread of frames ------------- #
    boot = []
    with FrameSource(spec, pic, stride=max(1, stride * 8)) as src:
        for _, f in src:
            boot.append(f.copy())
    if not boot:
        raise RuntimeError("no frames for " + spec.vid)
    turf = TurfModel.fit(boot)

    # ---- pass 2: features over the whole video ---------------------------- #
    idx, feats, hists = [], [], []
    fps = 30.0
    with FrameSource(spec, pic, stride=stride) as src:
        fps = src.fps or 30.0
        for i, f in src:
            ft = frame_features(f, turf)
            hists.append(ft.pop("hist"))
            idx.append(i)
            feats.append(ft)
    idx = np.asarray(idx)
    turf_frac = np.asarray([f["turf_frac_lower"] for f in feats])

    # ---- scene cuts ------------------------------------------------------- #
    hists = np.asarray(hists)
    d = np.zeros(len(hists))
    if len(hists) > 1:
        num = (hists[1:] * hists[:-1]).sum(1)
        den = np.linalg.norm(hists[1:], axis=1) * np.linalg.norm(hists[:-1], axis=1)
        d[1:] = 1.0 - num / np.maximum(den, 1e-9)
    cuts = idx[d > cut_threshold].tolist()

    # ---- gameplay decision ------------------------------------------------ #
    thr = _auto_threshold(turf_frac)
    flags = turf_frac >= thr
    min_run = max(1, int(round(min_run_seconds * fps / stride)))
    flags = _smooth_binary(flags, min_run)

    segments = []
    i = 0
    while i < len(flags):
        j = i
        while j < len(flags) and flags[j] == flags[i]:
            j += 1
        stop = int(idx[j - 1] + stride) if j < len(flags) else int(idx[-1] + stride)
        segments.append(Segment(int(idx[i]), stop,
                                "gameplay" if flags[i] else "other",
                                float(turf_frac[i:j].mean())))
        i = j

    play = [s for s in segments if s.kind == "gameplay"]
    total_play = sum(s.stop - s.start for s in play)

    d_out = spec.workdir / "stage1"
    d_out.mkdir(parents=True, exist_ok=True)
    (d_out / "turf_model.json").write_text(turf.to_json())
    payload = {
        "stride": stride,
        "fps": fps,
        "turf_threshold": thr,
        "turf_model_mean_ab": [round(float(x), 2) for x in turf.mean],
        "n_sampled": len(idx),
        "n_cuts": len(cuts),
        "cuts": cuts[:2000],
        "gameplay_frames": int(total_play),
        "gameplay_fraction": float(total_play / max(1, int(idx[-1]) + stride)),
        "n_segments": len(segments),
        "n_gameplay_segments": len(play),
        "segments": [asdict(s) for s in segments],
    }
    (d_out / "gating.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(d_out / "features.npz", idx=idx, turf_frac=turf_frac,
                        cut_dist=d,
                        mean_v=np.asarray([f["mean_v"] for f in feats]),
                        edge_density=np.asarray([f["edge_density"] for f in feats]))
    return payload


def load(spec: VideoSpec) -> tuple[TurfModel, dict]:
    """Read back stage 1: the turf model and the gating result."""
    d = spec.workdir / "stage1"
    turf = TurfModel.from_json((d / "turf_model.json").read_text())
    return turf, json.loads((d / "gating.json").read_text())


def gameplay_spans(gate: dict) -> list[tuple[int, int]]:
    """[(start, stop)] of the gameplay segments, in source frame indices."""
    return [(s["start"], s["stop"]) for s in gate["segments"]
            if s["kind"] == "gameplay"]
