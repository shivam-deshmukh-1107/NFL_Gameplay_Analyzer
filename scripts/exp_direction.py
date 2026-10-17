"""Does the WALK DIRECTION bias the absolute scale, or only the clip position?

    python scripts/exp_direction.py 4 2

THE QUESTION. With the default anchor (best quality, usually mid-play), frames
walked BACKWARD are also the EARLY frames of the clip, so the two cannot be
told apart. Measured that way, backward frames read +0.19 yd (clip_04) and
+0.25 yd (clip_02) above forward frames - which is either a real directional
defect in the backward walk, or an ordinary variation along the clip that the
anchor happens to split.

THE SEPARATION. Anchor at the LAST anchor-grade frame, so almost every frame is
reached walking backward; then anchor at the FIRST, so almost every frame is
reached walking forward. Compare the two on THE SAME FRAMES. If direction is
the cause the medians differ; if position is the cause they agree.

Written before running, as both parent projects insist: a prediction after the
fact is not a prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, observe, walk as WALK      # noqa: E402

TRUTH = {"clip2_congestion": 30.0, "clip4_swing": 29.0, "hail_mary_left": 30.0}


def gaps(res):
    return {r.frame: r.gap_yd for r in res.rows
            if r.certified and r.gap_yd is not None}


def main(argv):
    for key in (argv or ["4", "2"]):
        spec = config.get_video(key)
        truth = TRUTH.get(spec.vid)
        obs, pic, hit = observe.observe_cached(spec, stride=1)
        print("=== %s  (%s observations, HUD %s)"
              % (spec.vid, "cached" if hit else "fresh",
                 truth if truth else "no truth"))

        out = {}
        for mode in ("last", "first"):
            res = WALK.walk(spec, obs=obs, pic=pic, anchor_at=mode)
            anc = res.anchors[0]["frame"] if res.anchors else None
            out[mode] = (gaps(res), anc, res.counts())
            print("  anchor_at=%-5s anchor f%-4s certified %d  measured %d"
                  % (mode, anc, res.counts()["CERTIFIED"], len(out[mode][0])))

        g_back, a_back, _ = out["last"]
        g_fwd, a_fwd, _ = out["first"]
        shared = sorted(set(g_back) & set(g_fwd))
        if len(shared) < 6:
            print("  only %d shared measured frames; inconclusive" % len(shared))
            continue
        b = np.array([g_back[f] for f in shared])
        f = np.array([g_fwd[f] for f in shared])
        print("  %d frames measured under BOTH anchors" % len(shared))
        print("    anchored last  (walked backward): median %.3f" % np.median(b))
        print("    anchored first (walked forward):  median %.3f" % np.median(f))
        print("    difference (backward - forward):  %+.3f yd" % (np.median(b) - np.median(f)))
        print("    per-frame |difference|: med %.3f  p90 %.3f  max %.3f"
              % (np.median(np.abs(b - f)), np.percentile(np.abs(b - f), 90),
                 np.abs(b - f).max()))
        if truth:
            print("    error vs HUD: backward %+.3f   forward %+.3f"
                  % (np.median(b) - truth, np.median(f) - truth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
