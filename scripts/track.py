"""Walk one or more clips: anchor each play, then track outward both ways.

    python scripts/track.py all
    python scripts/track.py 4 --stride=1
    python scripts/track.py 2 --no-track          # camera only, no TRACKED tier
    python scripts/track.py 1 --require-agree     # proposer disagreement refuses

Prints the tier counts, the absolute-scale check over certified frames, and -
for everything not certified - WHICH GATE was responsible. That breakdown is
the point: a coverage number alone says nothing about whether the refusals are
correct, and the gate histogram is what tells you where the next stage has to
work.

Writes work/<NN>_<slug>/stage6/walk.json.

**Read the tiers separately, never as one number.** CERTIFIED is a camera
solved from that frame's own paint with every gate passed. TRACKED is a pose
carried through measured motion with no camera claim at all. Adding them
together would report the second as if it were the first, which is the single
easiest way to make this pipeline look better than it is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, observe, walk as WALK      # noqa: E402

TRUTH = {"clip2_congestion": 30.0, "clip4_swing": 29.0, "hail_mary_left": 30.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--require-agree", action="store_true")
    ap.add_argument("--no-track", action="store_true",
                    help="disable the TRACKED tier (max_track_age = 0)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-observe instead of using the cache. REQUIRED "
                         "after any change to detection, the proposers or the "
                         "motion estimator: nothing versions the cache, so a "
                         "stale one silently measures the old code")
    ap.add_argument("--free-focal", action="store_true",
                    help="let the solve move the focal. NOT RECOMMENDED: it is "
                         "a game constant and a free focal absorbs other error")
    a = ap.parse_args()

    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])

    print("%-18s %6s %6s %7s %6s %7s %7s %8s %9s %6s"
          % ("clip", "frames", "cert", "cert%", "track", "track%", "refused",
             "resid", "scale err", "secs"))
    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        t0 = time.time()
        obs, pic, hit = observe.observe_cached(spec, stride=a.stride,
                                               refresh=a.refresh)
        t_obs = time.time() - t0
        res = WALK.walk(spec, stride=a.stride, lock_focal=not a.free_focal,
                        require_agree=a.require_agree, obs=obs, pic=pic,
                        max_track_age=0 if a.no_track else WALK.MAX_TRACK_AGE)
        secs = time.time() - t0

        c = res.counts()
        n = max(res.n_frames, 1)
        rs = [r.resid_px for r in res.rows if r.certified and r.resid_px is not None]
        s = res.scale_summary()
        truth = TRUTH.get(spec.vid)
        err = ("%+.2f" % (s["median_yd"] - truth)
               if (truth and s.get("median_yd")) else "-")
        print("%-18s %6d %6d %6.1f%% %6d %6.1f%% %7d %8s %9s %6.0f"
              % (spec.vid, res.n_frames, c["CERTIFIED"],
                 100.0 * c["CERTIFIED"] / n, c["TRACKED"],
                 100.0 * c["TRACKED"] / n, c["REFUSED"],
                 ("%.2f" % np.median(rs)) if rs else "-", err, secs))
        if res.anchors:
            print("        anchors: %s"
                  % ", ".join("play %d f%d (%d lines, rms %.2f)"
                              % (x["play"], x["frame"], x["n_lines"],
                                 x["ladder_rms"]) for x in res.anchors[:6]))
        print("        observe %.0fs of %.0fs%s; scale n=%s median %s IQR %s"
              % (t_obs, secs, " (cached)" if hit else "", s.get("n"),
                 ("%.2f" % s["median_yd"]) if s.get("median_yd") else "-",
                 ("%.2f" % s["iqr_yd"]) if s.get("iqr_yd") else "-"))
        for k, v in sorted(res.reasons().items(), key=lambda t: -t[1])[:8]:
            print("        %5d  %s" % (v, k))

        d = spec.workdir / "stage6"
        d.mkdir(parents=True, exist_ok=True)
        (d / "walk.json").write_text(json.dumps(WALK.to_json(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
