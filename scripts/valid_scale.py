"""Absolute-scale referee: the measured yard gap vs the gap the game prints.

    python scripts/valid_scale.py 2 4                # the two clips with truth
    python scripts/valid_scale.py all --stride=4
    python scripts/valid_scale.py 2 4 --tol=0.35

THIS IS THE PROMOTION GATE. No change to the front end or the geometry ships
unless the error here does not move backwards.

Expect |error| <= 0.25 yd. **Read the IQR as well as the median**: a wide
spread means the scale is being recovered differently on different frames even
when the middle lands in the right place.

ONE RUN IS NOT A VERDICT. Shivam measured clip 4's line count at sd 4.6% across
IDENTICAL configurations, and a single run of a configuration that was
ultimately rejected passed every criterion. Replicate before concluding.

Writes work/<NN>_<slug>/stage4/scale.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, scalecheck      # noqa: E402


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()

#: Down-and-distance read off the HUD BY HAND, once, per capture.
#:
#: These are the only hand-entered numbers in the project, and they are here on
#: purpose: the automated geometry needs something external to be wrong
#: against. They are deliberately NOT read by `hudtext`, because that reader's
#: digit alphabet is taught by this very distance measurement - using it here
#: would make the check agree with itself by construction. See hudtext.py's
#: docstring, "THE CIRCULARITY".
#:
#: Sources: clip2 and clip4 are the two values both parent projects quote and
#: validate against. hail_mary_left is Shivam's entry for that capture.
TRUTH = {
    "clip2_congestion": 30.0,
    "clip4_swing": 29.0,
    "hail_mary_left": 30.0,
    # clip_03 IS DELIBERATELY ABSENT, and the reason is measured, not assumed.
    #
    # Shivam's out/field_template.json records hud_truth 29 for it, with his
    # measured 31.17 (IQR 4.34) and the note "tight pre-snap view; coloured
    # lines partly off-frame". We measure 31.11 at IQR 0.28 - the same median,
    # a far tighter spread - so against a truth of 29 it reads +2.11 yd and
    # looks like the one clip that fails this gate.
    #
    # IT IS THE TRUTH VALUE THAT IS WRONG. Measured 2026-10-17 by learning the
    # HUD alphabet from clip_04's geometry ONLY and applying it to clip_03's
    # banner text, so clip_03's own geometry never taught the book:
    #
    #   clip_04 distance glyphs -> clusters (0, 5) -> "29"   (matches its truth)
    #   clip_03 distance glyphs -> clusters (3, 4) on all 10 text lines
    #
    # Two entirely different glyph clusters, consistently. **clip_03 does not
    # print 29.** Its measured 31.11 is consistent with a printed 31, i.e. an
    # error near 0.1 yd, in line with clip_02 and clip_04 - but '3' and '1'
    # cannot be learned from this footage, so the reading cannot be completed
    # and the clip CANNOT be graded. Grading it against 29 would report a
    # pipeline failure that is really a bookkeeping error.
    #
    # Add it back only with a digit book that can read it. The cross-clip
    # transfer above is the mechanism; it needs the long captures for coverage.
    #
    # The frame-edge hypothesis this was first attributed to is DISPROVED: on
    # f35 the orange_yellow line touches no frame edge and the gap is still
    # 30.88, and the gap holds at 30.80-30.91 while the red blob's length jumps
    # 1177 -> 295 px, exactly as an SVD line fit should behave under clipping.
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"],
                    help="processing order 1-5, or slug, or 'all'")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.25,
                    help="|error| bar in yards for the verdict (default 0.25, "
                         "the standing expectation; the bar is printed with "
                         "the verdict so a looser agreed bar is visible)")
    ap.add_argument("--set-baseline", action="store_true",
                    help="record this run as the baseline to compare against")
    a = ap.parse_args()

    base_path = config.WORK / "scale_baseline.json"
    baseline = (json.loads(base_path.read_text())
                if base_path.exists() else None)

    summaries = {}
    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])

    print("%-18s %6s %6s %9s %7s %9s %8s %8s"
          % ("clip", "frames", "meas", "median", "IQR", "HUD says", "err", "p90|e|"))
    worst, graded = 0.0, 0
    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        res, _, _ = scalecheck.measure(spec, stride=a.stride)
        truth = TRUTH.get(spec.vid)
        s = res.summary(truth)

        d = spec.workdir / "stage4"
        d.mkdir(parents=True, exist_ok=True)
        (d / "scale.json").write_text(json.dumps(
            {**s, "frames": [vars(r) for r in res.rows]}, indent=1, default=str))

        if not s["n_measured"]:
            print("%-18s %6d %6d   no frame yielded both drawn lines"
                  % (spec.vid, s["n_frames"], 0))
            continue
        line = ("%-18s %6d %6d %8.2f %7s %9s %8s %8s"
                % (spec.vid, s["n_frames"], s["n_measured"], s["median_yd"],
                   ("%.2f" % s["iqr_yd"]) if s["iqr_yd"] is not None else "-",
                   ("%.0f" % truth) if truth else "-",
                   ("%+.2f" % s["error_yd"]) if truth else "-",
                   ("%.2f" % s["abs_error_p90_yd"]) if truth else "-"))
        print(line)
        if truth is not None:
            worst = max(worst, abs(s["error_yd"]))
            graded += 1
            summaries[spec.vid] = s

    if not graded:
        print("\nno clip had hand-entered truth; nothing graded")
        return 0

    # Against the BAR.
    print("\nworst |error| %.2f yd over %d graded clip(s)  vs bar %.2f  ->  %s"
          % (worst, graded, a.tol,
             "within bar" if worst <= a.tol else "ABOVE BAR"))

    # Against the BASELINE, which is a different question. On a first run there
    # is nothing to regress from, and calling that a regression would be
    # misleading: the word has to mean "worse than it was".
    rc = 0
    if baseline is None:
        print("no stored baseline: this run is the first measurement.")
        print("record it with --set-baseline.")
    else:
        print("vs baseline recorded %s:" % baseline.get("recorded", "?"))
        for vid, s in sorted(summaries.items()):
            b = baseline.get("clips", {}).get(vid)
            if b is None or b.get("error_yd") is None:
                print("  %-18s no baseline entry" % vid)
                continue
            d = abs(s["error_yd"]) - abs(b["error_yd"])
            verdict = ("REGRESSION" if d > 0.02
                       else "improved" if d < -0.02 else "unchanged")
            print("  %-18s |err| %.2f -> %.2f  (%+.2f)  %s"
                  % (vid, abs(b["error_yd"]), abs(s["error_yd"]), d, verdict))
            if d > 0.02:
                rc = 1

    if a.set_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(
            {"recorded": _today(), "stride": a.stride, "tol": a.tol,
             "clips": summaries}, indent=1))
        print("\nbaseline written to %s" % base_path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
