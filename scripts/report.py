"""The headline numbers, merged across stages. Four tiers, never one.

    python scripts/report.py
    python scripts/report.py 1 2 4

Reads whatever each stage has left on disk - the walk (stage 6), the carry
(stage 7), the referees (stage 7) and the scale check - and prints one table
per clip. Nothing is computed here: **if a number is missing, the stage that
produces it has not been run**, and that is shown as a gap rather than filled
in with a default.

WHY THE TIERS ARE NEVER SUMMED
------------------------------
    CERTIFIED  camera solved from this frame's own paint, every gate passed
    CARRIED    ECC turf alignment from a certified keyframe, validated against
               the visible paint, bounded at 30 frames
    TRACKED    pose carried through measured motion; no camera claim at all
    REFUSED    the evidence is not there

Adding them reports the weakest as if it were the strongest. A frame moved from
REFUSED to TRACKED is a gain in honesty; a frame moved from TRACKED to
CERTIFIED is a gain only if an independent referee says so.

**NEVER QUOTE THESE FROM A DOCUMENT.** Run this. Both parent projects carried
figures from superseded runs more than once, and one of them holds three
different values for the same measurement in three files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config      # noqa: E402

#: Hand-entered HUD truth, as in scripts/valid_scale.py. The only hand-entered
#: numbers in the project.
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

#: Yash's committed certified counts on the four shared clip files, for a
#: like-for-like column. His clips 05-07 are not in this repository and
#: hail_mary is not in his.
YASH = {"clip1_kickoff": (168, 253), "clip2_congestion": (173, 174),
        "clip3_qb": (183, 186), "clip4_swing": (260, 260)}


def read(p):
    return json.loads(p.read_text()) if p.exists() else None


def main(argv) -> int:
    keys = argv or None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])

    print("%-18s %6s %8s %8s %8s %8s %9s %9s %8s"
          % ("clip", "frames", "CERT", "CARRIED", "TRACKED", "REFUSED",
             "covered", "scale err", "vs Yash"))
    for spec in specs:
        w = read(spec.workdir / "stage6" / "walk.json")
        if w is None:
            w = read(spec.workdir / "stage5" / "walk.json")
        if w is None:
            print("%-18s no walk result; run scripts/track.py" % spec.vid)
            continue
        c = read(spec.workdir / "stage7" / "carry.json")
        r = read(spec.workdir / "stage7" / "referee.json")

        n = max(w["n_frames"], 1)
        counts = w["counts"]
        cert = counts.get("CERTIFIED", 0)
        trk = counts.get("TRACKED", 0)
        ref = counts.get("REFUSED", 0)
        carried = (c or {}).get("n_carried")
        # A carried frame was uncertified in the walk, so it is reclassified
        # out of TRACKED/REFUSED rather than added on top. Which of the two it
        # came from is in the per-frame records; for the headline it is taken
        # off REFUSED first, since those are the frames the carry targets.
        if carried:
            take_ref = min(carried, ref)
            ref -= take_ref
            trk -= (carried - take_ref)
        covered = cert + (carried or 0) + max(trk, 0)

        s = w.get("scale") or {}
        truth = TRUTH.get(spec.vid)
        err = ("%+.2f" % (s["median_yd"] - truth)
               if (truth and s.get("median_yd")) else "-")
        y = YASH.get(spec.vid)
        ycol = ("%+d" % (cert - y[0])) if y else "-"

        print("%-18s %6d %8d %8s %8d %8d %8.1f%% %9s %8s"
              % (spec.vid, w["n_frames"], cert,
                 str(carried) if carried is not None else "-",
                 max(trk, 0), ref, 100.0 * covered / n, err, ycol))
        print("        certified %.1f%%%s | anchors %s"
              % (100.0 * cert / n,
                 ("; carry %s" % ("not run" if c is None else "run"))
                 if c is None else "",
                 ", ".join("f%d" % x["frame"] for x in w.get("anchors", []))
                 or "none"))
        if s.get("n"):
            print("        scale: n=%d median %.2f yd IQR %s%s"
                  % (s["n"], s["median_yd"],
                     ("%.2f" % s["iqr_yd"]) if s.get("iqr_yd") else "-",
                     ("  (HUD %.0f)" % truth) if truth else "  (no truth)"))
        if r:
            fl = r.get("flow") or []
            slips = [v for v in fl if v.get("verdict") == "slip"]
            pt = [x["err_yd"] for x in (r.get("paint") or [])
                  if x.get("err_yd") is not None]
            dn = [x["density"] for x in (r.get("paint") or [])
                  if x.get("density") is not None]
            sc = r.get("selfcheck") or {}
            print("        referee: %s | flow %d pair(s), %d slip | "
                  "paint %s yd med over %d frame(s)%s"
                  % ("validated" if sc.get("injected") == "slip"
                     else "NOT VALIDATED",
                     len(fl), len(slips),
                     ("%.3f" % np.median(pt)) if pt else "-", len(pt),
                     ("  density med %.2f" % np.median(dn)) if dn else ""))
            if not r.get("certified_only"):
                print("        NOTE: referee run WITHOUT --certified-only, so "
                      "the flow verdict partly grades motion against motion")
        else:
            print("        referee: not run")
    print("\nTiers are never summed into one number; see this script's "
          "docstring for why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
