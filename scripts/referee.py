"""Run the independent referees over a walk result.

    python scripts/referee.py 1 --certified-only
    python scripts/referee.py 1 --range=125,160 --stage=stage5 --certified-only
    python scripts/referee.py all --stride=5 --certified-only

TWO INSTRUMENTS, NEITHER SHARING CODE WITH WHAT IT GRADES:

  flow      dense-flow yard IDENTITY: does a line's name stay with the pixels?
            The only thing that can see a whole-line slip, because a grid
            shifted by one painted line still lands on painted lines.
  paint     an independently written stripe detector: how far is the certified
            grid from paint it never saw, in yards, and is the grid the right
            DENSITY (one grid line per painted line)?

THE INSTRUMENT IS VALIDATED FIRST, EVERY RUN. Before grading anything, a known
5-yard slip is injected into a real certified pose and the flow referee must
flag it. If that self-check fails, the referee is broken and its verdicts on
real data are worthless - Yash lost a full round to a test that returned a
confident wrong answer because a quadratic form was being fed two vanishing
POINTS instead of two lines, and only a synthetic check caught it.

**PASS `--certified-only` WHEN JUDGING IDENTITY.** The flow referee is
independent for CERTIFIED poses, whose geometry came from paint. It is NOT
independent for TRACKED poses: those are derived from motion and the referee
grades against motion, so grading one partly measures the motion estimator
against itself.

Writes work/<NN>_<slug>/stage7/referee.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import (camera as CAM, config, flowcheck, gating,   # noqa: E402
                      hudmask, indep_paint, ingest)
from nflfield.config import YARDS_PER_CELL                        # noqa: E402


def load_walk(spec, stage=None):
    stages = [stage] if stage else ["stage6", "stage5"]
    for s in stages:
        p = spec.workdir / s / "walk.json"
        if p.exists():
            return json.loads(p.read_text()), s
    raise FileNotFoundError("no walk.json for %s in %s; run scripts/track.py"
                            % (spec.vid, stages))


#: Columns, as fractions of the picture width, at which the grid is compared
#: with the paint.
#:
#: **MEASURED AT THE CENTRE COLUMN ALONE, THIS METRIC IS ANGLE-BLIND**, which
#: is the exact blind spot Yash records for his own `paint_fit`. Reproduced
#: here unintentionally and caught by LOOKING at a rendered frame, not by any
#: number: on clip_02 f95 the centre column reads 0.7 px while the frame edge
#: reads 2.5 px median and 6.1 px p90, because the grid's median line angle is
#: -2.91 deg against the paint's -4.74 deg. A 1.83 deg disagreement is 17 px of
#: drift at the edge and nothing at all at the centre.
#:
#: Sampling three columns makes the error visible where it is largest. It does
#: NOT replace `camera.line_direction_errors`, which measures the same defect
#: directly in degrees - this is a referee, that is a gate.
PAINT_COLUMNS = (0.1, 0.5, 0.9)


def grid_vs_paint(H, frame, W, Hgt, columns=PAINT_COLUMNS):
    """(median yard error, density, n_stripes) against an independent detector.

    The error is the WORST of the sampled columns, not the centre: a grid that
    is right in the middle and wrong at the edges is wrong, and reporting the
    centre alone would hide exactly the failure this was changed to catch.

    `density` is grid lines per detected painted line over the span they share.
    A grid 2x too dense reads ~2.0 and one 2x too sparse ~0.5, and NEITHER
    shows up in a distance error - a mis-scaled grid still lands on paint. That
    is why both numbers are reported and why the error alone is not enough.
    """
    stripes = indep_paint.stripe_lines(frame)
    if len(stripes) < 2:
        return None, None, len(stripes)
    grid = CAM.grid_lines_image(H, W, Hgt)
    if len(grid) < 2:
        return None, None, len(stripes)
    gy = np.array([y for _, y, _ in grid], float)
    sy = np.array([y for y, _ in stripes], float)
    lo, hi = sy.min(), sy.max()
    gin = gy[(gy >= lo - 5) & (gy <= hi + 5)]
    if len(gin) < 2:
        return None, None, len(stripes)
    spacing = float(np.median(np.abs(np.diff(np.sort(gin)))))
    if spacing < 1e-6:
        return None, None, len(stripes)
    px_per_yd = spacing / YARDS_PER_CELL

    # Re-evaluate every grid line AND every stripe at each column, so an
    # angular disagreement shows up instead of cancelling at the centre.
    xc = W / 2.0
    g_sl = {y: sl for _, y, sl in grid}
    s_sl = dict(stripes)
    worst = 0.0
    for frac in columns:
        x = frac * W
        gx = np.array([y + g_sl.get(y, 0.0) * (x - xc) for y in gin])
        sx = np.array([y + s_sl.get(y, 0.0) * (x - xc) for y in sy])
        if not len(gx) or not len(sx):
            continue
        d = np.abs(sx[:, None] - gx[None, :]).min(axis=1)
        worst = max(worst, float(np.median(d)) / px_per_yd)
    return worst, float(len(gin) / len(sy)), len(stripes)


def run_one(spec, a) -> dict:
    walk, stage = load_walk(spec, a.stage)
    payload, pic = ingest.load(spec)
    turf, gate = gating.load(spec)
    hud = hudmask.load(spec)
    W, Hgt = pic.w, pic.h

    rng = None
    if a.range:
        lo, hi = a.range.split(",")
        rng = (int(lo), int(hi))

    rows = walk["frames"]
    if rng:
        rows = [r for r in rows if rng[0] <= r["frame"] <= rng[1]]
    posed, tier = {}, {}
    for r in rows:
        tier[r["frame"]] = r["tier"]
        posed[r["frame"]] = (np.asarray(r["homography"], float).reshape(3, 3)
                             if r.get("homography") else None)

    def eligible(f):
        if posed.get(f) is None:
            return False
        return (not a.certified_only) or tier.get(f) == "CERTIFIED"

    want = {f for f in posed if f in tier}
    print("=== %s  (walk from %s, %d frames in scope)"
          % (spec.vid, stage, len(rows)))

    # STREAMING: at most two frames are held at once. The first version of this
    # script read every in-scope frame into a dict, which is ~1 GB for a
    # 253-frame clip at 1356x1016x3 and simply will not run on the long
    # captures (65,608 frames). The referee never needs more than an adjacent
    # pair, so it should never hold more.
    verdicts, paint = [], []
    selfcheck = None
    # The previous ELIGIBLE frame, not merely the previous in-scope one. With
    # --certified-only an ineligible frame sits between two certified ones
    # often (f143 refused between f142 and f144 on clip_01), and tracking the
    # previous in-scope frame silently drops that pair - measured, 23 pairs
    # judged instead of 24. Still at most two frames held.
    prev_i, prev_fr = None, None
    n_cert_seen = 0

    with ingest.FrameSource(spec, pic, stride=1, seek=False) as src:
        for i, fr in src:
            if i not in want:
                continue
            cur = fr.copy()

            if (eligible(i) and prev_i is not None
                    and (i - prev_i) <= flowcheck.MAX_GAP):
                v = flowcheck.check_pair(prev_fr, cur, posed[prev_i], posed[i],
                                         turf, prev_i, i, hud=hud)
                v.note = "%s/%s" % (tier.get(prev_i), tier.get(i))
                verdicts.append(v)

                # Validate the instrument on the first clean certified pair.
                if (selfcheck is None and v.verdict == "consistent"
                        and tier.get(prev_i) == "CERTIFIED"
                        and tier.get(i) == "CERTIFIED"):
                    bad = flowcheck.check_pair(
                        prev_fr, cur, posed[prev_i],
                        flowcheck.inject_slip(posed[i], YARDS_PER_CELL),
                        turf, prev_i, i, hud=hud)
                    selfcheck = {"pair": [prev_i, i],
                                 "clean": v.verdict,
                                 "clean_err_px": v.err_solved,
                                 "injected": bad.verdict,
                                 "injected_err_px": bad.err_solved}

            if tier.get(i) == "CERTIFIED" and posed.get(i) is not None:
                if n_cert_seen % max(1, a.stride) == 0:
                    err, dens, n = grid_vs_paint(posed[i], cur, W, Hgt)
                    if err is not None:
                        paint.append((i, err, dens, n))
                n_cert_seen += 1

            if eligible(i):
                prev_i, prev_fr = i, cur

    if selfcheck is None:
        print("  SELF-CHECK: no clean certified adjacent pair found; the "
              "referee is NOT validated and the verdicts below are untrusted")
    else:
        ok = "PASS" if selfcheck["injected"] == "slip" else "FAIL"
        print("  SELF-CHECK %s: pair f%d/f%d - clean pose %.1f px, a known "
              "+5 yd slip %.1f px -> %s"
              % (ok, selfcheck["pair"][0], selfcheck["pair"][1],
                 selfcheck["clean_err_px"] or -1,
                 selfcheck["injected_err_px"] or -1, selfcheck["injected"]))
        if ok == "FAIL":
            print("  the referee cannot see an injected slip; NOT grading")
            return {"clip": spec.vid, "walk_stage": stage,
                    "selfcheck": selfcheck, "flow": [], "paint": []}

    slips = [v for v in verdicts if v.verdict == "slip"]
    noans = [v for v in verdicts if v.verdict == "no_answer"]
    errs = [v.err_solved for v in verdicts if v.err_solved is not None]
    print("  FLOW: %d pair(s) judged, %d consistent, %d SLIP, %d no answer"
          % (len(verdicts), len(verdicts) - len(slips) - len(noans),
             len(slips), len(noans)))
    if errs:
        print("        solved-pose flow error: med %.1f  p90 %.1f px"
              % (np.median(errs), np.percentile(errs, 90)))
    for v in slips[:14]:
        alt = v.err_minus if v.best_shift < 0 else v.err_plus
        print("        SLIP f%d->f%d (%s): solved %.1f px, %+.0f yd "
              "relabelling %.1f px"
              % (v.frame_a, v.frame_b, v.note, v.err_solved or -1,
                 v.best_shift, alt or -1))

    if paint:
        e = np.array([x[1] for x in paint])
        dn = np.array([x[2] for x in paint])
        print("  PAINT: %d certified frame(s) scored against an independent "
              "detector" % len(paint))
        print("        grid-to-stripe error: med %.3f  p90 %.3f yd"
              % (np.median(e), np.percentile(e, 90)))
        print("        density (grid per painted line): med %.2f  range "
              "%.2f-%.2f" % (np.median(dn), dn.min(), dn.max()))
    else:
        print("  PAINT: no frame scored (too few independent stripes)")

    return {"clip": spec.vid, "walk_stage": stage, "range": rng,
            "certified_only": bool(a.certified_only),
            "selfcheck": selfcheck,
            "flow": [v.as_row() for v in verdicts],
            "paint": [{"frame": f, "err_yd": e, "density": d, "n_stripes": n}
                      for f, e, d, n in paint]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"])
    ap.add_argument("--stride", type=int, default=3,
                    help="score every Nth certified frame for the paint check")
    ap.add_argument("--range", default=None, help="LO,HI frame range")
    ap.add_argument("--stage", default=None, choices=["stage5", "stage6"],
                    help="grade a specific walk result. Use stage5 to grade "
                         "poses a later stage stopped certifying - the only "
                         "way to ask whether those were false accepts")
    ap.add_argument("--certified-only", action="store_true",
                    help="judge only pairs where BOTH frames are CERTIFIED. "
                         "See the module docstring: the flow referee is not "
                         "independent for TRACKED poses")
    a = ap.parse_args()

    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])
    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        try:
            out = run_one(spec, a)
        except FileNotFoundError as exc:
            print("%-18s %s" % (spec.vid, exc))
            continue
        d = spec.workdir / "stage7"
        d.mkdir(parents=True, exist_ok=True)
        (d / "referee.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
