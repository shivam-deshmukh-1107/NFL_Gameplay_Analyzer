"""Why does a frame fail the residual gate: our detector, or the frame?

    python scripts/exp_residual.py 3
    python scripts/exp_residual.py 3 --frames=42,44,115,166

For each frame the walk refused on residual, compare OUR line set against the
independently written stripe detector, and try fitting a camera to each.

THREE OUTCOMES, WRITTEN DOWN BEFORE RUNNING (docs/LESSONS):

  * the independent detector finds a clean 5-yard family where ours does not
    -> our detector is contaminating the pencil; the fix is in detection
  * neither finds a clean family
    -> the frames are genuinely unsolvable, and the parent project's higher
       certified count on this clip deserves the scrutiny rather than ours
  * ours is clean and the fit still fails
    -> the warm start is in a bad basin and the cold multistart is not
       rescuing it

The point of enumerating them first is that each implies a different fix, and
after the fact any single result can be told as a story about the others.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import (camera as CAM, config, gating, hudmask,     # noqa: E402
                      indep_paint, ingest, ladder as LAD, observe)


def ladder_of(ys):
    """(steps, rms, clean) for a sorted set of line positions."""
    if len(ys) < 4:
        return None, None, False
    lad = LAD.assign_ladder(np.sort(np.asarray(ys, float)))
    if lad is None:
        return None, None, False
    return list(lad[0]), float(lad[2]), bool(max(lad[0]) == 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--frames", default=None, help="comma list; default = the "
                                                   "frames refused on residual")
    a = ap.parse_args()
    spec = config.get_video(a.video)
    payload, pic = ingest.load(spec)
    turf, gate = gating.load(spec)
    hud = hudmask.load(spec)
    W, Hgt = pic.w, pic.h
    ay = CAM.aspect_ay(W, Hgt)

    wp = spec.workdir / "stage6" / "walk.json"
    if not wp.exists():
        wp = spec.workdir / "stage5" / "walk.json"
    walk = json.loads(wp.read_text())
    if a.frames:
        want = [int(x) for x in a.frames.split(",")]
    else:
        want = [r["frame"] for r in walk["frames"]
                if (r.get("reason") or "").startswith("residual")]
    if not want:
        print("no frames refused on residual in %s" % wp)
        return 0
    byf = {r["frame"]: r for r in walk["frames"]}
    obs = {o.frame: o for o in (observe.load(spec, 1) or [])}

    print("=== %s: %d frame(s) refused on residual" % (spec.vid, len(want)))
    print("%-6s | %-26s | %-26s | %s"
          % ("frame", "OURS (lines/rms/clean)", "INDEP (lines/rms/clean)",
             "fit to indep lines"))
    verdicts = {"detector": 0, "frame": 0, "basin": 0, "inconclusive": 0}
    rows = []
    with ingest.FrameSource(spec, pic, stride=1, seek=False) as src:
        for i, fr in src:
            if i not in want:
                continue
            ob = obs.get(i)
            ours = [y for y, _ in (ob.lines if ob else [])]
            o_steps, o_rms, o_clean = ladder_of(ours)

            st = indep_paint.stripe_lines(fr)
            ind = [y for y, _ in st]
            i_steps, i_rms, i_clean = ladder_of(ind)

            # Can a camera fit the INDEPENDENT line set on this frame?
            #
            # TWO CORRECTIONS to the first version of this experiment, both of
            # which hid the signal that was already in the table:
            #
            #  * It required `clean` (no multi-step rungs) before trying a fit.
            #    But the independent detector MISSES FAINT FAR LINES by design,
            #    so it legitimately returns 2-step gaps - that is a gap, not
            #    contamination, and the ladder handles multi-step rungs fine.
            #    Requiring `clean` meant most frames were never tested.
            #  * It fitted through `bootstrap_pose`, which refuses below
            #    ANCHOR_MIN_LINES = 6. The independent sets here have 3-5
            #    lines, so every one came back "fit failed" for a reason that
            #    had nothing to do with the frame.
            #
            # Now: fit with `fit_camera` directly at whatever DOF the line
            # count supports, exactly as `solve_frame` would.
            fit_txt, kind = "-", "inconclusive"
            gatepx = CAM.MAX_FIT_RESID_PX * pic.px_scale
            if i_steps is not None and len(st) >= CAM.MIN_SOLVE_LINES:
                lad = LAD.assign_ladder(np.sort(np.asarray(ind, float)))
                order = np.argsort(ind)
                lines_s = [(float(ind[k]), float(st[k][1])) for k in order]
                yards = 5.0 * np.asarray(lad[1], float)
                free = CAM.FREE_ALL.copy()
                free[0] = False
                start = np.array([config.GAME_FOCAL_PX * pic.px_scale, 0.0,
                                  np.radians(-28.5), 0.0,
                                  float(np.mean(yards)) - 40.0, 16.0])
                H, pp, r = CAM.fit_camera(lines_s, yards, W, Hgt, start, free,
                                          multistart=True, ay=ay)
                if H is not None and r is not None:
                    med = float(np.median(r))
                    fit_txt = "%.2f px" % med
                    kind = "detector" if med <= gatepx else "frame"
                else:
                    fit_txt = "fit failed"
                    kind = "frame"
            else:
                fit_txt = "too few indep lines (%d)" % len(ind)
                kind = "frame" if (o_rms is None or o_rms > gatepx) else "basin"
            # The clearest single signal, independent of any fit: how much
            # better does the independent pencil fit the 1-D projective model?
            if o_rms and i_rms and o_rms > 3.0 * i_rms:
                kind = "detector"
            verdicts[kind] += 1

            print("%-6d | %2d / %-6s / %-5s      | %2d / %-6s / %-5s      | %s -> %s"
                  % (i, len(ours),
                     ("%.2f" % o_rms) if o_rms is not None else "-",
                     o_clean, len(ind),
                     ("%.2f" % i_rms) if i_rms is not None else "-",
                     i_clean, fit_txt, kind))
            rows.append({"frame": i, "ours_n": len(ours), "ours_rms": o_rms,
                         "ours_clean": o_clean, "indep_n": len(ind),
                         "indep_rms": i_rms, "indep_clean": i_clean,
                         "indep_fit": fit_txt, "verdict": kind})
            want.remove(i)
            if not want:
                break

    print()
    print("verdicts: %s" % verdicts)
    print("  detector     = the independent pencil fits far better than ours")
    print("                 (camera inside the gate, or ladder rms 3x tighter),")
    print("                 so OUR pencil was the problem")
    print("  frame        = neither set yields a clean fit; the frame is hard")
    print("  basin        = our set is clean but the fit failed anyway")
    d = spec.workdir / "stage7"
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp_residual.json").write_text(json.dumps(
        {"clip": spec.vid, "verdicts": verdicts, "frames": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
