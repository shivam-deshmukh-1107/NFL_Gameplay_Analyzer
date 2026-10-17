"""Apply the carry pass to a walk result: CARRIED frames, never certified.

    python scripts/carry.py 1
    python scripts/carry.py all --max-frames=30

Reads work/<NN>_<slug>/stage6/walk.json and writes stage7/carry.json. **It
never modifies the walk result.** The carry is a display-only tier: it adds
`carried_homography`, `carried_from`, `carried_frames`, `cc` and
`carry_check_yd` on frames the camera refused, and touches nothing a certified
frame owns.

WHY THIS CAN RECOVER WHAT THE WALK COULD NOT
--------------------------------------------
A refused frame freezes the reference, deliberately, because advancing it was
measured to produce false accepts. The cost is a cascade: on clip_01, by f142
the jump gate was comparing against f132's pose - ten frames of real camera
motion old - and every frame of f133-148 failed on `moves model` 3-8 yd.

The carry does not go through the solve. It aligns pixels directly to the
nearest certified keyframe, so it never sees the stale reference. That is the
whole point, and f133-148 is the test case.

READ THE STOP REASONS, NOT JUST THE COUNT. A carry that stops on
`paint_disagrees` found evidence against itself and quit, which is the system
working. One that stops on `max_frames` simply ran out of budget. Those are
different situations and the histogram separates them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import (carry as CARRY, config, gating, hudmask,   # noqa: E402
                      ingest, motion as MO, observe)


class Row:
    """Minimal stand-in for a walk FrameState, for `carry.carry_runs`."""

    def __init__(self, d):
        self.frame = d["frame"]
        self.tier = d["tier"]
        self.homography = d.get("homography")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"])
    ap.add_argument("--max-frames", type=int, default=CARRY.CARRY_MAX_FRAMES)
    a = ap.parse_args()

    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])

    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        wp = spec.workdir / "stage6" / "walk.json"
        if not wp.exists():
            print("%-18s no stage6/walk.json; run scripts/track.py first"
                  % spec.vid)
            continue
        walk = json.loads(wp.read_text())
        payload, pic = ingest.load(spec)
        turf, gate = gating.load(spec)
        hud = hudmask.load(spec)
        W, Hgt = pic.w, pic.h

        rows = [Row(d) for d in walk["frames"]]
        obs = observe.load(spec, 1) or []
        lines_by_frame = {o.frame: o.lines for o in obs}
        state_by_frame = {d["frame"]: d for d in walk["frames"]}

        runs = CARRY.carry_runs(rows)
        need = set()
        for i, j, before, after in runs:
            for k in range(i, j):
                need.add(rows[k].frame)
            for b in (before, after):
                if b is not None:
                    need.add(rows[b].frame)
        if not need:
            print("%-18s nothing to carry (no uncertified frames)" % spec.vid)
            continue

        frames = {}
        with ingest.FrameSource(spec, pic, stride=1, seek=False) as src:
            for i, fr in src:
                if i in need:
                    frames[i] = fr.copy()

        out, stops = [], {}
        for i, j, before, after in runs:
            for key_idx, direction in ((before, "fwd"), (after, "back")):
                if key_idx is None:
                    continue
                kf = rows[key_idx].frame
                H_key = np.asarray(rows[key_idx].homography,
                                   float).reshape(3, 3)
                if kf not in frames:
                    continue
                seq = (range(i, j) if direction == "fwd"
                       else range(j - 1, i - 1, -1))
                warp = np.eye(3)
                for k in seq:
                    f = rows[k].frame
                    dist = abs(f - kf)
                    row = CARRY.CarryRow(frame=f, carried_from=kf,
                                         carried_frames=dist)
                    if dist > a.max_frames:
                        row.stopped = "max_frames"
                        stops["max_frames"] = stops.get("max_frames", 0) + 1
                        out.append(row)
                        break
                    st = state_by_frame.get(f, {})
                    if f not in frames:
                        row.stopped = "frame_missing"
                        out.append(row)
                        break
                    M, cc = CARRY.align(frames[kf], frames[f], turf, hud,
                                        warp_init=warp)
                    if M is None or cc is None:
                        row.stopped = "align_failed"
                        stops["align_failed"] = stops.get("align_failed", 0) + 1
                        out.append(row)
                        break
                    row.cc = round(cc, 4)
                    if cc < CARRY.CARRY_MIN_CC:
                        row.stopped = "low_cc"
                        stops["low_cc"] = stops.get("low_cc", 0) + 1
                        out.append(row)
                        break
                    warp = M
                    Hc = M @ H_key
                    chk = CARRY.paint_disagreement_yd(
                        Hc, lines_by_frame.get(f, []), W, Hgt)
                    row.carry_check_yd = (None if chk is None
                                          else round(chk, 3))
                    if chk is not None and chk > CARRY.CARRY_LINE_MAX_YD:
                        row.stopped = "paint_disagrees"
                        stops["paint_disagrees"] = stops.get(
                            "paint_disagrees", 0) + 1
                        out.append(row)
                        break
                    row.homography = [float(x) for x in Hc.ravel()]
                    out.append(row)

        # Each frame keeps whichever keyframe is nearer, as Yash's does.
        best = {}
        for r in out:
            if not r.ok:
                best.setdefault(r.frame, r)
                continue
            cur = best.get(r.frame)
            if cur is None or not cur.ok or r.carried_frames < cur.carried_frames:
                best[r.frame] = r
        carried = [r for r in best.values() if r.ok]

        n_unc = sum(1 for r in rows if r.tier != "CERTIFIED")
        chk = [r.carry_check_yd for r in carried if r.carry_check_yd is not None]
        ccs = [r.cc for r in carried if r.cc is not None]
        print("%-18s %d uncertified -> %d CARRIED (%.0f%%), %d run(s)"
              % (spec.vid, n_unc, len(carried),
                 100.0 * len(carried) / max(n_unc, 1), len(runs)))
        if chk:
            print("        paint check: med %.3f  p90 %.3f  max %.3f yd "
                  "(stop rule %.1f)"
                  % (np.median(chk), np.percentile(chk, 90), max(chk),
                     CARRY.CARRY_LINE_MAX_YD))
        if ccs:
            print("        alignment cc: med %.3f  min %.3f (stop rule %.2f)"
                  % (np.median(ccs), min(ccs), CARRY.CARRY_MIN_CC))
        if stops:
            print("        stopped: %s"
                  % ", ".join("%s x%d" % (k, v)
                              for k, v in sorted(stops.items(),
                                                 key=lambda t: -t[1])))

        d = spec.workdir / "stage7"
        d.mkdir(parents=True, exist_ok=True)
        (d / "carry.json").write_text(json.dumps({
            "clip": spec.vid,
            "max_frames": a.max_frames,
            "n_uncertified": n_unc,
            "n_carried": len(carried),
            "stops": stops,
            "frames": [vars(r) for r in sorted(best.values(),
                                               key=lambda r: r.frame)],
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
