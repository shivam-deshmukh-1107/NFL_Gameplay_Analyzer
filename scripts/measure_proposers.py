"""Measure how often the two identity proposers agree, per source.

    python scripts/measure_proposers.py all [--stride N]

Writes work/<NN>_<slug>/stage4/proposers.json and prints one table.

This answers MERGE_PLAN.md Phase 3 step 2, which says to MEASURE the
disagreement rate before deciding what disagreement means. Nothing here feeds
the pipeline; it is an instrument.

Read the columns as follows. `both` is the share of frames where each proposer
returned an answer at all, which is the only population where agreement is
defined. `agree` is the share of THOSE frames whose integer rung patterns are
identical. A low `both` with a high `agree` is a different situation from a
high `both` with a low `agree`, and collapsing them into one number would hide
which one we are in.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import (config, gating, hudmask, ingest,      # noqa: E402
                      markings, propose)


def run_one(spec, stride: int) -> dict:
    payload, pic = ingest.load(spec)
    turf, gate = gating.load(spec)
    hud = hudmask.load(spec)
    spans = gating.gameplay_spans(gate) or [(0, payload["meta"]["nb_frames"] or 0)]

    rows = []
    t0 = time.time()
    for lo, hi in spans:
        with ingest.FrameSource(spec, pic, stride=stride, start=lo, stop=hi) as src:
            for i, fr in src:
                fm = markings.analyse_frame(fr, turf, hud=hud, frame_index=i)
                p = propose.compare(fm.pencil, pic.w)
                r = p.as_row()
                r["frame"] = int(i)
                rows.append(r)
    secs = time.time() - t0

    n = len(rows)
    both = [r for r in rows if r["lattice_ok"] and r["ladder_ok"]]
    agree = [r for r in both if r["agree"]]
    only_lad = [r for r in rows if r["ladder_ok"] and not r["lattice_ok"]]
    only_lat = [r for r in rows if r["lattice_ok"] and not r["ladder_ok"]]
    rms = [r["ladder_rms"] for r in rows if r["ladder_rms"] is not None]
    sc = [r["lattice_score"] for r in rows if r["lattice_score"] is not None]
    dropped = [r["n_dropped"] for r in rows]

    out = {
        "clip": spec.vid,
        "stride": stride,
        "n_frames": n,
        "n_both": len(both),
        "n_agree": len(agree),
        "agree_rate": (len(agree) / len(both)) if both else None,
        "n_only_ladder": len(only_lad),
        "n_only_lattice": len(only_lat),
        "ladder_rms_median": float(np.median(rms)) if rms else None,
        "ladder_rms_p90": float(np.percentile(rms, 90)) if rms else None,
        "lattice_score_median": float(np.median(sc)) if sc else None,
        "impostors_dropped_total": int(np.sum(dropped)),
        "impostors_frames_touched": int(np.sum(np.asarray(dropped) > 0)),
        "seconds": round(secs, 1),
        "frames": rows,
    }
    d = spec.workdir / "stage4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "proposers.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str]) -> int:
    stride = 5
    for a in argv:
        if a.startswith("--stride"):
            stride = int(a.split("=")[1]) if "=" in a else stride
    keys = [a for a in argv if not a.startswith("--")] or ["all"]
    specs = (config.VIDEOS if keys == ["all"]
             else [config.get_video(k) for k in keys])

    print("%-18s %6s %6s %7s %8s %8s %9s %8s %7s"
          % ("clip", "frames", "both", "agree", "lad-only", "lat-only",
             "lad_rms", "lat_score", "drops"))
    results = []
    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        r = run_one(spec, stride)
        results.append(r)
        print("%-18s %6d %6d %6s%% %8d %8d %9s %8s %7d"
              % (r["clip"], r["n_frames"], r["n_both"],
                 ("%.1f" % (100 * r["agree_rate"])) if r["agree_rate"] is not None else "n/a",
                 r["n_only_ladder"], r["n_only_lattice"],
                 ("%.2f" % r["ladder_rms_median"]) if r["ladder_rms_median"] is not None else "n/a",
                 ("%.3f" % r["lattice_score_median"]) if r["lattice_score_median"] is not None else "n/a",
                 r["impostors_dropped_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
