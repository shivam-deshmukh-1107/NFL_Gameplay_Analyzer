"""Stages 0-2 for one or more videos: ingest, colour + gating, HUD mask.

    python scripts/prepare.py 1 2 3 4 5        # by processing order
    python scripts/prepare.py clip4_swing      # or by slug
    python scripts/prepare.py all
    python scripts/prepare.py all --stage0     # re-run ingest only

Each stage persists to work/<NN>_<slug>/stageN/ so it can be re-run on its own
without redoing the rest. Nothing here opens a video directly: every frame
comes through `ingest.FrameSource`, which is the only thing that knows about
pillarbox bars.

The gating stride adapts to length. The hand-cut clips are a few seconds long
and need dense sampling to give the mixture model anything to work with, while
a 20-40 minute capture only needs ~2 samples/second to segment shots - and at
that stride the frame source switches to seeking, which avoids decoding tens of
thousands of frames we would throw away.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, fieldcolor, gating, hudmask, ingest   # noqa: E402


def stage0(spec) -> dict:
    p = ingest.run(spec)
    pic, m, d = p["picture"], p["meta"], p["derived"]
    print("  [0] picture   %4dx%-4d at (%d,%d)  %-11s aspect %.4f  anisotropy %.4f"
          % (pic["w"], pic["h"], pic["x"], pic["y"],
             "CROPPED" if pic["cropped"] else "full frame",
             pic["aspect"], pic["anisotropy"]))
    print("      derived   px_scale %.4f  focal (%.1f, %.1f) px  %.3f fps  %s frames"
          % (d["px_scale"], d["focal_x_px"], d["focal_y_px"],
             m["fps"], m["nb_frames"]))
    if pic["refused_crop"]:
        print("      crop refused: %s" % pic["refused_crop"])
    return p


def stage1(spec, pic, duration: float) -> dict:
    stride = max(1, round((duration and 30.0) / 2)) if duration > 60 else 3
    min_run = 1.0 if duration > 60 else 0.5
    g = gating.run(spec, pic, stride=stride, min_run_seconds=min_run)
    print("  [1] turf      Lab(a,b) %s   mask threshold %.3f   stride %d"
          % (g["turf_model_mean_ab"], g["turf_threshold"], g["stride"]))
    print("      gating    gameplay %.1f%% in %d segment(s), %d cut(s), %d sampled"
          % (100 * g["gameplay_fraction"], g["n_gameplay_segments"],
             g["n_cuts"], g["n_sampled"]))
    return g


def stage2(spec, pic, turf, spans) -> dict:
    h = hudmask.run(spec, pic, turf, segments=spans)
    print("  [2] HUD       %.2f%% of the picture   std_max %.1f   %d sample(s)"
          % (100 * h["hud_fraction"], h["std_max"], h["n_samples"]))
    return h


def main(argv: list[str]) -> int:
    only = {a for a in argv if a.startswith("--")}
    keys = [a for a in argv if not a.startswith("--")] or ["all"]
    want = (lambda s: not only or ("--" + s) in only)

    specs = (config.VIDEOS if keys == ["all"]
             else [config.get_video(k) for k in keys])
    rc = 0
    for spec in specs:
        if not spec.path.exists():
            print("MISSING %-18s %s" % (spec.vid, spec.path.name))
            rc = 2
            continue
        t0 = time.time()
        print("%-18s build %s" % (spec.vid, spec.build or "(not established)"))
        try:
            if want("stage0"):
                stage0(spec)
            payload, pic = ingest.load(spec)
            duration = payload["meta"]["duration"] or 0.0
            if want("stage1"):
                stage1(spec, pic, duration)
            turf, gate = gating.load(spec)
            if want("stage2"):
                stage2(spec, pic, turf, gating.gameplay_spans(gate))
        except Exception as exc:                       # noqa: BLE001
            print("  FAILED    %s: %s" % (type(exc).__name__, exc))
            rc = 1
            continue
        print("  done in %.1fs" % (time.time() - t0))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
