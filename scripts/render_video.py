"""Render the annotated video: one file per clip, in out/.

    python scripts/render_video.py all
    python scripts/render_video.py 4 --half-step        # add the 2.5 yd control
    python scripts/render_video.py 1 --no-audio

Reads the walk (stage 6) and the carry (stage 7) and draws each frame in its
tier's colour - green CERTIFIED, blue CARRIED, amber TRACKED, nothing at all
for REFUSED, with the banner naming the gate that refused it.

ENCODE: raw BGR piped to ffmpeg, libx264 at CRF 18, original audio muxed back.
Not `cv2.VideoWriter`, whose MPEG-4 Part 2 default Shivam measured as far
larger and softer for this material. ffmpeg is used here because encoding is
what ffmpeg is for - it contributes nothing to detection or geometry.

**THE RENDER IS THE CHECK THAT CANNOT BE FAKED.** Both parent projects record
defects that passed every numeric test and were obvious on screen - a grid
drawn vertically down the picture reporting 0.03 deg of angle error, a pose
certified while the grid bunched into the banner. Watch the output:

  * do the green lines sit ON the paint, at every depth?
  * one grid line per painted line, or every other one?
  * does the geometry jump at a tier change?
  * where nothing is drawn, is the stated reason plausible?

Output is written at PICTURE resolution - the active content box, pillarbox
bars removed - because that is the frame every measurement was made in.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, ingest, observe, render      # noqa: E402


def load_rows(spec):
    """Per-frame state, with carry results merged in. Returns (rows, source)."""
    wp = spec.workdir / "stage6" / "walk.json"
    src = "stage6"
    if not wp.exists():
        wp = spec.workdir / "stage5" / "walk.json"
        src = "stage5"
    if not wp.exists():
        return None, None
    walk = json.loads(wp.read_text())
    rows = {r["frame"]: dict(r) for r in walk["frames"]}

    cp = spec.workdir / "stage7" / "carry.json"
    n_carried = 0
    if cp.exists():
        for c in json.loads(cp.read_text())["frames"]:
            f = c["frame"]
            if c.get("homography") and f in rows:
                # A carried frame was NOT certified, so promoting it to CARRIED
                # cannot overwrite a certified pose - assert that rather than
                # assume it.
                if rows[f]["tier"] == "CERTIFIED":
                    continue
                rows[f].update(tier="CARRIED",
                               homography=c["homography"],
                               carried_from=c.get("carried_from"),
                               carried_frames=c.get("carried_frames"),
                               carry_check_yd=c.get("carry_check_yd"))
                n_carried += 1
    return rows, (src, n_carried, walk.get("counts", {}))


def ffmpeg_proc(out_path: Path, w: int, h: int, fps: float,
                source: Path | None, audio: bool):
    """Open an ffmpeg process accepting raw bgr24 on stdin."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (w, h), "-r", "%.6f" % (fps or 30.0), "-i", "-"]
    if audio and source is not None:
        cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"])
    ap.add_argument("--half-step", action="store_true",
                    help="also draw the 2.5 yd control lines, which must fall "
                         "BETWEEN the painted lines; if they land ON paint the "
                         "grid is half-scaled, which no distance metric sees")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])
    config.OUT.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING source" % spec.vid)
            continue
        rows, meta = load_rows(spec)
        if rows is None:
            print("%-18s no walk result; run scripts/track.py first" % spec.vid)
            continue
        src_stage, n_carried, counts = meta
        payload, pic = ingest.load(spec)
        fps = payload["meta"]["fps"] or 30.0

        # Drawn-line segments come from the observation cache when present.
        obs = observe.load(spec, 1) or []
        drawn = {o.frame: o.drawn for o in obs}

        out = spec.video_out
        proc = ffmpeg_proc(out, pic.w, pic.h, fps, spec.path, not a.no_audio)
        tally = {}
        n = 0
        try:
            with ingest.FrameSource(spec, pic, stride=1, seek=False) as fsrc:
                for i, fr in fsrc:
                    row = rows.get(i) or {"frame": i, "tier": "REFUSED",
                                          "reason": "not analysed"}
                    img = fr.copy()
                    render.frame_overlay(img, row, spec.vid,
                                         drawn=drawn.get(i),
                                         half_step=a.half_step)
                    tally[row["tier"]] = tally.get(row["tier"], 0) + 1
                    proc.stdin.write(img.tobytes())
                    n += 1
        finally:
            if proc.stdin:
                proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            rc = proc.wait()
        if rc != 0:
            print("%-18s ffmpeg failed (rc %d): %s"
                  % (spec.vid, rc, err.strip()[:200]))
            continue
        size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
        print("%-18s %4d frames %dx%d -> %s (%.1f MB)"
              % (spec.vid, n, pic.w, pic.h, out.name, size_mb))
        print("        %s | carried merged in: %d"
              % ("  ".join("%s %d" % (k, v) for k, v in sorted(tally.items())),
                 n_carried))
    print("\nWatch them. The render is the check that cannot be faked - both "
          "parent projects\nrecord defects that passed every numeric test and "
          "were obvious on screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
