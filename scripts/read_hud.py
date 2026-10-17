"""Decode the HUD alphabet from the footage, then read down, distance and
absolute field position.

    python scripts/read_hud.py all --stride=5
    python scripts/read_hud.py 2 4

Glyphs are pooled across every video given, because digit coverage is the
limiting factor: geometry can only teach digits that actually occur in the
distance field, and Blitz needs 30 yards for a first down, so "AND 4" through
"AND 9" are rare. Shivam pooled three long captures (18-40 min) to reach 10/10
digits over 1,814 readings.

NOTHING HERE IS HAND-LABELLED. The alphabet comes from text structure plus the
geometric distance between the two drawn lines.

=============================================================================
WHAT THIS CAN AND CANNOT BE USED FOR
=============================================================================
**It cannot grade the geometry of a capture it was taught on.** The digit
alphabet is voted using that capture's own measured distances, so a check of
"measured gap vs printed distance" would agree by construction. See hudtext.py,
"THE CIRCULARITY". The absolute-scale referee is `scripts/valid_scale.py`,
which uses a hand-entered distance instead.

What it IS for: absolute field position ("ON THE 23"), which is the reference
the lattice structurally lacks - the lattice measures distance along the field
beautifully and its origin is arbitrary. That is the input to yard identity
(MERGE_PLAN.md Phase 4).

Note "ON THE N" counts from the NEARER goal line, so it is symmetric about
midfield: it gives the distance to a goal line, not which half the ball is in.
Resolving that needs a direction cue - the chevrons, the midfield logo or the
end zone.

Writes work/hud_book.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflfield import config, hudtext, scalecheck      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["all"])
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()

    keys = a.videos if a.videos != ["all"] else None
    specs = (config.VIDEOS if keys is None
             else [config.get_video(k) for k in keys])

    all_glyphs, all_lines, geo = [], [], {}
    for spec in specs:
        if not spec.path.exists():
            print("%-18s MISSING" % spec.vid)
            continue
        n_l0, n_g0 = len(all_lines), len(geo)
        res, glyph_lines, g = scalecheck.measure(spec, stride=a.stride,
                                                collect_glyphs=True)
        for key, gl in glyph_lines:
            ids = []
            for gly in gl:
                all_glyphs.append(gly)
                ids.append(len(all_glyphs) - 1)
            all_lines.append((key, ids))
        geo.update(g)
        print("%-18s +%4d text line(s), +%3d frame(s) with a measured distance"
              % (spec.vid, len(all_lines) - n_l0, len(geo) - n_g0))

    print("\npooled: glyphs=%d  text_lines=%d  distance_frames=%d"
          % (len(all_glyphs), len(all_lines), len(geo)))
    if len(all_glyphs) < 32:
        print("not enough glyphs to cluster an alphabet.")
        print("EXPECTED on short clips: the 'ON THE' banner appears between")
        print("plays, and these five sources are single plays. Pool the long")
        print("captures (Yash's clips 05-07, Shivam's videos 5-7) to get it.")
        return 0

    labels, K = hudtext.cluster_glyphs(all_glyphs)
    seqs = [(key, [int(labels[k]) for k in ids]) for key, ids in all_lines]
    book = hudtext.build_book(seqs, geo, labels)
    print("glyph clusters: %d" % K)
    print("AND block:      %s" % (book.and_block,))
    print("ON THE prefix:  %s" % (book.onthe_prefix,))
    print("digits labelled: %d/10 -> %s"
          % (len(book.digit),
             dict(sorted((int(k), v) for k, v in book.digit.items()))))

    dist, d_unread = hudtext.read_distances(seqs, book)
    pos, p_unread = hudtext.read_positions(seqs, book)
    print("\ndistance readings ('N' after AND): %d  (unreadable %d)"
          % (len(dist), d_unread))
    if dist:
        c = collections.Counter(v for _, v in dist)
        print("  distribution: %s" % dict(sorted(c.items())))
    print("position readings ('ON THE N'):    %d  (unreadable %d)"
          % (len(pos), p_unread))
    if pos:
        vals = np.array([v for _, v in pos])
        c = collections.Counter(vals.tolist())
        print("  distribution: %s" % dict(sorted(c.items())))
        print("  min %d  max %d" % (vals.min(), vals.max()))
        # The field-length inference, stated as the inference it is.
        print("  IF the hard ceiling is midfield, half-field = %d yd"
              " => total length %d yd" % (vals.max(), 2 * vals.max()))
        print("  (Shivam's 1,814 readings across three captures ceiling at"
              " exactly 50, which is the evidence for 100 yd. A ceiling from"
              " this few readings is not evidence of anything.)")

    out = config.WORK / "hud_book.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "videos": [s.vid for s in specs if s.path.exists()],
        "stride": a.stride,
        "n_glyphs": len(all_glyphs),
        "n_text_lines": len(all_lines),
        "n_distance_frames": len(geo),
        "n_clusters": int(K),
        "and_block": list(book.and_block),
        "onthe_prefix": list(book.onthe_prefix),
        "digit": {str(k): v for k, v in book.digit.items()},
        "taught_by_geometry": book.taught_by_geometry,
        "distance_readings": [[list(k), v] for k, v in dist],
        "position_readings": [[list(k), v] for k, v in pos],
    }, indent=1))
    print("\nwritten to %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
