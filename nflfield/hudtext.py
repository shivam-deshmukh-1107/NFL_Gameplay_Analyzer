"""Read the HUD's down, distance and absolute field position.

PROVENANCE
----------
Ported from Shivam's `blitzfield/hudtext.py`, with every pixel-sized constant
scaled (they were measured at `SHIVAM_REF_W`; this project works at the native
picture width).

Yash's project independently identified this as the missing instrument - his
field-map notes call the play-select banner "a free identity referee" - and did
not build it. This is the single most complementary piece of the merge.

WHAT IT GIVES
-------------
Blitz prints the ball's absolute position ("ON THE 23") in a fixed bitmap font,
which is the absolute reference the lattice structurally lacks: the lattice
measures distance along the field beautifully and its origin is arbitrary per
frame.

NOTHING HERE IS HAND-LABELLED. Glyphs are clustered by appearance, and the TEXT
STRUCTURE names them:

  * every down-and-distance line shares a three-glyph block - that is `AND`;
  * the three glyphs before it are the ordinal, giving 1ST/2ND/3RD/4TH and
    hence the digits 1-4;
  * short lines share a five-glyph prefix - that is `ON THE`;
  * trailing glyphs are digits, and their VALUES come from geometry: where the
    scrimmage and first-down lines are both visible, the yard gap between them
    is measured directly, and the trailing glyphs must spell that number.

=============================================================================
THE CIRCULARITY, AND WHY THE SCALE REFEREE DOES NOT USE THIS
=============================================================================
Read the last bullet again. **The digit alphabet is taught BY the geometric
distance measurement.** `build_book` takes the measured LOS-to-first-down gap,
rounds it, and asserts that the trailing glyphs spell that integer.

So this reader CANNOT grade the geometry on the same capture. If the lattice
had a systematic scale error, a true 30 yd gap would measure (say) 33, the
glyphs for "3" and "0" would both be voted "3", and a later reading of "ON THE
20" would come back wrong - while a check of "measured gap vs the number the
HUD prints" would agree perfectly, because the book was built to make it agree.
That check would be vacuous.

Therefore:

  * `scalecheck.py` + `scripts/valid_scale.py` - the absolute-scale referee -
    uses a per-capture distance ENTERED BY HAND (three numbers, flagged as the
    only hand-entered values in the project). That is genuinely independent.
  * This module's job is ABSOLUTE FIELD POSITION for yard identity
    (MERGE_PLAN.md Phase 4), not scale validation.
  * Its own validation is that its reading agrees with the hand-entered truth,
    which tests the READER, not the geometry.

A fully automatic and non-circular scale check is possible and is Phase 4 work:
learn the digit book on one capture, then apply it to a DIFFERENT one, where
that capture's geometry had no part in building it. `normalise_glyph` already
scales every glyph into a fixed box, so cross-capture transfer should work; it
has not been tested.

DIGIT COVERAGE IS THE LIMITING FACTOR
-------------------------------------
Geometry can only teach digits that actually occur in the distance field, and
because Blitz needs 30 yards for a first down, "AND 4" through "AND 9" are
rare. Shivam pooled glyphs across three long captures (18-40 min each) to see
enough of them and reached 10/10 digits over 1,814 readings. This project's
footage is five short clips, so expect poor coverage until the long captures
are added.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from .config import det_scale

#: Box every glyph is normalised into before comparison. Not a measurement of
#: the footage - an arbitrary fixed target - so it does not scale.
GLYPH_W, GLYPH_H = 20, 28


# --------------------------------------------------------------------------- #
# glyph extraction
# --------------------------------------------------------------------------- #
def yellow_mask(frame: np.ndarray) -> np.ndarray:
    """The HUD banner's yellow text."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= 18) & (H <= 38) & (S > 110) & (V > 150)).astype(np.uint8)


def normalise_glyph(g: np.ndarray) -> np.ndarray:
    """Scale to a fixed box, centre by mass, and blur.

    Resizing the bounding box to a fixed size also removes the anisotropic
    stretch of a non-4:3 upload, so glyphs from every source are directly
    comparable - which is what makes cross-capture digit transfer plausible.
    The blur absorbs sub-pixel misalignment; without it the same character
    fragments into many clusters.
    """
    g = cv2.resize(g, (GLYPH_W, GLYPH_H),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    M = cv2.moments(g)
    if M["m00"] > 1e-6:
        T = np.float32([[1, 0, GLYPH_W / 2 - M["m10"] / M["m00"]],
                        [0, 1, GLYPH_H / 2 - M["m01"] / M["m00"]]])
        g = cv2.warpAffine(g, T, (GLYPH_W, GLYPH_H))
    return cv2.GaussianBlur(g, (5, 5), 1.2)


def text_lines(frame: np.ndarray, top: float = 0.62, min_glyphs: int = 4):
    """Rows of large yellow glyphs in the lower banner.

    THE HEIGHT FILTER MATTERS: play-select screens fill the same area with
    small yellow X markers, which outnumber the banner text several to one.

    Every size bound here was measured at `SHIVAM_REF_W` and is scaled to this
    picture. `top` is a fraction of frame height and does not scale.
    """
    h, w = frame.shape[:2]
    s = det_scale(w)
    min_px = 90 * s * s                      # an area
    h_lo, h_hi = 20 * s, 80 * s              # glyph height
    w_lo, w_hi = 5 * s, 70 * s               # glyph width

    m = yellow_mask(frame)
    m[:int(h * top)] = 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cand = [(st[j, 0], st[j, 1], st[j, 2], st[j, 3], j) for j in range(1, n)
            if st[j, 4] >= min_px and h_lo <= st[j, 3] <= h_hi
            and w_lo <= st[j, 2] <= w_hi
            and st[j, 2] / max(st[j, 3], 1) <= 2.2]
    if len(cand) < min_glyphs:
        return []
    cand.sort(key=lambda c: c[1] + c[3] / 2)
    lines, cur = [], [cand[0]]
    for c in cand[1:]:
        if abs((c[1] + c[3] / 2) - (cur[-1][1] + cur[-1][3] / 2)) <= 0.55 * cur[-1][3]:
            cur.append(c)
        else:
            lines.append(cur)
            cur = [c]
    lines.append(cur)

    out = []
    for ln in lines:
        if len(ln) < min_glyphs:
            continue
        hh = np.array([c[3] for c in ln], float)
        if hh.std() / max(hh.mean(), 1) > 0.30:    # not one text row
            continue
        ln.sort(key=lambda c: c[0])
        out.append([normalise_glyph(
            (lbl[y:y + h_, x:x + w_] == j).astype(np.uint8) * 255)
            for (x, y, w_, h_, j) in ln])
    return out


# --------------------------------------------------------------------------- #
# the glyph book
# --------------------------------------------------------------------------- #
@dataclass
class GlyphBook:
    labels: np.ndarray           # cluster id per glyph
    digit: dict                  # cluster id -> character
    and_block: tuple
    onthe_prefix: tuple
    n_clusters: int

    @property
    def taught_by_geometry(self) -> bool:
        """True whenever `digit` came from measured distances.

        Kept as a flag so a caller cannot forget: a book with this set must
        never be used to grade the geometry of the same capture. See the module
        docstring.
        """
        return True


def cluster_glyphs(glyphs, threshold: float = 0.25) -> tuple[np.ndarray, int]:
    """Cluster glyph bitmaps by appearance. No labels, no training data."""
    from sklearn.cluster import AgglomerativeClustering
    X = np.array([g.reshape(-1) for g in glyphs], np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                 linkage="average").fit(X).labels_
    return lab, int(lab.max()) + 1


def build_book(seqs, geo: dict, labels: np.ndarray,
               min_votes: int = 3, purity: float = 0.6) -> GlyphBook:
    """Decode the alphabet from text structure plus geometric distances.

    **This is where the circularity enters** - see the module docstring. `geo`
    is the measured LOS-to-first-down gap per frame, and it is the teacher for
    every digit.
    """
    mid = collections.Counter(tuple(s[3:6]) for _, s in seqs if len(s) >= 8)
    and_block = mid.most_common(1)[0][0] if mid else ()

    votes = collections.defaultdict(collections.Counter)

    # geometry: trailing glyphs of a down line spell the measured distance
    for fi, s in seqs:
        if len(s) < 8 or tuple(s[3:6]) != and_block or fi not in geo:
            continue
        val = geo[fi]
        if abs(val - round(val)) > 0.45 or not (1 <= round(val) <= 40):
            continue
        txt = str(int(round(val)))
        tail = s[6:6 + len(txt)]
        if len(tail) != len(txt):
            continue
        for cl, ch in zip(tail, txt):
            votes[cl][ch] += 1

    digit = {}
    for cl, cnt in votes.items():
        ch, n = cnt.most_common(1)[0]
        if n >= min_votes and n / sum(cnt.values()) >= purity:
            digit[cl] = ch

    pre = collections.Counter(tuple(s[:5]) for _, s in seqs if 6 <= len(s) <= 7)
    onthe = pre.most_common(1)[0][0] if pre else ()
    return GlyphBook(labels, digit, and_block, onthe, int(labels.max()) + 1)


def read_distances(seqs, book: GlyphBook):
    """Printed first-down distances, `(frame_key, yards)`, plus failures.

    The number after `AND` on a down-and-distance line. This is the quantity
    `scalecheck` compares against - but only ever with a book that was NOT
    taught on the capture being graded.
    """
    out, unread = [], 0
    for fi, s in seqs:
        if len(s) < 8 or tuple(s[3:6]) != book.and_block:
            continue
        tail = s[6:]
        if not tail or any(t not in book.digit for t in tail):
            unread += 1
            continue
        try:
            out.append((fi, int("".join(book.digit[t] for t in tail))))
        except ValueError:
            unread += 1
    return out, unread


def read_positions(seqs, book: GlyphBook):
    """Absolute field position readings, `(frame_key, yard)`, plus failures.

    "ON THE N". N counts from the NEARER goal line and so is symmetric about
    midfield - it gives the distance to a goal line, not which half of the
    field the ball is in. That ambiguity is what the chevrons, the midfield
    logo and the end zone resolve (MERGE_PLAN.md Phase 4).
    """
    out, unread = [], 0
    for fi, s in seqs:
        if not (6 <= len(s) <= 7) or tuple(s[:5]) != book.onthe_prefix:
            continue
        tail = s[5:]
        if any(t not in book.digit for t in tail):
            unread += 1
            continue
        try:
            out.append((fi, int("".join(book.digit[t] for t in tail))))
        except ValueError:
            unread += 1
    return out, unread
