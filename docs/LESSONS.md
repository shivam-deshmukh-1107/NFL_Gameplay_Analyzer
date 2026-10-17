# Lessons — measured in THIS project

Facts measured in the merged pipeline, and claims that turned out wrong here.
Both parent projects keep a document like this and both say the same thing about
it: every claim that got retracted was stated confidently first and checked
second. The inherited lessons live in the parents' own files
(`Yash_Field_Tracker/easely-done/docs/LESSONS.md`, 12 retracted claims;
`Field_Tracker/ENGINEERING_NOTES.md`, the failed-experiment log) and are not
repeated here — read both before touching the camera model, a refusal gate, or
anything about yard identity.

**Never quote a figure from this file as current.** Re-run the script named
beside it.

---

## Verified on arrival (2026-10-17)

The environment and the first four stages reproduce both parents' independent
measurements, which is the only free validation this merge has.

| what | merged pipeline | parent, independently |
|---|---|---|
| picture frame, clips 1–4 | 1356×1016, full frame, `px_scale` 1.0000 | Yash's camera constants were measured at exactly this width |
| picture frame, Hail Mary | 1444×1080 at x=238 | Yash's `content_box` measured a comparable pillarboxed capture at x[239..1681] = 1443×1080 |
| focal, Hail Mary | 1609.1 px | Yash computed 1608 px for a 1443-wide picture |
| frame counts | 253 / 174 / 186 / 260 / 221 | identical in both parents |
| turf chromaticity, clips | Lab (a,b) ≈ (110, 126) | Shivam documents (110, 126) |
| turf chromaticity, Hail Mary | Lab (a,b) = (119.37, 132.85) | Shivam documents (119.3, 132.9) |
| long lines/frame, clips 1–4 | 11 / 10 / 9 / 15 (median) | Shivam documents 10 / 12 / 9 / 14 |

Agreement on the pillarbox crop and the turf colour is worth more than it looks:
they were measured by different code, in a different language of thresholds, on
a file neither project shares with the other.

## The two proposers agree (`scripts/measure_proposers.py`, stride 5)

The merge's one genuinely new component: Shivam's comb-score lattice and Yash's
beam-search ladder, run on the same line set, compared on the integer rung
pattern.

| clip | frames | both answered | **agree** | ladder rms (med) | lattice score (med) |
|---|---|---|---|---|---|
| clip1_kickoff | 51 | 36 | **88.9%** | 1.18 | 0.966 |
| clip2_congestion | 35 | 35 | **100.0%** | 1.06 | 0.972 |
| clip3_qb | 38 | 34 | **88.2%** | 1.32 | 0.967 |
| clip4_swing | 52 | 52 | **98.1%** | 1.12 | 0.967 |
| hail_mary_left | 45 | 45 | **33.3%** | **8.05** | **0.621** |

Two things to read off this and nothing more:

* On clips 1–4 two independently written estimators of line identity agree on
  88–100% of frames where both answer. That is evidence the line sets are
  genuinely one 5-yard family on those clips.
* On Hail Mary they agree on a third, and **both independently say why**: the
  ladder residual is 7× the clips' and the lattice score is 0.35 below theirs.
  The cross-check is flagging a source we already suspected from detection
  counts, not disagreeing arbitrarily.

**What agreement does NOT prove.** Both proposers are blind to the same
scene-scale degeneracy — a ladder and the same ladder doubled have identically
zero projective residual, and a comb score at pitch α also scores well at α/2.
Agreement here says nothing about it. `MAX_JUMP_YD` in the camera gates is the
only thing that catches it.

## Three bugs caught by measurement, all silent

Recorded because each produced plausible-looking output and none raised an
error. All three are the same species: **a number carried from one project into
the other without its units.**

1. **`LONG_CUT` reference width.** Shivam measured the full-line/tick cut at
   110 px on his 960-wide canonical buffer; carried over as a flat 110 at the
   1356-wide picture frame it admits short segments as full yard lines. Measured
   on clip_04 f0: 20 long lines against 15 at the correct 155.4 px. Per-frame
   medians on clips 1–4 read 12/14/9/16 against Shivam's 10/12/9/14.

2. **Every other detection constant, same cause.** `merge_collinear`'s 80 px gap
   tolerance, `classify_flanks`'s ±5 px probe, `detect_segments`'s 18 px minimum
   and all the morphology kernels were measured at 960 and used unscaled. Too
   small a gap tolerance leaves fragments of one player-broken line unmerged.
   Worst on the dark-graded Hail Mary capture (turf L of 19 against the clips'
   100, so paint has low contrast and fragments more): the long-line count read
   **36 per frame against Shivam's documented 10** — the extra 26 were pieces of
   lines already counted, each individually long enough to pass the length cut,
   all heading for the ladder. Fixed by `config.det_scale()`; there are now two
   named reference widths and no bare numbers.

3. **The proposer comparison was reversed, not the proposers.** `LatticeFit.steps`
   written as `diff(sort(index))` sorts the index and destroys its
   correspondence with position order; on a descending index (negative `alpha`)
   it returns the rung pattern backwards. Agreement read 47–77% on clips 1–4,
   and **every clip_02 disagreement was an exact reversal** of the other
   proposer's answer. Fixed to `abs(diff(index))`, agreement went to 88–100%.

   This one is the lesson, not the bug: a 47% agreement rate between two
   independent estimators is a publishable-sounding finding, and it was an
   artefact of the instrument. **Assume the metric has the blind spot before
   concluding anything about what it measures.**

## The absolute-scale referee, pre-tracker (`scripts/valid_scale.py`, stride 4)

The first measurement in this project against ground truth the geometry never
sees. Recorded as the **pre-tracker baseline** in `work/scale_baseline.json`.

| clip | frames | measured | median | IQR | HUD | error | p90 \|err\| |
|---|---|---|---|---|---|---|---|
| clip2_congestion | 44 | 9 | 30.15 | 0.13 | 30 | **+0.15** | 0.35 |
| clip4_swing | 65 | 19 | 29.37 | 0.36 | 29 | **+0.37** | 1.01 |
| hail_mary_left | 56 | 19 | 30.32 | 0.92 | 30 | +0.32 | **48.61** |

Against Shivam's published figures for the same two clips (+0.17/+0.24 with a
tracker and the trained net), clip2 is better and clip4 is worse. Worst error
0.37 yd is **above the 0.25 yd bar.**

Three things to read, and one not to.

* **clip2 is better than the parent and much tighter** (IQR 0.13 against his
  0.93). Measured through a cold per-frame lattice with no tracker and no net.
* **hail_mary's median is luck, not evidence.** +0.32 looks passable; the IQR
  is 0.92 and the worst 10% of frames are off by 48.61 yd. The scale is not
  being recovered consistently there at all, which agrees with everything else
  this capture has shown (ladder rms 8.05, lattice score 0.62, 605 raw
  segments). **Read the IQR, never the median alone** - this row is why.
* **The yield is low and the reason is the instrument, not the geometry.**
  Only 9 of 44 and 19 of 65 frames produce a number, because both drawn lines
  must be fitted and usually only one is. Breakdown on clip_02's 35 missing
  frames: 21 found `cyan` + `orange_yellow`, 7 found `blue` + `cyan`, 6 found
  three families but not the pair.
* **Do not read "ABOVE BAR" as a regression.** There is nothing to regress
  from yet; this is the first measurement. The script now separates the two
  questions, because calling a first run a regression is meaningless.

### A falsifiable prediction, written down before the tracker exists

**clip4's +0.37 is predicted to be cold-fit horizon jitter, not a scale error,
and the tracker is predicted to remove most of it.**

The reasoning, and the measurement behind it. The drawn-line gap is computed
from the lattice `(v, alpha, beta)` - and those are exactly the quantities that
hop. Measured here on clip_04, consecutive sampled frames: horizon `v` moves a
median **33 px** and a maximum **122.6 px**, pitch `alpha` by a median 10% and
a maximum 34%. Shivam measured the same phenomenon in his cold fit (up to
111 px of `v`, up to 58% of pitch) and noted the lattice those numbers describe
did not move at all - the objective is nearly flat along that ridge.

Supporting evidence already visible: clip_04's 19 gap measurements are
**bimodal** - eight cluster at 28.92-29.09 and seven at 29.34-29.50, plus
30.00, 30.09 and one 37.11 - which is what hopping between two ridge positions
looks like, and is not what a systematic scale error looks like.

His fix was `fit_labelled` plus a horizon prior toward the motion prediction,
which cut horizon jitter 3.2x and pitch jitter 4x. That requires labels, which
requires the tracker (stage 6).

**So: if the tracker lands and clip4's error does not fall materially, this
explanation is wrong and the scale needs looking at directly.** The baseline is
recorded so that is checkable rather than arguable.

## The HUD reader works, and cannot do its job on this footage yet

`scripts/read_hud.py` pooled over all five sources at stride 6:

| | |
|---|---|
| glyphs pooled | 179 |
| text lines found | 23 |
| frames with a measured distance | 30 |
| glyph clusters | 20 |
| `AND` block found from structure | yes, (6, 9, 0) |
| `ON THE` prefix found from structure | yes, (3, 8, 2, 13, 12) |
| **digits labelled** | **2 of 10** ('2', '9') |
| distance readings | 6, all "29" |
| **position readings ("ON THE N")** | **0** (5 unreadable) |

The mechanism is verified end to end - text structure finds both blocks
unaided - but the alphabet cannot be learned from this footage. Expected, for a
stated reason: geometry can only teach digits that occur in the distance field,
and these five sources are single plays that never show the between-plays
banner. Shivam pooled three captures of 18-40 minutes to reach 10/10 digits
over 1,814 readings.

**This is a concrete dependency, not a defect: absolute field position needs
the long captures** (Yash's clips 05-07, Shivam's videos 5-7).

### The circularity, which is the real finding here

The digit alphabet is voted using the measured LOS-to-first-down distance, so
**the HUD reader cannot grade the geometry of a capture it was taught on.**
Visible in the numbers above: it learned '2' and '9' from a geometry that
measured 29.37, then read back "29". A check of "measured gap vs printed
distance" built on that book would agree by construction and prove nothing.

Hence the division now in the code: `valid_scale.py` grades the geometry using
three hand-entered distances (the only hand-entered numbers in the project),
and `read_hud.py` produces absolute position for identity and never grades
anything. A non-circular automatic check is possible - teach the book on one
capture, apply it to another - and is Phase 4 work; `normalise_glyph` already
puts every glyph in a fixed box, so it should transfer, and that is untested.

## Stage 5: the camera, and the fourth silent bug

### "Locked" silently meant "chosen arbitrarily by multistart"

The worst bug in this project so far, and the one most worth understanding,
because **no position-based metric could have caught it.**

`bootstrap_pose` locks the focal to the game constant and asks `fit_camera` for
a multistart. The multistart grid assigned each candidate its own `f0` from
(700, 1100, 1600, 2400) — and because the parameter was *not free*, each
candidate simply kept the value it was handed. Every candidate then fitted the
data equally well, for a reason this project had already proved in a test:
**the focal is not observable from yard lines.** So the winner was arbitrary,
and it was the first grid entry.

Result: every certified frame of clip_01 and clip_02 came back with

| | measured here (bug) | Yash measured |
|---|---|---|
| focal | **700.0** on all 354 frames | 1511 |
| elevation | **−49.4°** | −28.4 / −28.5 / −28.5 / −28.5° (four clips) |
| height | 13.0 / 18.9 yd | 16–19 yd |

Yash's own focal table gives elevation **−49.1° at focal 700**, so the fault
was legible in a single number — if you knew to look at elevation, which is
the point.

**What makes this the instructive case: every position metric said fine.** The
median residual was 0.69 px. The jump gate read 0.29 yd p90. `line_dir` read
0.368° median. The absolute-scale referee passed at +0.15 yd. The yard lines
landed on the paint. All of that is *correct and expected*, because the
focal/scale degeneracy preserves the projected line positions exactly — it
moves only height, elevation and anything cross-field. The downfield axis was
genuinely unaffected; everything else was silently wrong.

The fix is a general invariant, not a special case for the focal: **a parameter
that is not free must hold the same value in every start.** Guarded by
`test_locked_focal_stays_locked_through_multistart`, which checks the focal
*and* asserts the recovered elevation is inside the range this footage has ever
shown — it now returns 28.5°, independently reproducing Yash's four-clip
measurement.

### Synthetic tests reproduce both documented degeneracies

`tests/test_camera.py`, 11 tests, no footage required. Two matter most:

| test | result | Yash's measurement |
|---|---|---|
| scale degeneracy is exact | max \|dy\| = **1.71e-13 px** over k = 0.5/2/4 | ~3e-12 px |
| focal not constrained by yard lines | residual spread **8.53e-14 px** over a 6× focal range, while height moves 7.2 → 18.8 yd | residual bit-identical, height 18.7 → 7.0 yd |

The second is a tripwire: **if it ever fails, the model has gained a constraint
and `config.GAME_FOCAL_PX` must be re-derived, not trusted.**

The jump gate catches the k=2 degenerate twin at 17.0 yd against its 3.0 yd
threshold — which is the whole reason that guard is measured in yards and not
in pixels, since the twin's yard lines are pixel-identical.

### A test that asserted the wrong mechanism

Recorded because the distinction is real. The first version of the banner-edge
test gave the solve **eleven** lines, one of them a flat banner edge, and
demanded a refusal. The solve certified — correctly. With eleven lines the
impostor sits 28.5 px from its model line, `OUTLIER_PX` drops it, and ten clean
lines remain; refusing there would throw away a good frame for one bad segment,
which is precisely what outlier rejection exists to prevent.

What actually saved Yash's clip_01 f206 was the **line count**: drop one of four
and three remain, which `MIN_KEPT_AFTER_REJECTION` refuses because a
5-parameter fit to 3 lines is near-exact by construction. The test now checks
both halves — many lines → excluded and certified; four lines → refused — and
reproduces f206's exact refusal message.

### Stage 5 coverage, and what the refusals were

Forward-only walk, stride 1, after the focal fix. Yash's committed counts are
on the same four clip files, so this column is like for like.

| clip | ours (certified) | Yash | difference, and why |
|---|---|---|---|
| clip1_kickoff | **193/253 = 76.3%** | 168/253 = 66.4% | **+25 frames**; 48 of our 60 refusals are "too few lines" in the rotating tail |
| clip2_congestion | 161/174 = 92.5% | 173/174 = 99.4% | −12; **11 lost to forward-only anchoring** (f0–10) |
| clip3_qb | 166/186 = 89.2% | 183/186 = 98.4% | −17; 13 refused on residual, unexplained |
| clip4_swing | 246/260 = 94.6% | 260/260 = 100% | −14; **12 lost to forward-only anchoring** (f0–11) |
| hail_mary_left | 28/221 = 12.7% | (not his clip) | the contaminated-pencil capture, refusing correctly |

Camera parameters are now stable and reproduce his independent measurements:
focal constant at 1511.0 on every frame of every clip, **elevation median
−28.4 / −28.3 / −28.5°** against his −28.4/−28.5 across four clips.

### The clip4 prediction, tested

The prediction written down before stage 5 existed was: *clip4's +0.37 yd scale
error is cold-fit horizon jitter, not a scale error.* Result:

| route | n | error vs HUD 29 | IQR |
|---|---|---|---|
| through the lattice (stage 4 baseline) | 19 | **+0.37** | 0.36 |
| through the camera (stage 5) | 51 | **+0.18** | 0.34 |

**The claim is confirmed; the predicted mechanism of fix was not.** I expected
the tracker's `fit_labelled` plus horizon prior to remove the jitter. What
actually removed it was measuring through the camera at all: the camera route
inverse-projects the drawn line's crossing of the centre column straight to a
field coordinate, so `(v, alpha, beta)` — and therefore all of their jitter —
never enters. The error halved and the sample size nearly tripled.

Worth separating those two, because "the prediction was right" would otherwise
take credit for a mechanism that turned out to be irrelevant.

clip2 the same way: +0.15 (lattice, n=9) → **+0.17** (camera, n=22), IQR 0.13 →
0.17. Both inside the 0.25 yd bar.

### The walk is forward-only, and it costs measurable coverage

Not a bug — a missing feature, identified by measurement and worth fixing in
stage 6 rather than guessing at now.

`walk.walk` cold-starts at the first anchor-grade frame it finds and goes
forward. Everything before that frame is refused as "no anchor yet". On clip_02
that is frames **0–10**, eleven frames lost because the ladder rms there is
0.98–1.44 px against an anchor bar of 1.0, and the walk then anchors at f11.

Yash does not pay this: his anchors sit **mid-clip** (clip_01 f51, clip_02 f17,
clip_03 f6, clip_04 f0) and he walks outward in *both* directions, so an anchor
found at f17 still covers f0–16. His `auto_anchor` also tries several candidate
frames per play and accepts one only if the *whole play* passes an independent
gate, rather than taking the first frame that clears a per-frame bar.

Both are stage 6 work. The second matters more than the first: a bootstrapped
anchor's own residual is not evidence, and only a whole-clip measurement can
catch a depth-scale error in it.

## Stage 6: bidirectional walking, and a certified count that went DOWN

Bidirectional walking recovered exactly what was predicted on the clips that
lost frames to forward-only anchoring, and **cost certified frames on clip_01**.
Both halves are reported, because the second is the interesting one.

| clip | stage 5 certified | stage 6 certified | TRACKED | refused |
|---|---|---|---|---|
| clip2_congestion | 161 (92.5%) | **171 (98.3%)** | 3 | **0** |
| clip1_kickoff | 193 (76.3%) | **187 (73.9%)** | 14 | 52 |

clip2 behaved as designed: the anchor moved from f11 to the best-quality frame
f38, the eleven frames before the old anchor came back, and nothing is refused
at all. Within two frames of Yash's 99.4%.

### Why clip_01 lost six frames, and why that is probably right

The six are **f142, f144–148** — the exact stretch Yash's own `flow_bridge`
records as unresolvable ("could not resolve clip_01's 16-frame gap f132-148;
no relabel better than 4.2 yd"). They were removed by two gates whose entire
purpose is catching a one-line identity slip:

    f142  disagrees with chained motion 3.6 yd   (CHAIN_AGREE_MAX_YD, new in stage 6)
    f144  moves model 3.9 yd                     (MAX_JUMP_YD)
    f145-148  track age exceeded after the same

**The mechanism is the frozen reference, and it is deliberate.** A refused
frame does not advance the reference, so by f142 the jump is being measured
against f132's pose — ten frames of real camera motion away — and
`MAX_JUMP_YD` is an absolute 3.0 yd that is deliberately NOT divided by the
gap. On a kickoff return, with `pan+zoom` on 231 of 253 frames, that cascade is
structural rather than accidental. Yash tried the obvious fix (advance the
reference on a refused jump) and measured the result: density_max 1.17 → 1.43
and 1.12 → 1.91, and the frames it gained were false accepts.

Two reasons to read this as the gates working rather than a regression:

* **We moved toward Yash, not past him.** He certifies 168 on this clip;
  stage 5 gave 193, stage 6 gives 187. Stage 5 was the outlier.
* **The chain gate is well calibrated.** Over 186 certified frames the chain
  disagreement is median **0.095 yd**, p90 0.263, max 2.07 — far tighter than
  the 0.00–1.28 yd Yash measured for correct solves against ORB, which is what
  a better motion estimator should give.

**Not proven, and the proof is a known instrument.** Whether those six frames
were false accepts in stage 5 can only be settled by an independent referee —
Yash's dense-flow identity check, which asks whether a line's yard index stays
with the pixels. That is stage 7 work alongside the carry pass. **This stretch
is also exactly what the carry pass is for**: ECC turf alignment from the
nearest certified keyframe does not go through the solve, so it does not
inherit the stale reference.

One suspect to carry forward: **f149, the re-acquisition frame**, certified at a
chain disagreement of 2.07 yd against a 2.5 yd gate. Yash warns that
re-acquiring after a gap is where no referee can verify identity.

### The observation cache pays for itself immediately

Separating detection from tracking was forced by the bidirectional walk, and
the measured side effect matters more: on clip_02, `observe` takes 94 s and the
walk itself **8 s**. Re-walking with a threshold moved therefore costs 8 s
instead of 102 s. Both parent projects judge every change against a frozen
baseline, and at the old cost a five-clip re-walk was nine minutes of
recomputing identical detections.

The property that makes the cache sound: **no observation depends on a pose**,
so a re-walk on cached observations is exactly a re-walk and not an
approximation of one. The drawn-line *gap* is deliberately not cached, because
it needs a pose; the segments are cached and the gap computed per walk.

## Stage 7: the referees, and a claim of mine they overturned

### RETRACTED: "clip_01's six lost frames were the gates working"

In the stage 6 notes above I wrote that the six frames stage 6 stopped
certifying (f142, f144–148) were "probably right" to refuse, because they sit
in the stretch Yash's `flow_bridge` could not resolve and because they were
removed by gates built to catch one-line slips. **That was wrong, and both
referees say so.**

Grading **stage 5's** poses on those frames, certified pairs only:

| instrument | result |
|---|---|
| flow identity | **24 pairs judged, 24 consistent, 0 slips** |
| independent paint | grid-to-stripe error **0.082 yd** median, p90 0.151 |
| self-check | clean pose 9.3 px vs an injected +5 yd slip **141.1 px** → flagged |

The pairs inside f142–148 are covered (gaps of 1–2), and the instrument is
validated on a known answer in the same run. So those poses were correct, and
my reading of the refusals as "the gates working" was the comfortable
interpretation rather than the measured one.

### What was actually wrong: the chain was stale, and it was my bug

`H_chain = chain_pose(H_ref, M)` applies **one frame** of motion to the
reference — and the reference is frozen on every refusal, so after a refused
run it is N frames old while `M` covers only the last one. The prediction is
then wrong by N−1 frames of camera motion, the correct solve lands far from it,
and `CHAIN_AGREE_MAX_YD` refuses a good pose. On clip_01 the reference was ten
frames stale by f142 and the gate read 3.6 yd.

Fixed by accumulating motion since the last certified frame, and by making that
accumulated pose the single source for **both** the chain prediction and the
TRACKED tier — they are one quantity and must not be able to disagree about
where the camera is.

Measured effect on clip_01, all at stride 1:

| | certified | TRACKED | refused |
|---|---|---|---|
| stage 5 (forward only) | 193 | — | 60 |
| stage 6, stale chain | **187** | 14 | 52 |
| stage 6, chain fixed | **193** | 12 | 48 |

The six frames come back, and the referee has now confirmed all 193.

**The lesson is not the bug.** It is that a gate firing is not evidence the
gate is right, and "our stricter version refuses more, therefore it is safer"
is exactly the reasoning an independent referee exists to check. I had the
referee available and reached for the interpretation first.

### Walk direction does not bias the pose

The +0.19 / +0.25 yd difference between backward- and forward-walked frames was
**entirely position in the clip**, not direction. Separated by anchoring at the
last anchor-grade frame (so nearly everything is walked backward) and then at
the first, and comparing the same frames:

| clip | n shared | backward | forward | difference | per-frame \|diff\| median |
|---|---|---|---|---|---|
| clip4_swing | 62 | 29.207 | 29.183 | +0.024 | **0.000** |
| clip2_congestion | 32 | 30.233 | 30.209 | +0.023 | **0.000** |

A per-frame median difference of exactly zero means the two directions produce
the same pose, which also validates the motion inversion the backward walk
depends on. Predicted beforehand, and for the right reason: the camera is
re-solved from each frame's own paint, and the chain only supplies labelling,
which moves yards in whole 5-yard steps or not at all.

Still open: why clip4's early frames measure a slightly larger drawn-line gap
than its late frames.

### The carry pass, and a stop rule that was silently inactive

On clip_01, 6 of 60 uncertified frames carried; alignment correlation median
0.966; paint check median 0.173 yd, max 0.274 against a 0.5 stop rule. Seven of
eight runs stop on low correlation — the deep-roll tail cannot be ECC-aligned
to a keyframe, which is the honest outcome rather than a bad pose. (Yash
measured clips 02/03 stopping on correlation before 30 frames as well.)

**The first run of the carry reported 9 frames and no paint check at all**, and
the missing line was the signal: `paint_disagreement_yd` returned `None` on
every frame because the observation cache was empty — `track.py` called
`observe()` directly rather than `observe_cached()`, so nothing was ever
written. The carry's most important stop rule, the only one that consults the
image's own evidence rather than the alignment's self-report, was inactive and
the carry still looked like it was working.

With it active, three of those nine frames are removed, one run stopping on
`paint_disagrees` with a healthy correlation — which is exactly the silent
failure mode the rule exists for (Yash's clip_03 f70 landed 2 yd off at a good
correlation).

**A guard that does not fire is not a guard, and a guard whose input is missing
does not announce itself.** Third instance in this project.

### Density below 1 is not automatically a grid fault

The referee's `density` is grid lines per *independently detected* painted
line, and its denominator is an imperfect detector. On clip_01's high-roll
stretch (f125–160) it reads 0.86 median; on ordinary frames it reads ~1.10:

    f40   10 independent stripes   y = 6, 32,  _, 99, 142, 195, 263, 351, 472, 646, 927
          11 grid lines in span    y = 6, 32, 62, 97, 140, 194, 261, 350, 471, 648, 926
          per-stripe distance to the nearest grid line: 0.3 – 2.1 px

The grid places a line at y=62 that the independent detector missed, in the
near-horizon band where paint is faintest and its coverage test is hardest.
Yash documents exactly this limitation — his stripe detector "misses faint far
lines, giving 2-step gaps", which is why he uses it as a referee and a
fallback, never as a replacement.

So read density for the DANGEROUS direction. A grid 2x too dense reads ~2.0 and
is a real fault that no distance metric can see; a grid slightly above 1
usually means the referee missed a faint line. The ten matched lines at 0.1–2.5
px are the signal worth having.

**An earlier note of mine read 0.86 as "we draw fewer grid lines than there are
painted lines". That was a range artefact** — it was measured only on the
high-roll stretch, and the ordinary-frame figure is above 1, in the other
direction.

### Honest caveat on the flow referee

It is independent for CERTIFIED poses, whose geometry came from paint. It is
**not** independent for TRACKED poses: a TRACKED pose is derived from motion
and the referee grades against motion, so grading one partly measures the
motion estimator against itself. `scripts/referee.py --certified-only` is the
flag that keeps the distinction, and the clip_01 verdict above was taken with
it set.

## clip_03's 13 residual refusals: ANSWERED - our detector

The last substantive gap to Yash on the shared clips (our 166 against his
183/186). The separation at the gate is total:

| | residual |
|---|---|
| certified frames | median **0.64**, p90 1.22, max 2.76 px |
| the 13 refused frames | **6.3 - 22.1 px** |

The gate is 4.0 px, so there is no marginal case: every refused frame misses by
a factor of 1.6 to 5.5. The gate is not borderline-rejecting good frames.

### The experiment, and the answer

`scripts/exp_residual.py 3` scores our pencil against the independently written
stripe detector on exactly those frames, and tries a camera on each set. Of the
three outcomes written down beforehand, it is the first:

    detector 11   frame 0   basin 2

**The independent pencil fits a camera to 0.19-2.99 px - inside the gate - on
the same frames where ours fails at 6-22 px.** Our ladder residual is 3-10x
worse than the independent one on identical frames (f42 8.62 vs 1.16, f50 7.69
vs 1.19, f96 15.03 vs 2.62, f163 9.61 vs 1.15, f166 8.68 vs 0.65).

So **our detector is contaminating the pencil on clip_03, and the fix is in
detection, not in the gates.** Yash reaches 183/186 because his line sets on
these frames are cleaner, not because his gates are looser.

Two honest qualifications: f163 is classified on the ladder-rms signal rather
than its fit (12.59 px on 4 lines, poorly conditioned), and f176/f177 are
untestable because the independent detector found only 3 lines there.

### The candidate fix, not yet tried

The independent detector differs in three ways, and the third is the likely
discriminator:

1. a wider top-hat (25 px at reference width, against our ~15);
2. Hough on the paint mask instead of thinning + LSD;
3. **a COVERAGE TEST** - a line is kept only if at least 50% of its length over
   turf is actually paint.

Our pencil has a length cut (`LONG_CUT_PX`) but no coverage test, so a long
segment that is only partly on paint survives. Adding coverage to our own
detector is the measurable next change. Note it must not simply import the
referee's implementation: that would make the referee share code with what it
grades, which is the whole thing `indep_paint` exists to avoid.

### And my own experiment hid the answer at first

The first run of `exp_residual.py` reported `frame 13` - i.e. "these frames are
just hard" - while the table beside it already showed our ladder residuals
3-10x worse than the independent detector's. Two instrumentation faults:

* It required a `clean` ladder (no multi-step rungs) before attempting a fit.
  But the independent detector **misses faint far lines by design**, so it
  legitimately returns 2-step gaps. A gap is not contamination, and the ladder
  handles multi-step rungs; requiring `clean` meant most frames were never
  tested at all.
* It fitted through `bootstrap_pose`, which refuses below `ANCHOR_MIN_LINES`
  = 6. The independent sets here hold 3-5 lines, so every one returned "fit
  failed" for a reason that had nothing to do with the frame.

Fourth instance of the same lesson in this project: **assume the instrument has
the blind spot before concluding anything about what it measures.** The signal
was in the numbers and the verdict logic overwrote it with a conclusion.

## The complete stage 0-7 picture (2026-10-17)

Run `scripts/report.py`; these are the figures it printed on the day, recorded
for before/after comparison only.

**TWO FIGURES BELOW WERE LATER SUPERSEDED, AND ARE LEFT IN PLACE WITH THIS
POINTER RATHER THAN EDITED**, because a dated record that gets quietly
corrected is how both parents ended up holding three values for one
measurement:

* clip_03's **+2.11 yd** scale error is **withdrawn** - the truth value was
  wrong, not the measurement. The clip does not print 29; see "Fix 1 was a fix
  for a problem that did not exist". It is now ungraded, and shows as "-".
* the grid-to-stripe errors of **0.069 - 0.082 yd** are **centre-column only**
  and therefore angle-blind. Three-column figures are 0.130 - 0.198 yd; see
  "The renderer found what every number had missed".

Everything else in this section still reproduces on the current code.

| clip | frames | CERT | CARRIED | TRACKED | REFUSED | covered | scale err | vs Yash |
|---|---|---|---|---|---|---|---|---|
| clip1_kickoff | 253 | 193 (76.3%) | 6 | 12 | 42 | 83.4% | - | **+25** |
| clip2_congestion | 174 | 171 (98.3%) | 2 | 1 | **0** | **100%** | +0.23 | -2 |
| clip3_qb | 186 | 166 (89.2%) | 17 | 3 | **0** | **100%** | **+2.11** | -17 |
| clip4_swing | 260 | 258 (99.2%) | 1 | 1 | **0** | **100%** | +0.22 | -2 |
| hail_mary_left | 221 | 53 (24.0%) | 9 | 30 | 129 | 41.6% | +0.20 | - |

### RETRACTED: "the referees agree with Yash's own accuracy figures"

| | this project, as measured then | Yash's committed audit |
|---|---|---|
| grid-to-stripe error, median | ~~**0.069 - 0.082 yd**~~ **centre column only** | 0.051 - 0.090 yd |
| grid-to-stripe error, p90 | ~~0.122 - 0.180 yd~~ centre column only | 0.095 - 0.133 yd |
| density, median | 1.00 - 1.11 | 1.00 |
| flow identity slips | **0 of 830 certified pairs** | - |

**The agreement claim is withdrawn, and the paint rows are superseded.** Our
figures came from the centre column alone, where the error is smallest by
construction; three-column worst-case figures are 0.130 - 0.198 yd median.
Whether that is better or worse than his 0.051 - 0.090 is **unknown**, because
nobody has checked whether HIS audit samples the centre or the width. Two
numbers that measure different places are not a comparison.

What survives unchanged: the measurement was made against a detector that
shares no code with ours, on every clip, with the instrument self-validated
each run. The density and slip rows are unaffected - density is a count of grid
lines per painted line and does not depend on where across the frame it is
counted.

**830 certified pairs and zero identity slips** is the headline number, because
a whole-line slip is the one error that every paint-based metric in both parent
projects is structurally blind to.

### The carry recovers where ECC can see, and refuses where it cannot

| clip | uncertified | carried | stopped on |
|---|---|---|---|
| clip3_qb | 20 | **17 (85%)** | low_cc x11, paint_disagrees x2 |
| clip2_congestion | 3 | 2 (67%) | low_cc x2 |
| clip4_swing | 2 | 1 (50%) | low_cc x2 |
| clip1_kickoff | 60 | 6 (10%) | low_cc x7, paint_disagrees x1 |
| hail_mary_left | 168 | 9 (5%) | low_cc x12 |

Paint check on every carried frame stayed inside the 0.5 yd stop rule (medians
0.17-0.31, worst single frame 0.479). Where the carry cannot align it says so
rather than producing a pose: clip_01's deep-roll tail and Hail Mary's long
refused runs stop on correlation, which is the honest outcome.

### clip_03 FAILS the absolute-scale gate at +2.11 yd

Found by adding clip_03 to the graded set rather than leaving it out. Shivam's
`field_template.json` records its HUD truth as 29 with his measured 31.17 at
IQR 4.34, noting "tight pre-snap view; coloured lines partly off-frame". We
measure **31.11 at IQR 0.28** - the same median, a far tighter spread.

**The tight IQR is the informative part.** We are consistently 2.1 yd wrong
rather than scattered, which points at a geometric cause and not noise: a drawn
line clipped at the frame edge has its fitted centroid pulled inward, so the
measured gap is systematically wrong by the same amount on every frame.
`drawnlines.fit_drawn_lines` has no frame-edge test.

Reporting this at all required a correction: clip_03 was showing as "no truth"
in my own table, which hid a failing measurement behind a missing one. **A gate
that silently skips the case it fails is worse than no gate.** Candidate fix,
untested: reject a drawn-line blob whose bounding box touches the picture edge,
or measure the gap only from the portion of each line that is fully inside.

### A weakness in the referee's self-check

The self-check validates on the FIRST clean certified pair it meets, which is
usually f0/f1 - the worst-conditioned place in a clip. On clip_02 and Hail Mary
that pair's clean-pose error is 50.0 and 54.1 px against 3.4-11.4 px elsewhere,
so the margin over the injected slip is only ~2.3x rather than the ~10x seen on
a well-conditioned pair. The verdict is still correct, but a self-check should
be run where the instrument is at its best, not at the first opportunity.

Related, and honest: our flow error on true pairs is **15-20 px median**
against the 6-16 px Yash validated. Probably our half-resolution Farneback
(`FLOW_SCALE = 0.5`). The slip signal is 94-147 px, so the margin is still
5-7x, but the baseline is wider than his and that is worth knowing before
trusting a marginal verdict.

## The two scoped fixes: one disproved, one reverted

Both were attempted, neither shipped, and the reasons are worth more than the
code would have been.

### Fix 1 was a fix for a problem that did not exist

clip_03's +2.11 yd scale error was attributed to a drawn line clipped at the
frame edge. **Disproved by measurement**: on f35 the `orange_yellow` line
touches no frame edge and the gap still reads 30.88, and the gap holds at
30.80-30.91 while the red blob's length jumps 1177 -> 295 px - which is exactly
how an SVD line fit behaves under clipping, since the fitted LINE is unchanged.

The real cause is a **wrong hand-entered truth value**. The HUD alphabet was
learned from clip_04's geometry alone and applied to clip_03's banner, so
clip_03's own geometry never taught the book:

| clip | distance glyph clusters | reads |
|---|---|---|
| clip_04 (teacher) | (0, 5) | **29** - matches its truth |
| clip_03 (subject) | **(3, 4)**, on all 10 text lines | neither digit is 2 or 9 |

**clip_03 does not print 29.** Our 31.11 at IQR 0.28 is consistent with a
printed 31, i.e. an error near 0.1 yd and in line with clip_02 and clip_04.
'3' and '1' cannot be learned from this footage, so the clip cannot be graded
and was removed from the truth table with the evidence recorded. Writing the
frame-edge test would have been code fixing nothing, and it would have been
"validated" by a number that was itself wrong.

Side benefit: this is the **cross-clip digit transfer** that Phase 4 needs,
demonstrated working on real data - teach from one clip's geometry, read
another's text.

### Fix 2, attempt 1: a coverage test, measured VACUOUS

Dropped **0 segments on all 13** target frames; coverage p10 1.00, median
1.00. Structural and predictable in hindsight: our segments are produced BY
thinning and LSD on `prep.mask`, so they lie on it by construction, and
dilating it +-3 px takes coverage to ~1.0. `indep_paint`'s version works only
because it measures against a DIFFERENT, wider mask. Left wired in and inert.

### Fix 2, attempt 2: the split-edge merge, measured NET NEGATIVE and reverted

Comparing our line positions against the referee's found the actual
contaminant - pairs ~10-11 px apart where the referee sees one line (f42 ours
447 and 457 against indep 457; f96 412 and 423 against 426). That is the split
edge Yash documented. On the target frames the ported merge is dramatic:
f42 goes 13 lines / 8.62 ladder rms -> 8 / **1.12**, against the independent
detector's 1.16 on the same frame.

End to end it is not acceptable:

| clip | certified before | after | |
|---|---|---|---|
| clip_03 | 166 | **173** | +7, the intended target |
| hail_mary | 53 | **66** | +13 |
| clip_01 | 193 | 183 | -10 |
| clip_02 | 171 | **124** | **-47** |
| clip_04 | 258 | 258 | 0, but scale +0.22 -> **+0.27**, over the bar |

Reverted, per the standing rule: a change that makes a figure worse is
reverted and rethought, not patched.

**Why it fails here when it works for Yash.** His test fires when a gap is
`< ratio * min(neighbouring gaps)`, and `min(nb)` is fragile when a line is
MISSING: gaps of 100, 15, 100 fire even though the 15 is a real 5-yard gap,
because both neighbours are inflated by absent lines. clip_02 is the
congestion clip, so occluded lines are its defining failure mode. His version
survives because his input is not pre-clustered; ours arrives already merged at
~10 px by `cluster_1d`, so the clearly-sub-threshold gaps his test keys on are
already gone and only ambiguous ones remain.

### Attempt 3 was not made, and the reason is the finding

A median-gap condition was added to tighten it and **measured not to work**:
with gaps of 100, 15, 100 the median IS 100, so it fires anyway. Tested
synthetically before spending a 16-minute run on it.

That result generalises: **from the gap pattern alone, a 15 px gap flanked by
100 px gaps is indistinguishable from a split edge.** No statistical test on
gap sizes can separate them, so no amount of tuning the ratio will.

The discriminator has to be physical, and it is the flank test applied to the
gap: **a split edge has PAINT between its two halves, because it is one
stripe; a real adjacent pair has TURF between them.** That needs the masks, so
it belongs in `markings` at the segment level, not in `propose`, which only
ever sees positions. Not attempted here - three iterations on one fix in a
single pass is how a pipeline acquires changes nobody has measured.

Baseline confirmed restored: clip_02 back to 171 certified (98.3%), 3 TRACKED,
0 refused, scale +0.23, and all 11 camera tests pass.

## The paint-between-the-lines test: premise disproved by its own measurement

Implemented as asked, and it answered a different question than the one it was
built for. Three probes across clip_03 f42, at each candidate's y:

| probe | L | chroma | in line mask |
|---|---|---|---|
| confirmed painted line y=457 | **166.0** | 17.0 | **0.88** |
| our extra line y=447 | **89.5** | 19.0 | **0.03** |
| between them y=452 | 94.5 | 18.4 | 0.02 |
| plain turf | 86-100 | 18.6-19.4 | 0.00-0.03 |

### Finding 1: the pairs are not split edges

y=447 has TURF luminance and is essentially absent from the line mask. It is a
**false line on bare turf**, not the second edge of the stripe at 457. So the
split-edge diagnosis recorded above is **RETRACTED**: the remedy is to reject
that line, not to fuse it. `fuse_edges` is off by default.

It also explains why the reverted gap-ratio merge "improved" f42 from 8.62 to
1.12 ladder rms while costing 47 frames on clip_02: it was **removing a real
observation**, and a 1-D projective fit with three parameters gets cheaper with
every point you take away. Fit quality proves nothing - the oldest lesson in
both parent projects, and it caught me in a new costume.

### Finding 2: chroma does not discriminate paint from turf in this game

Paint reads chroma 17.0 against turf's 18.6-19.4 - no separation at all. And
Blitz's pale-cyan paint has nearly turf CHROMATICITY (Mahalanobis 1.48 against
turf's 1.21-1.38), so `~turf_mask` fires only **0.10** on a real painted line.

My `paint_like` predicate was therefore broken, and its "0 of 13 fused" result
was never evidence about split edges - it was evidence about the predicate. I
took the 26.0 chroma value from `white_line_response`, where it is a SOFT RAMP
multiplying an already-isolating top-hat, and used it as a hard gate that paint
must pass. **A constant lifted out of the expression it was tuned inside is a
different constant.** Fifth instance of the reference-frame class of bug in
this project.

**Luminance is the discriminating channel**: 166 against 86-100.

### Finding 3: why the first coverage test was vacuous, exactly

It sampled along each **segment's own extent**, where a short spurious segment
scores 1.0 by construction - 0 of 13 frames touched, coverage p10 1.00.
`indep_paint.coverage` works because it samples across the **whole frame
width**: y=457 scores 0.88 there and y=447 scores 0.03. Same idea, different
denominator, opposite outcome.

Reimplemented that way (`line_coverage`), it is still **not a clean win**: on
the 13 target frames, 4 better, **1 worse** (f96 20.75 against 15.03), and f42
unchanged at 13 lines / 8.62 despite 59 segments dropped. So it ships OFF
pending a full-clip run, because the precedent is pointed - the split-edge
merge looked better locally (6 of 13 improved) and was catastrophic end to end.

### Why I stopped

Three attempts at one fix, two retracted diagnoses, and the motivating frame
still unfixed. The standing rule covers exactly this: stop and re-plan rather
than stacking a third fix on the second. The repository is left at the verified
baseline - clip_02 back to 171 certified (98.3%), 0 refused, scale +0.23, 11 of
11 tests passing - with all three attempts wired in, inert, and carrying their
measurements.

**What a fourth attempt should do differently.** Every attempt so far has tried
to filter the pencil AFTER detection. The measurements point somewhere else:
y=447 is absent from the line mask (0.03), so it should never have become a
segment at all. It exists because `merge_collinear` extended a short on-mask
fragment into a long line across bare turf. The gap tolerance - 80 px at
reference width, ~113 px here - is the suspect, and the right experiment is to
measure the length distribution of merges that produce low-coverage lines
before changing anything.

## The renderer found what every number had missed

Built stage 10 and looked at the output, which is the check both parent
projects insist on and which this project had been deferring.

### What it confirmed

* On clip_01 f60 the grid lands ON the paint - painted lines at y = 35, 120,
  222, 355 against drawn lines at 37, 120, 222, 350 - and at ONE grid line per
  painted line, not every other. Density and placement both visibly right.
* On clip_01 f215 nothing is drawn and the banner reads "2 lines | too few
  lines (2) for a ladder". The frame is a hard-rolled close-up under a "46
  YARD RETURN" banner. The refusal is correct and the stated reason is the
  true one.

### What it found: our paint referee was ANGLE-BLIND

On clip_02 f95 the drawn lines visibly diverge from the paint toward the frame
edge while the referee reported 0.079 yd. Both were right, because they measure
different places. Measured at three columns:

| column | grid-to-paint |
|---|---|
| x = 0.1W | 1.2 px median |
| **x = 0.5W (centre)** | **0.7 px** |
| x = 0.9W | 2.5 px median, p90 6.1 |

The cause is angular: on THIS FRAME the median grid line angle is -2.91 deg
against the paint's -4.74 deg, a 1.83 deg disagreement, which is 17.4 px of
drift at the frame edge and nothing at all at the centre.

**f95 IS AN OUTLIER AND I GENERALISED FROM IT.** Measured properly, over 8
sampled certified frames per clip against the independent detector:

| clip | grid deg | paint deg | difference |
|---|---|---|---|
| clip1_kickoff | 0.06 | 0.02 | **0.14** |
| clip2_congestion | -1.84 | -2.62 | **0.78** |
| clip3_qb | -0.39 | -0.08 | **0.31** |
| clip4_swing | 0.15 | 0.00 | **0.14** |

The typical disagreement is 0.14-0.78 deg, not 1.83. clip_02 is the worst, as
its rolled and congested camera would predict, and even that sits inside the
0.79 deg p90 our own `line_dir` reports against our own detections. The
referee's blind spot was real and worth fixing; the size of the defect it was
hiding was not what one frame implied.

`grid_vs_paint` sampled only the centre column, so it could not see it. **That
is the exact blind spot Yash records for his own `paint_fit`** - "paint_fit is
angle-blind" - reproduced here without noticing, in a referee built to catch
other people's blind spots.

Fixed: the referee now samples three columns and reports the WORST, because a
grid that is right in the middle and wrong at the edges is wrong.

### The figures this corrects

**Every paint error quoted before this point was centre-column only.** On
clip_02, re-measured:

| | median | p90 |
|---|---|---|
| centre column only (previously reported) | 0.079 yd | 0.136 |
| three columns, worst (honest) | **0.168 yd** | **0.299** |

Roughly double. Do not compare the new figure against Yash's committed
0.051-0.090 median without first checking how HIS audit samples - that
comparison was never verified and is now doubly suspect.

### Two things this does not change

* **The downfield measurement is unaffected.** The absolute-scale check reads
  the drawn-line gap through the camera at the centre column, where the grid is
  accurate to 0.7 px, and it still passes at +0.20 to +0.23 yd against the
  game's own down-and-distance.
* **The gate already bounds this defect**, at `MAX_LINE_DIR_DEG = 3.0`, and
  1.83 deg is inside it. So the frames are not false accepts by the project's
  own standard - but the standard is looser than the referee made it look.

### Why this one matters more than the bugs

Every earlier finding in this file was caught by an instrument. This one was
caught by a screenshot, after the instrument said the frame was fine. Both
parent projects say it plainly and this project had been proving it by
omission: **render the overlay and look at it.** Seven stages of measurement
machinery were built before anything could be looked at.

**And then the looking produced its own over-read.** "A 1.8 deg systematic
angular bias" was ONE FRAME, reported as a property of the pipeline before the
other clips were measured - the third single-point generalisation in this
session, after "the gates were working" and "the contaminant is a split edge".

The pattern is specific enough to be actionable: a striking number from one
sample gets a causal story attached before the sample is widened. The fix is
mechanical - widen the sample BEFORE writing the sentence - and the tell is the
word "systematic" sitting next to an n of 1.

## Open, not resolved

Checked against the current code and the 2026-10-17 run. In rough order of what
would unblock the most:

* **Stage 8 cannot start: not enough digits.** The HUD reader works and the
  cross-clip transfer is demonstrated (clip_04's geometry taught the book that
  read clip_03's banner), but five single-play clips print 2 of 10 digits.
  Needs Yash's clips 05-07 and Shivam's videos 5-7. No code change will fix
  this, and writing code against 2 digits would be code validated by nothing.
* **clip_03's 17-frame gap is a DETECTOR problem, diagnosed and unfixed.** An
  independently written detector fits a camera to those frames at 0.19-2.99 px
  where ours fails at 6-22 px. Three attempts, none shipped, all three left
  inert with their measurements. **The next move is a measurement, not a
  change**: y=447 is absent from the line mask (0.03), so it should never have
  become a segment at all; it exists because `merge_collinear` extended a short
  on-mask fragment across bare turf. Measure the length distribution of merges
  that produce low-coverage lines before touching the gap tolerance.
* **Are our paint figures comparable to Yash's?** Unknown. Ours are now
  three-column worst-case (0.130-0.198 yd median); his committed 0.051-0.090
  may be centre-column, in which case the two numbers measure different
  things. Read his audit before anyone reports a comparison.
* **The referee's self-check runs at the first clean pair**, usually f0/f1,
  which is the worst-conditioned place in a clip - margin ~2.3x on clip_02 and
  Hail Mary against ~28x on clip_03. The verdict is still correct; a self-check
  should run where the instrument is at its best.
* **Our flow error on true pairs is 15-20 px median** against the 6-16 px Yash
  validated, probably the half-resolution Farneback (`FLOW_SCALE = 0.5`). The
  slip signal is 94-147 px so the margin holds, but the baseline is wider than
  his.
* **Hail Mary over-detects.** 605 raw segments per frame against 137-244 on the
  clips, 27 long lines against a documented 10, after the scale fix. Hypothesis,
  untested: Shivam's documented figure is from his render *with* the trained net,
  which has Hail Mary rows, while this pipeline is on the colour path. Competing
  hypothesis: the dark grade (turf L 19) gives compression noise a relatively
  larger top-hat response. It is now checkable in a way it was not before -
  there is a pose to judge the lines against - and 24% certified with a single
  138-frame refused run is the visible cost.
* **Residual proposer disagreement on clip_01 and clip_03** (~11% of frames
  where both answer) has not been broken down. It may be stray lines, or the
  same class of instrument error as bug 3 above. Unknown.
* **`det_scale` is applied to lengths and areas but assumes isotropy.** Correct
  for every source here (anisotropy <= 1.0028); it is not correct for an
  anisotropically stretched upload, where the camera model's (fx, fy) handles
  the geometry but a circular morphology kernel is still wrong.
* **Stage 9 (the metric template) has not been started**, and nothing in stages
  0-7 depends on it.
