# NFL Gameplay Analyzer — merged field tracker

Takes NFL Blitz (1997 arcade) gameplay video and works out where the football
field is: the yard lines, their spacing, their identity relative to each other,
and a metric camera that maps pixels to yards. Classical geometry throughout;
the one learned component is wired in and **off by default**.

Every frame leaves the pipeline in one of four tiers with a named evidence
standard, and frames that cannot be solved are **refused by name** rather than
filled in:

| tier | what it means | drawn as |
|---|---|---|
| **CERTIFIED** | camera solved from this frame's own paint, all 8 gates passed | green grid |
| **CARRIED** | ECC turf alignment from a certified keyframe, re-checked against visible paint, bounded at 30 frames | blue grid |
| **TRACKED** | pose moved through measured optical flow; **no camera claim** | amber grid |
| **REFUSED** | the evidence is not there; the banner names the gate | nothing drawn |

This merges two independent trackers — see [`../MERGE_PLAN.md`](../MERGE_PLAN.md)
for the comparison that motivated it and the phased plan being followed:

* **`Field_Tracker/`** (Shivam Deshmukh) — lattice tracker: comb-score yard
  lattice in horizon-relative inverse coordinates, learned/colour front end,
  unsupervised gating, HUD text reader, parallel render.
* **`Yash_Field_Tracker/easely-done/`** (Yash Kamerkar) — camera tracker:
  6-parameter pinhole solve per frame, named refusal gates, measured metric
  field template, independent referees.

**New here?** Read this page, then [`docs/LESSONS.md`](docs/LESSONS.md) — every
measurement this project has made and every claim it has retracted, including
the four silent bugs it caught with instruments and the fifth it caught only by
looking at a rendered frame. Then both parents' lessons files before changing
anything geometric.

---

## Status

Stages 0–7 and 10 are built and verified against real footage; the five
annotated videos in `out/` were regenerated from the current code. Stage 8
(absolute yard naming) is **blocked on data**, not on code: five single-play
clips yield only 2 of 10 HUD digits, so it needs the long captures (Yash's
clips 05–07, Shivam's videos 5–7).

| stage | module | state |
|---|---|---|
| 0 · ingest | `ingest.py` | **done** — active picture, invertible transform, (fx, fy) |
| 1 · colour + gating | `fieldcolor.py`, `gating.py` | **done** — per-video turf model, unsupervised play/menu gate |
| 2 · HUD mask | `hudmask.py` | **done** — learned persistence × stability |
| 3 · paint detection | `markings.py`, `geometry.py` | **done** — ridge → thin → LSD → merge → flank test → VP |
| 4 · identity proposers | `lattice.py`, `ladder.py`, `propose.py` | **done** — both proposers + cross-check |
| 4R · **scale referee** | `scalecheck.py`, `drawnlines.py`, `hudtext.py` | **done** — vs the game's own down-and-distance |
| 5 · **certify (camera + 8 gates)** | `camera.py`, `walk.py` | **done** — 6-param solve, every refusal named |
| 5T · synthetic tests | `tests/test_camera.py` | **done** — 11 tests, no footage needed, 2 are degeneracy tripwires |
| 6 · **track (LK motion, bidirectional)** | `motion.py`, `observe.py`, `walk.py` | **done** — anchor mid-play, walk both ways, TRACKED tier |
| 7 · **carry + referees** | `carry.py`, `flowcheck.py`, `indep_paint.py` | **done** — CARRIED tier, flow identity + independent paint |
| 8 · name (absolute yards) | — | **blocked: needs the long captures** |
| 9 · metric template | — | not started |
| 10 · **render** | `render.py`, `scripts/render_video.py` | **done** — four-tier overlay, x264 via ffmpeg |

### What the pipeline produced

Printed by `scripts/report.py` on **2026-10-17**, from a clean `--refresh`
re-run of all five clips followed by carry → referee → render. Recorded here
for before/after comparison only — **run the script, do not read these**:

| clip | frames | CERT | CARRIED | TRACKED | REFUSED | covered | scale err | vs Yash |
|---|---|---|---|---|---|---|---|---|
| clip1_kickoff | 253 | 193 (76.3%) | 6 | 12 | 42 | 83.4% | — | **+25** |
| clip2_congestion | 174 | 171 (98.3%) | 2 | 1 | **0** | **100%** | +0.23 | −2 |
| clip3_qb | 186 | 166 (89.2%) | 17 | 3 | **0** | **100%** | — | −17 |
| clip4_swing | 260 | 258 (99.2%) | 1 | 1 | **0** | **100%** | +0.22 | −2 |
| hail_mary_left | 221 | 53 (24.0%) | 9 | 30 | 129 | 41.6% | +0.20 | — |

*covered* = CERTIFIED + CARRIED + TRACKED, and it is **not** an accuracy
figure: a TRACKED frame makes no camera claim at all. *vs Yash* is our
certified count minus his committed count on the same four clip files (he has
no Hail Mary; we have none of his clips 05–07). *scale err* is yards against
the game's own printed down-and-distance; clip3 is deliberately ungraded and
the reason is recorded in `scripts/report.py` beside the truth table.

Across all five clips: **830 certified pairs judged by the flow referee, 0
identity slips**, with the instrument self-validated on an injected 5-yard slip
before each run.

### The videos

`out/<clip>_field.mp4`, at picture resolution with the original audio:

| file | MB |
|---|---|
| `clip1_kickoff_field.mp4` | 8.8 |
| `clip2_congestion_field.mp4` | 6.0 |
| `clip3_qb_field.mp4` | 5.9 |
| `clip4_swing_field.mp4` | 8.8 |
| `hail_mary_left_field.mp4` | 11.7 |

**Watch them before trusting any number on this page.** That is not a
formality. This project shipped seven stages of measurement machinery before
anything could be looked at, and the first rendered frame exposed a blind spot
in its own paint referee that every numeric check had passed — the referee
sampled only the centre column, so a 1.83° angular disagreement worth 17 px of
drift at the frame edge read as 0.7 px. See `docs/LESSONS.md`, *"The renderer
found what every number had missed"*.

While watching:

* do the green lines sit **on** the paint, at every depth — not just near the
  middle of the frame?
* one grid line **per** painted line, or every other one?
* does the geometry jump at a tier change (green → blue → amber)?
* where nothing is drawn, is the banner's stated reason plausible?

No yard numbers are printed, and the caption says so: painted numbers are
mirror-symmetric, so a "40" sits at yard 40 and at yard 60, and absolute
identity is stage 8. Showing a confident "40" the system cannot stand behind
would be the most misleading thing it could do.

## Quick start

Prerequisites: Python 3.13 (verified 3.13.9), **`opencv-contrib-python`** — the
detector needs `ximgproc.thinning` and `LSD_REFINE_STD`, and a plain
`opencv-python` install fails with an `AttributeError` three stages in — and
**`ffmpeg` on `PATH`** for stage 10 only (verified with 4.3.1).

```bash
C:/Users/shiva/anaconda3/python.exe -m venv nfl_venv
./nfl_venv/Scripts/python.exe -m pip install -r requirements.txt
```

The full chain, in order. Each step reads only what the one before it left on
disk, so any step can be re-run on its own:

```bash
./nfl_venv/Scripts/python.exe scripts/prepare.py all               # stages 0-2, ~2 min
./nfl_venv/Scripts/python.exe scripts/measure_proposers.py all --stride=5
./nfl_venv/Scripts/python.exe scripts/valid_scale.py 2 4 5         # THE PROMOTION GATE
./nfl_venv/Scripts/python.exe scripts/read_hud.py all              # decode the HUD alphabet
./nfl_venv/Scripts/python.exe tests/test_camera.py                 # 11 synthetic tests, no footage
./nfl_venv/Scripts/python.exe scripts/track.py all --refresh       # solve-or-refuse per frame
./nfl_venv/Scripts/python.exe scripts/carry.py all                 # the CARRIED tier
./nfl_venv/Scripts/python.exe scripts/referee.py all --certified-only
./nfl_venv/Scripts/python.exe scripts/render_video.py all          # the videos
./nfl_venv/Scripts/python.exe scripts/report.py                    # the headline table
```

**`track.py --refresh` after any change to the detector or the colour front
end.** Detection is cached per clip (`stage6/observations.json.gz`) and the
observations depend on no pose, which is what makes a re-walk exact and cheap —
on clip2, detection takes 94 s and the walk itself 8 s — but it also means a
detector change is invisible without `--refresh`.

`render_video.py --half-step` additionally draws the 2.5 yd control lines,
which must fall **between** the painted lines. If they land on paint the grid
is half-scaled — a failure no distance metric can see, because a half-scaled
grid still lands on real paint.

`scripts/referee.py` validates itself on a known injected 5-yard slip before
grading anything, every run. **Pass `--certified-only`**: the flow referee is
independent for certified poses, whose geometry came from paint, but not for
TRACKED ones, which are derived from motion and would be graded against motion.

`tests/test_camera.py` needs no footage. Two of its tests are tripwires on
documented degeneracies: if `test_focal_is_not_constrained_by_yard_lines` ever
fails, the model has **gained** a constraint and `config.GAME_FOCAL_PX` must be
re-derived rather than trusted. Exit codes are load-bearing — 0 pass, 1 real
failure, 2 infrastructure — and are never conflated.

`valid_scale.py` is the gate: no change to the front end or the geometry ships
unless the error there does not move backwards. It separates two questions that
are easy to conflate — whether the error is within the 0.25 yd **bar**, and
whether it is worse than the recorded **baseline** (`--set-baseline`, stored in
`work/scale_baseline.json`). A first run has nothing to regress from.

`Data/` holds the source videos. `work/<NN>_<slug>/stageN/` holds everything the
stages hand each other and is rebuildable, so it can be deleted at any time.
`out/` is for deliverables only. All three are git-ignored.

## Two reference widths — read this before touching a constant

The merge inherits pixel-valued constants from two projects that measured them
at different resolutions, and conflating them has already caused three silent
bugs (`docs/LESSONS.md`). There are therefore exactly two named conversions in
`config.py` and no bare numbers:

| | reference | converts with | used by |
|---|---|---|---|
| camera / geometry gates | `PX_REF_W = 1356` | `px_scale(w)` | the 6-param solve, its residual and jump gates, the focal |
| detection constants | `SHIVAM_REF_W = 960` | `det_scale(w)` | LSD lengths, merge tolerances, flank probe, morphology |

Clips 1–4 are natively 1356×1016 with no pillarbox, so `px_scale == 1.0000` on
them and **every camera gate is used at exactly the value it was measured at.**
That is why this project works in the *picture frame* (the active content box at
native resolution) rather than resampling to a canonical buffer: it keeps our
numbers on clips 1–4 directly comparable to Yash's committed results, which is
the only free validation the merge has. Hail Mary is 1920×1080 letterboxed to a
1444×1080 picture at x=238, so `px_scale = 1.0649` there and every gate scales.

## What is verified

**Do not read figures out of this file.** Run the script named beside each
claim. Both parent projects carried figures from superseded runs more than
once, and one of them holds three different values for the same measurement in
three files. This project has already corrected two of its own documented
numbers for the same reason, and one of those corrections is below.

```bash
./nfl_venv/Scripts/python.exe scripts/report.py            # the headline table
```

* **Stage 0 reproduces both parents independently.** Hail Mary's pillarbox crop
  comes out 1444×1080 at x=238 where Yash's unrelated `content_box` measured
  1443×1080 at x=239 on a comparable capture, and its focal 1609.1 px against
  his computed 1608. Frame counts match both parents exactly.
* **Stage 1 reproduces the turf chromaticity** Shivam documents: (110, 126) on
  the clips, (119.37, 132.85) against his (119.3, 132.9) on Hail Mary.
* **Stage 5 recovers the camera Yash measured**, on all five clips:

  | clip | certified | focal px | elevation | height | azimuth span | residual med / p90 |
  |---|---|---|---|---|---|---|
  | clip1_kickoff | 193 | 1511.0 | −28.38° | 10.9 yd | 29.5° | 0.68 / 1.27 px |
  | clip2_congestion | 171 | 1511.0 | −28.39° | 14.2 yd | 16.0° | 0.68 / 1.11 px |
  | clip3_qb | 166 | 1511.0 | −28.57° | 12.0 yd | 21.4° | 0.64 / 1.20 px |
  | clip4_swing | 258 | 1511.0 | −28.46° | 16.4 yd | 9.7° | 0.67 / 1.19 px |
  | hail_mary_left | 53 | 1609.1 | −28.55° | 16.1 yd | 4.3° | 0.88 / 1.31 px |

  Elevation lands within 0.2° of Yash's measured −28.4°/−28.5° on every clip,
  the focal holds one value per clip (it is **locked**, not fitted — see
  below), and roll stays inside ±1.6° everywhere. Both documented degeneracies
  are pinned by synthetic tests: the scale degeneracy exact to 1.7e-13 px, the
  focal unobservable to 8.5e-14 px across a 6× range.
* **The focal is locked because it is not observable from yard lines.** Pin it
  anywhere from 700 to 4200 px and the line positions do not move: Yash
  measured a camera height swing of 18.7 → 7.0 yd on real frames, and
  `test_focal_is_not_constrained_by_yard_lines` reproduces it synthetically at
  7.2 → 18.8 yd with a residual spread of 8.5e-14 px. No position-based metric
  can catch a wrong focal, which is exactly how this project once certified 354
  frames at a focal of 700.0 and a −49.4° elevation with nothing complaining.
* **The two identity proposers agree on 88–100% of clips 1–4** where both
  answer — a comb-score lattice and a beam-search ladder that share no code.
  On Hail Mary they agree on 33% and both say why.
* **The absolute-scale referee passes** on the clips that have a drawn-line
  pair, measured through the camera against down-and-distance the geometry
  never sees: +0.23, +0.22 and +0.20 yd at IQR 0.17–0.27 over n = 32/63/36.
  `scripts/valid_scale.py` is the promotion gate.
* **The flow referee found 0 identity slips in 830 certified pairs**, and
  validated itself on an injected 5-yard slip before every run — the clean pose
  reads 3.4–54.1 px against 94.1–146.9 px for the slip, on all five clips. A
  whole-line slip is the one error that every paint-based metric in both parent
  projects is structurally blind to, because a grid shifted by one painted line
  still lands on painted lines.
* **The independent paint referee reads 0.130–0.198 yd median** grid-to-stripe
  error at density 1.00–1.08, against a stripe detector that shares no code
  with the tracker. **These are three-column figures and they supersede the
  0.069–0.082 yd quoted earlier**, which was centre-column only and therefore
  angle-blind. Do not compare them with Yash's committed 0.051–0.090 median
  until someone checks how *his* audit samples across the frame; that
  comparison was never verified and is now doubly suspect.
* **Walk direction does not bias the pose**: per-frame median difference
  between a fully-backward and a fully-forward walk is exactly 0.000 yd on both
  clips tested (`scripts/exp_direction.py`).
* **The carry recovers where ECC can see and refuses where it cannot** — 17 of
  clip3's 20 uncertified frames, 6 of clip1's 60, 9 of Hail Mary's 168 — with
  the paint check on every carried frame inside the 0.5 yd stop rule (per-clip
  medians 0.173–0.310 yd, worst single frame 0.479).

## Where it refuses, and why

Refusals are not noise, and on these clips they are not scattered either:

* **clip1_kickoff** refuses frames 208–252 plus three singletons, every one of
  them `too few lines (0–3) for a ladder`. That is the hard-rolled close-up tail
  under a "46 YARD RETURN" banner. This is the rotation clip: the paint reaches
  32.8° of apparent tilt, the pipeline certifies through about 12.7° of it, and
  above 10° it refuses 46 frames of 57. The carry recovers the three singleton
  refusals at f200/202/206 and **nothing** in the long run — where the turf has
  rolled out of alignment, ECC correctly declines rather than producing a pose.
* **hail_mary_left** refuses one long run, frames 83–220, as `track age > 12`
  once the detector stops producing a solvable line set. It is a different
  capture with a dark desaturated grade (turf L ≈ 19) and it over-detects badly
  — 605 raw segments per frame against 137–244 on the clips. 24% certified is
  the honest number for a capture the detector was not tuned on, and it is
  reported rather than smoothed over.
* **clips 2, 3 and 4 refuse nothing**: every frame is CERTIFIED, CARRIED or
  TRACKED.

## Open, blocked, and deliberately inert

* **Stage 8 is blocked on data.** The HUD reader works — it learned clip4's
  digit alphabet from clip4's geometry alone and read clip3's banner with it,
  which is exactly the cross-clip transfer stage 8 needs — but five single-play
  clips print only 2 of the 10 digits. It needs the long captures.
* **clip3's 17-frame certified gap** to Yash is diagnosed and unfixed: an
  independently written detector fits a camera to those frames at 0.19–2.99 px
  where ours fails at 6–22 px, so the problem is in detection, not in the
  camera. Three fixes were attempted and **none shipped**; the next step is
  specified in `docs/LESSONS.md` and is a measurement, not a change. The carry
  pass also recovers 17 frames on that clip — the equal count is a coincidence
  and the overlap between the two sets has not been checked — but it places
  them in CARRIED, so it closes the *coverage* gap and not the certification
  gap.
* **Whether our paint figures are comparable to Yash's** is unknown until
  someone reads how his audit samples across the frame.
* **The referee's self-check runs at the first clean pair**, usually f0/f1 —
  the worst-conditioned place in a clip. The verdict is still correct, but the
  margin over the injected slip is 2.3× on clip2 and 2.1× on Hail Mary against
  28× on clip3. A self-check should run where the instrument is at its best,
  not at the first opportunity.
* **Code that is wired in and inert, on purpose.** Each carries the measurement
  that retired it, because a disproved idea with its measurement beside it is
  worth more than a deleted one:

  | | default | why it is off |
  |---|---|---|
  | `markings.COVERAGE_MIN` | `None` | full-width coverage test: of 13 target frames 4 better, **1 worse**, motivating frame unchanged — not a clean win, and the precedent below is pointed |
  | `markings.analyse_frame(fuse_edges=)` | `False` | split-edge merge: premise **disproved**. The tight pairs are false lines on bare turf (luma 89.5, 0.03 of the line mask), not two edges of one stripe |
  | `propose.SPLIT_EDGE_ENABLED` | `False` | gap-ratio merge: clip3 +7 and Hail Mary +13, but clip1 −10 and **clip2 −47**. Reverted rather than patched |
  | `markings.paint_like` | unused | chroma does not discriminate paint (17.0) from turf (18.6–19.4) in this game; luminance does (166 against 86–100) |
  | the learned front end | no `net` passed | the colour path is the default on any capture without its own auto-labelled rows, which has been the right default three times |

* **`det_scale` assumes isotropy.** Correct for every source here (anisotropy
  ≤ 1.0028); not correct for an anisotropically stretched upload, where (fx, fy)
  handles the geometry but a circular morphology kernel is still wrong.
* **Player positions are out of scope** until the field pipeline is frozen.

## The rules this project runs by

Inherited from both parents, which arrived at most of them independently:

* **A false accept is far worse than a false refusal.** Never widen a gate to
  raise coverage. A frame gained by relaxing a gate is a false accept until an
  independent referee says otherwise.
* **Never quote a headline number from a markdown file.** Run the script.
* **Fit quality proves nothing.** A homography fitted to 4 points has zero
  residual by construction, and a 1-D projective fit gets *cheaper* every time
  you remove a point — which is how a change that deleted real observations
  once looked like an improvement.
* **Assume the current metric has a blind spot, and go find it.** Every metric
  built in either parent hid the next defect. This project's third bug was the
  comparison instrument rather than the thing compared, and its fifth was found
  by a screenshot after the instrument said the frame was fine.
* **Render the overlay and look at it.** Both parents say it plainly; this
  project proved it by omission.
* **Measured constants carry their measurement beside them in the code.** Do
  not change one without re-measuring, and write the new measurement next to
  it. A constant lifted out of the expression it was tuned inside is a
  *different* constant.
* **One run is not a verdict**, and one frame is not a pattern. The tell is the
  word "systematic" next to an n of 1 — it has appeared three times in this
  project's own notes and was wrong every time.
* **Revert and rethink; do not stack a second fix on the first.**
* Output is **tiered** and reported as four numbers, never one. Adding them
  reports the weakest as if it were the strongest.
* **Certify on `solve_ok`, never on `applied`.** `applied` is true whenever roll
  alone could be refreshed, which is nearly every frame; conflating the two
  certified 68 of 798 frames carrying a held-over pose.
* **A refused frame freezes the reference, it does not advance it.** Separating
  those two was tried and measured: density_max went 1.17 → 1.43 and 1.12 →
  1.91, and the frames gained were false accepts.

## Layout

```
Data/                  source videos (5 registered; git-ignored)
nflfield/              the library
  config.py              paths, measured constants, the two reference widths
  ingest.py            [0] active picture, invertible transform, (fx, fy)
  fieldcolor.py        [1] turf model, ridge response, drawn-line masks
  gating.py            [1] gameplay vs menus, per-frame field test
  hudmask.py           [2] static HUD overlay map
  geometry.py              projective helpers, vanishing-point RANSAC
  markings.py          [3] paint -> segments -> flank test -> pencil
  lattice.py           [4] proposer A: comb score (Shivam)
  ladder.py            [4] proposer B: beam search + impostor rejection (Yash)
  propose.py           [4] run both, compare the rung patterns
  drawnlines.py            the game's drawn scrimmage / first-down lines
  hudtext.py               HUD alphabet, printed distance, absolute position
  scalecheck.py            the absolute-scale referee
  camera.py            [5] the 6-param camera and all eight refusal gates
  motion.py            [6] LK optical flow (replaces the ORB chain)
  observe.py           [6] one decode pass -> per-frame observations (cached)
  walk.py              [6] anchor a play, track outward both directions
  carry.py             [7] ECC turf alignment from a certified keyframe
  flowcheck.py         [7] dense-flow yard IDENTITY referee
  indep_paint.py       [7] an independently written stripe detector
  render.py           [10] four-tier overlay; draws nothing on a refused frame
scripts/
  prepare.py             stages 0-2 for one or more videos
  measure_proposers.py   the proposer agreement instrument
  valid_scale.py         THE PROMOTION GATE: measured gap vs the printed gap
  read_hud.py            decode the HUD alphabet; absolute field position
  track.py               the walk; --refresh after any detector change
  carry.py               apply the carry pass to a walk result
  referee.py             the two independent referees, self-validated each run
  render_video.py        the annotated videos; --half-step for the 2.5 yd control
  report.py              the headline table, read off disk
  exp_direction.py       does walk direction bias the scale? (no)
  exp_residual.py        is a residual refusal our detector, or the frame?
tests/
  test_camera.py         11 synthetic tests; 2 are degeneracy tripwires
work/                  per-video intermediates and run logs, rebuildable
out/                   the annotated videos
docs/LESSONS.md        what this project measured, and what it got wrong
```
