# NFL Gameplay Analyzer — merged field tracker

Takes NFL Blitz (1997 arcade) gameplay video and works out where the football
field is: the yard lines, their identity, and a metric camera that maps pixels
to yards. Classical geometry throughout; the one learned component is optional
and off by default.

This merges two independent trackers — see [`../MERGE_PLAN.md`](../MERGE_PLAN.md)
for the comparison that motivated it and the phased plan being followed:

* **`Field_Tracker/`** (Shivam Deshmukh) — lattice tracker: comb-score yard
  lattice in horizon-relative inverse coordinates, learned/colour front end,
  unsupervised gating, HUD text reader, parallel render.
* **`Yash_Field_Tracker/easely-done/`** (Yash Kamerkar) — camera tracker:
  6-parameter pinhole solve per frame, named refusal gates, measured metric
  field template, independent referees.

**New here?** Read this page, then [`docs/LESSONS.md`](docs/LESSONS.md) — what
this project has measured, and the three silent bugs it has already caught.
Then both parents' lessons files before changing anything geometric.

---

## Status

Stages 0–7 and 10 are built and verified against real footage. Five annotated
videos are in `out/`. Stage 8 (absolute yard naming) is **blocked on data**,
not on code: five single-play clips yield only 2 of 10 HUD digits, so it needs
the long captures (Yash's clips 05–07, Shivam's videos 5–7).

**Watch the videos before trusting any number here.** That is not a formality:
this project shipped seven stages of measurement machinery before anything
could be looked at, and the first rendered frame exposed a blind spot in its
own referee that every numeric check had passed. See `docs/LESSONS.md`.

| stage | module | state |
|---|---|---|
| 0 · ingest | `ingest.py` | **done** — active picture, invertible transform, (fx, fy) |
| 1 · colour + gating | `fieldcolor.py`, `gating.py` | **done** — per-video turf model, unsupervised play/menu gate |
| 2 · HUD mask | `hudmask.py` | **done** — learned persistence × stability |
| 3 · paint detection | `markings.py`, `geometry.py` | **done** — ridge → thin → LSD → merge → flank test → VP |
| 4 · identity proposers | `lattice.py`, `ladder.py`, `propose.py` | **done** — both proposers + cross-check |
| 4R · **scale referee** | `scalecheck.py`, `drawnlines.py`, `hudtext.py` | **done** — vs the game's own down-and-distance |
| 5 · **certify (camera + 8 gates)** | `camera.py`, `walk.py` | **done** — 6-param solve, every refusal named |
| 6 · **track (LK motion, bidirectional)** | `motion.py`, `observe.py`, `walk.py` | **done** — anchor mid-play, walk both ways, TRACKED tier |
| 7 · **carry + referees** | `carry.py`, `flowcheck.py`, `indep_paint.py` | **done** — CARRIED tier, flow identity + independent paint |
| 8 · name (absolute yards) | — | **blocked: needs the long captures** |
| 9 · template | — | |
| 10 · **render** | `render.py`, `scripts/render_video.py` | **done** — four-tier overlay, x264 via ffmpeg |

## Quick start

```bash
C:/Users/shiva/anaconda3/python.exe -m venv nfl_venv
./nfl_venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
./nfl_venv/Scripts/python.exe scripts/prepare.py all          # stages 0-2, ~2 min
./nfl_venv/Scripts/python.exe scripts/measure_proposers.py all --stride=5
./nfl_venv/Scripts/python.exe scripts/valid_scale.py 2 4 5    # THE PROMOTION GATE
./nfl_venv/Scripts/python.exe scripts/read_hud.py all         # decode the HUD alphabet
./nfl_venv/Scripts/python.exe scripts/track.py all            # solve-or-refuse per frame
./nfl_venv/Scripts/python.exe tests/test_camera.py            # 11 synthetic tests
./nfl_venv/Scripts/python.exe scripts/carry.py all            # the CARRIED tier
./nfl_venv/Scripts/python.exe scripts/referee.py all --certified-only
./nfl_venv/Scripts/python.exe scripts/render_video.py all     # the videos
./nfl_venv/Scripts/python.exe scripts/report.py               # the headline table
```

`render_video.py --half-step` additionally draws the 2.5 yd control lines,
which must fall BETWEEN the painted lines. If they land ON paint the grid is
half-scaled — a failure no distance metric can see, because a half-scaled grid
still lands on real paint.

`scripts/referee.py` validates itself on a known injected 5-yard slip before
grading anything, every run. **Pass `--certified-only`**: the flow referee is
independent for certified poses, whose geometry came from paint, but not for
TRACKED ones, which are derived from motion and would be graded against motion.

`tests/test_camera.py` needs no footage. Two of its tests are tripwires on
documented degeneracies: if `test_focal_is_not_constrained_by_yard_lines` ever
fails, the model has GAINED a constraint and `config.GAME_FOCAL_PX` must be
re-derived rather than trusted. Exit codes are load-bearing — 0 pass, 1 real
failure, 2 infrastructure — and are never conflated.

`valid_scale.py` is the gate: no change to the front end or the geometry ships
unless the error there does not move backwards. It separates two questions that
are easy to conflate — whether the error is within the 0.25 yd **bar**, and
whether it is worse than the recorded **baseline** (`--set-baseline`). A first
run has nothing to regress from.

`Data/` holds the source videos. `work/<NN>_<slug>/stageN/` holds everything the
stages hand each other and is rebuildable, so it can be deleted at any time.
`out/` is for deliverables only.

**Interpreter matters.** This needs `opencv-contrib-python` for
`ximgproc.thinning` and `LSD_REFINE_STD`; a plain `opencv-python` install
cannot run the pipeline, and the failure is an `AttributeError` three stages in.

## Two reference widths — read this before touching a constant

The merge inherits pixel-valued constants from two projects that measured them
at different resolutions, and conflating them has already caused three silent
bugs (`docs/LESSONS.md`). There are therefore exactly two named conversions in
`config.py` and no bare numbers:

| | reference | converts with | used by |
|---|---|---|---|
| camera / geometry gates | `PX_REF_W = 1356` | `px_scale(w)` | the 6-param solve, its residual and jump gates, the focal |
| detection constants | `SHIVAM_REF_W = 960` | `det_scale(w)` | LSD lengths, merge tolerances, flank probe, morphology |

Clips 1–4 are natively 1356 px wide, so `px_scale == 1.0000` on them and **every
camera gate is used at exactly the value it was measured at.** That is why this
project works in the *picture frame* (the active content box at native
resolution) rather than resampling to a canonical buffer: it keeps our numbers
on clips 1–4 directly comparable to Yash's committed results, which is the only
free validation the merge has.

## What is verified so far

**Do not read figures out of this file.** Run the script named beside each
claim; `scripts/report.py` prints the current headline table from whatever the
stages have left on disk. Both parent projects carried figures from superseded
runs more than once, and one of them holds three different values for the same
measurement in three files. This project has already corrected two of its own
documented numbers for the same reason.

```bash
./nfl_venv/Scripts/python.exe scripts/report.py            # the headline table
```

What has been *established*, as distinct from measured-today:

* **Stage 0 reproduces both parents independently.** Hail Mary's pillarbox crop
  comes out 1444x1080 at x=238 where Yash's unrelated `content_box` measured
  1443x1080 at x=239 on a comparable capture, and its focal 1609.1 px against
  his computed 1608. Frame counts match both parents exactly.
* **Stage 1 reproduces the turf chromaticity** Shivam documents: (110, 126) on
  the clips, (119.37, 132.85) against his (119.3, 132.9) on Hail Mary.
* **Stage 5 recovers the camera Yash measured.** Focal constant at 1511.0 on
  every certified frame; elevation median -28.3 to -28.5 deg against his
  -28.4/-28.5 across four clips. Both documented degeneracies are pinned by
  synthetic tests - scale degeneracy exact to 1.7e-13 px, and the focal
  unobservable to 8.5e-14 px across a 6x range.
* **The two identity proposers agree on 88-100% of clips 1-4** where both
  answer. On Hail Mary they agree on 33% and both say why.
* **The absolute-scale referee passes** on the clips that have a drawn-line
  pair, measured through the camera against down-and-distance the geometry
  never sees. `scripts/valid_scale.py` is the promotion gate.
* **Walk direction does not bias the pose**: per-frame median difference
  between a fully-backward and a fully-forward walk is exactly 0.000 yd on
  both clips tested (`scripts/exp_direction.py`).
* **The flow referee validates itself** on an injected 5-yard slip every run -
  clean pose ~9 px against ~141 px for the slip - and found 0 slips in 24
  certified pairs over clip_01's hardest stretch.

## The rules this project runs by

Inherited from both parents, which arrived at most of them independently:

* **A false accept is far worse than a false refusal.** Never widen a gate to
  raise coverage. A frame gained by relaxing a gate is a false accept until an
  independent referee says otherwise.
* **Never quote a headline number from a markdown file.** Run the script. Both
  parents carried figures from superseded runs more than once, and one of them
  has three different values for the same measurement in three documents.
* **Fit quality proves nothing.** A homography fitted to 4 points has zero
  residual by construction. Only a measurement against something independent
  means anything.
* **Assume the current metric has a blind spot, and go find it.** Every metric
  built in either parent hid the next defect, and this project's third bug was
  the comparison instrument, not the thing compared.
* **Measured constants carry their measurement beside them in the code.** Do
  not change one without re-measuring, and write the new measurement next to it.
* **One run is not a verdict.** A rejected configuration once passed every
  criterion on a single run and was the best of five samples from a
  distribution whose median fails.
* Output is **tiered** — CERTIFIED / CARRIED / TRACKED / REFUSED, each with a
  named evidence standard — and reported as four numbers, never one. Only
  CERTIFIED and REFUSED exist yet, so today's certified fraction is a **floor**
  on the merged pipeline's coverage, not an estimate of it. See MERGE_PLAN.md §7.
* **Certify on `solve_ok`, never on `applied`.** `applied` is true whenever roll
  alone could be refreshed, which is nearly every frame; conflating the two
  certified 68 of 798 frames carrying a held-over pose.
* **A refused frame freezes the reference, it does not advance it.** Separating
  those two was tried and measured: density_max went 1.17 → 1.43 and 1.12 →
  1.91, and the frames gained were false accepts.
* Player positions are **out of scope** until the field pipeline is frozen.

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
scripts/
  prepare.py             stages 0-2 for one or more videos
  measure_proposers.py   the proposer agreement instrument
  valid_scale.py         THE PROMOTION GATE: measured gap vs the printed gap
  read_hud.py            decode the HUD alphabet; absolute field position
  track.py               the walk; --refresh after any detector change
  carry.py               apply the carry pass to a walk result
  referee.py             the two independent referees
  exp_direction.py       does walk direction bias the scale? (no)
work/                  per-video intermediates, rebuildable (git-ignored)
out/                   deliverables only
docs/LESSONS.md        what this project measured, and what it got wrong
```
