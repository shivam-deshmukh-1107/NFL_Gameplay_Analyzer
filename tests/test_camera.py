"""Load-bearing tests for the camera model and the two identity proposers.

    ./nfl_venv/Scripts/python.exe tests/test_camera.py

EXIT CODES ARE LOAD-BEARING, as in both parent projects:
    0  pass
    1  a real failure
    2  an infrastructure problem (import error, missing file)
Never conflate 1 and 2.

Everything here is SYNTHETIC - a camera we construct, projecting a field we
define - so these tests need no footage and cannot be broken by a change in the
data. They exist to pin down facts about the MODEL that the parent projects
measured the hard way, two of which are degeneracies that no amount of fitting
can see.

The most important test in this file is `test_scale_degeneracy_is_exact`. If it
ever fails, the model has changed in a way nobody intended. If
`test_focal_is_not_constrained_by_yard_lines` fails, the model has GAINED a
constraint and `config.GAME_FOCAL_PX` must be re-derived rather than trusted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from nflfield import camera as CAM
    from nflfield import config, ladder as LAD, lattice as LAT
except Exception:                                          # noqa: BLE001
    traceback.print_exc()
    print("INFRASTRUCTURE: imports failed")
    raise SystemExit(2)

W, H = 1356, 1016
AY = CAM.aspect_ay(W, H)

#: A plausible Blitz-like pose: low, behind the field, looking down it.
#:
#: GETTING THIS WRONG MAKES EVERY TEST BELOW LIE, so the fixture is checked by
#: `test_camera_cannot_be_foreshortening_free`, which asserts the gap ratio is
#: CAMERA-LIKE rather than merely greater than 1. First attempt here put the
#: camera at Cx=20 with yards 0..55, i.e. half the field BEHIND the lens: the
#: visible lines then crammed against the horizon at a gap ratio of 269 (a real
#: view is ~2.8), sub-pixel spacing made both proposers return nonsense, and
#: three tests failed for a reason that had nothing to do with the code.
#:
#: Roll is deliberately non-zero: at roll 0 and azimuth 0 the projected yard
#: lines are exactly horizontal, so a dead-flat banner edge is
#: indistinguishable from real paint and the direction-gate test cannot run.
#: Yash's recovered elevation across four clips collapsed to -28.4/-28.5 deg,
#: so that is the value used.
POSE = np.array([config.GAME_FOCAL_PX, 0.0, np.radians(-28.5),
                 np.radians(4.0), -25.0, 16.0])


def visible_yards(p, lo=0.0, hi=100.0, step=5.0, margin=0.05):
    """The 5-yard lines this pose actually puts inside the frame.

    A detector only ever reports lines that are on screen, so a test fed the
    whole field is not testing what the code sees. Lines closer together than
    a pixel are also dropped: two lines the image cannot separate are not two
    observations, and keeping them tests the fitter against data no detector
    could produce.
    """
    Hm = CAM.camera_H(p, W, H, AY)
    assert Hm is not None, "fixture pose does not project"
    xc = W / 2.0
    out = []
    for y in np.arange(lo, hi + 1e-9, step):
        yy = CAM.line_y_at(Hm, float(y), np.array([xc]))
        if yy is None or not np.all(np.isfinite(yy)):
            continue
        if margin * H <= yy[0] <= (1.0 - margin) * H:
            out.append((float(y), float(yy[0])))
    out.sort(key=lambda t: t[1])
    keep = [out[0]] if out else []
    for y, yy in out[1:]:
        if yy - keep[-1][1] >= 2.0:
            keep.append((y, yy))
    return np.array([y for y, _ in keep])


def project_lines(p, yards, xs=None):
    """Image y of each yard line at the given columns, or None if degenerate."""
    Hm = CAM.camera_H(p, W, H, AY)
    if Hm is None:
        return None
    if xs is None:
        xs = np.array([0.2 * W, 0.5 * W, 0.8 * W])
    out = []
    for y in yards:
        yy = CAM.line_y_at(Hm, float(y), xs)
        if yy is None or not np.all(np.isfinite(yy)):
            return None
        out.append(yy)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# the two degeneracies
# --------------------------------------------------------------------------- #
def test_scale_degeneracy_is_exact():
    """Scaling (Cx, Cz) by k and relabelling yards by k is PIXEL-IDENTICAL.

    This is the single most important fact about this model, and the model does
    NOT protect you from it. Yash measured max |dy| = 3e-12 px at
    k = 0.5 / 2 / 4, with the solver reaching it at 0.000 px residual - and it
    happened for real on clip_04 f159-163, producing a camera 148 yd in the air.

    `MAX_JUMP_YD` is the only guard that catches it. Any change that appears to
    make this test fail has almost certainly broken the model, not fixed the
    degeneracy.
    """
    yards = visible_yards(POSE)
    base = project_lines(POSE, yards)
    assert base is not None, "base pose did not project"
    worst = 0.0
    for k in (0.5, 2.0, 4.0):
        p = POSE.copy()
        p[4] *= k          # Cx
        p[5] *= k          # Cz
        got = project_lines(p, yards * k)
        assert got is not None, "scaled pose k=%s did not project" % k
        worst = max(worst, float(np.max(np.abs(got - base))))
    assert worst < 1e-6, ("scale degeneracy is not exact: max |dy| = %.3e px "
                          "(expected ~1e-12). The model changed." % worst)
    return "max |dy| = %.2e px over k = 0.5/2/4" % worst


def test_focal_is_not_constrained_by_yard_lines():
    """Pin the focal anywhere and the other five parameters absorb it.

    A family of equally spaced parallel coplanar lines puts exactly five
    independent numbers into an image (the perpendicular vanishing point, 2;
    the 1-D projective map along a transversal, 3). The camera has six unknowns
    once the lateral position is pinned. **The one unconstrained direction is
    the focal length.**

    Yash measured this on real frames: pin the focal from 700 to 4200 px, refit
    the other five, and the residual is bit-for-bit identical while camera
    height moves 18.7 -> 7.0 yd. A frame with 11 lines is exactly as
    uninformative about focal as one with three.

    **IF THIS TEST EVER FAILS, the model has acquired a new constraint and
    `config.GAME_FOCAL_PX` must be re-derived, not trusted.**
    """
    yards = visible_yards(POSE)
    truth = project_lines(POSE, yards)
    assert truth is not None
    xs = np.array([0.2 * W, 0.5 * W, 0.8 * W])
    xc = W / 2.0
    # Build the (y_at_centre, slope) line list the fitter consumes.
    lines = []
    for row in truth:
        sl = (row[2] - row[0]) / (xs[2] - xs[0])
        y_c = row[1] + sl * (xc - xs[1])
        lines.append((float(y_c), float(sl)))

    resids, heights = [], []
    for f_pin in (700.0, 1100.0, 1511.0, 2600.0, 4200.0):
        free = CAM.FREE_ALL.copy()
        free[0] = False                       # focal pinned
        start = POSE.copy()
        start[0] = f_pin
        Hm, p, r = CAM.fit_camera(lines, yards, W, H, start, free, ay=AY)
        assert Hm is not None, "fit failed at focal %s" % f_pin
        resids.append(float(np.median(r)))
        heights.append(float(p[5]))
    spread = max(resids) - min(resids)
    h_range = max(heights) - min(heights)
    assert spread < 0.05, (
        "residual varies by %.4f px across a 6x focal range: the focal has "
        "become observable from yard lines. Re-derive GAME_FOCAL_PX." % spread)
    assert h_range > 1.0, (
        "camera height barely moved (%.3f yd) across a 6x focal range, which "
        "means the fit is not absorbing the focal as expected." % h_range)
    return ("residual spread %.2e px, height moved %.1f yd (%.1f -> %.1f)"
            % (spread, h_range, min(heights), max(heights)))


# --------------------------------------------------------------------------- #
# the model is a camera, not an affine map
# --------------------------------------------------------------------------- #
def test_camera_cannot_be_foreshortening_free():
    """A pinhole viewing a plane must foreshorten. This is why we fit a camera.

    Every anchor the parent project used before 2026-09-07 was near-affine:
    clip_04's mapped 100 yards into 600 px with a largest/smallest 5-yard gap
    ratio of 1.06, against 2.80 for the paint it described. Re-fitting a free
    8-dof homography does not prevent that, because an affine map IS a
    homography. Fitting the camera removes the failure by construction.
    """
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    ys = got[:, 1]
    gaps = np.abs(np.diff(ys))
    ratio = float(gaps.max() / max(gaps.min(), 1e-9))
    # Lower bound: a real camera cannot be affine. Upper bound: the FIXTURE
    # must be camera-like. Yash measured 2.80 on clip_04's paint; a ratio in
    # the hundreds means the pose is looking along the plane, not at it.
    assert ratio > 1.5, ("5-yard gap ratio is %.2f: this pose is nearly "
                         "affine, which a real camera cannot be." % ratio)
    assert ratio < 25.0, ("5-yard gap ratio is %.1f: the FIXTURE pose is not "
                          "camera-like (clip_04's paint measures 2.80), so "
                          "every other test here is testing nothing."
                          % ratio)
    return "largest/smallest 5-yard gap ratio %.2f over %d visible lines" % (
        ratio, len(yards))


def test_aspect_ay_is_one_for_a_4_3_picture():
    """`ay` is derived, not fitted, and must be 1 on an undistorted 4:3 render.

    This is the generalisation the merge made to Yash's single-focal camera. It
    has to be exactly inert where his measurements were taken, or every one of
    his constants silently shifts.
    """
    assert abs(CAM.aspect_ay(1356, 1017) - 1.0) < 1e-12, CAM.aspect_ay(1356, 1017)
    # The real clips are 1356x1016, one pixel short of 4:3.
    ay = CAM.aspect_ay(1356, 1016)
    assert 0.998 < ay < 1.0, ay
    fx, fy = config.focal_px(1356, 1016)
    assert abs(fx - config.GAME_FOCAL_PX) < 1e-9, fx
    assert abs(fy / fx - ay) < 1e-12, (fy / fx, ay)
    return "ay(1356,1017) = 1 exactly; ay(1356,1016) = %.5f" % ay


def test_reference_width_scaling():
    """The two reference widths must stay separate and must be inert at ref.

    Three silent bugs in this project came from conflating them (docs/LESSONS).
    """
    assert config.px_scale(config.PX_REF_W) == 1.0
    assert config.det_scale(config.SHIVAM_REF_W) == 1.0
    assert abs(config.det_scale(1356) - 1.4125) < 1e-4, config.det_scale(1356)
    # LONG_CUT was measured at 960 and must be converted, not carried flat.
    assert abs(config.LONG_CUT_PX - 155.4) < 0.1, config.LONG_CUT_PX
    g = CAM.gates_for(config.PX_REF_W)
    assert g["max_fit_resid_px"] == CAM.MAX_FIT_RESID_PX
    return ("px_scale(%d)=1, det_scale(%d)=1, LONG_CUT=%.1f px"
            % (config.PX_REF_W, config.SHIVAM_REF_W, config.LONG_CUT_PX))


# --------------------------------------------------------------------------- #
# the two proposers, on data with a known answer
# --------------------------------------------------------------------------- #
def test_both_proposers_recover_a_synthetic_ladder():
    """Both proposers must return all-ones on a complete projected 5-yd family.

    The merge's cross-check is only meaningful if the two agree on data whose
    answer is known. This also pins the step-ORDER convention that bug 3 in
    docs/LESSONS got wrong: `LatticeFit.steps` must be in POSITION order.
    """
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    s = np.sort(got[:, 1])

    lad = LAD.assign_ladder(s)
    assert lad is not None, "ladder returned None on a clean synthetic family"
    assert max(lad[0]) == 1 and min(lad[0]) == 1, (
        "ladder found steps %s on a complete 5-yard family" % (lad[0],))

    f = LAT.fit(s)
    assert f is not None, "lattice returned None on a clean synthetic family"
    f = LAT.refine(s, f)
    steps = f.steps
    assert np.all(steps == 1), ("lattice found steps %s on a complete 5-yard "
                                "family" % steps.tolist())
    assert np.array_equal(steps, np.asarray(lad[0], int)), (
        "proposers disagree on synthetic data: %s vs %s"
        % (steps.tolist(), list(lad[0])))
    return "both returned %d all-one rungs" % len(steps)


def test_proposers_agree_with_a_missing_line():
    """A hidden line must become a 2-rung step for both, at the same place.

    This is the normal case on real footage - a player stands on a yard line -
    and it is the case that exposed the step-ordering bug, because the single
    "2" is what reveals a reversed sequence.
    """
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    ys = np.sort(got[:, 1])
    drop = len(ys) // 2
    s = np.delete(ys, drop)

    lad = LAD.assign_ladder(s)
    f = LAT.fit(s)
    assert lad is not None and f is not None
    f = LAT.refine(s, f)
    lad_steps = np.asarray(lad[0], int)
    lat_steps = f.steps
    assert lad_steps.sum() == len(yards) - 1, lad_steps
    assert np.array_equal(lat_steps, lad_steps), (
        "proposers disagree with one line missing: lattice %s vs ladder %s "
        "(a REVERSAL here means the step order convention regressed)"
        % (lat_steps.tolist(), lad_steps.tolist()))
    assert (lad_steps == 2).sum() == 1, lad_steps
    return "both found %s" % lad_steps.tolist()


def test_ladder_rejects_an_impostor():
    """A line off the 5-yard family must be dropped, not absorbed.

    Modelled on clip_04 f161, where the yellow first-down marker sat 18 px from
    a real line and inflated the whole ladder, after which the solve rode the
    scale degeneracy to a camera 148 yd in the air.
    """
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    ys = np.sort(got[:, 1])
    # An impostor a short way from a real line, as the drawn marker was: on
    # clip_04 f161 it sat 18 px from a real line where the genuine 5-yard gap
    # at that depth was ~100 px, i.e. about a fifth of a rung.
    k = len(ys) // 2
    impostor = ys[k] + 0.18 * (ys[k + 1] - ys[k])
    lines = [(float(y), 0.0) for y in np.sort(np.append(ys, impostor))]
    kept, dropped = LAD.drop_impostors(lines)
    assert dropped >= 1, ("impostor rejection dropped nothing; ladder on the "
                          "contaminated set was %s"
                          % (LAD.assign_ladder([y for y, _ in lines])[0],))
    lad = LAD.assign_ladder([y for y, _ in kept])
    assert lad is not None and max(lad[0]) == 1, (
        "ladder still not clean after %d drop(s): %s" % (dropped, lad[0]))
    return "dropped %d, ladder clean at %d lines" % (dropped, len(kept))


# --------------------------------------------------------------------------- #
# the gates refuse what they are supposed to refuse
# --------------------------------------------------------------------------- #
def test_jump_gate_catches_a_doubled_scale():
    """The scale degeneracy must be caught by `yard_shift`, since nothing else can.

    Construct the degenerate twin (k=2) and confirm the jump guard sees a large
    movement even though the projected yard lines are pixel-identical. That
    combination - identical pixels, large yard shift - IS the degeneracy, and
    it is why the guard is measured in yards and not in pixels.
    """
    H1 = CAM.camera_H(POSE, W, H, AY)
    p2 = POSE.copy()
    p2[4] *= 2.0
    p2[5] *= 2.0
    H2 = CAM.camera_H(p2, W, H, AY)
    assert H1 is not None and H2 is not None
    jump = CAM.yard_shift(H1, H2, W, H)
    assert jump > CAM.MAX_JUMP_YD, (
        "doubled scale produced a jump of only %.2f yd, under the %.1f yd "
        "guard: the degeneracy would pass." % (jump, CAM.MAX_JUMP_YD))
    return "k=2 twin shifts the model %.1f yd (guard %.1f)" % (jump, CAM.MAX_JUMP_YD)


def _lines_from(got, xs, xc):
    out = []
    for row in got:
        sl = (row[2] - row[0]) / (xs[2] - xs[0])
        out.append((float(row[1] + sl * (xc - xs[1])), float(sl)))
    return out


def test_banner_edge_is_excluded_or_refused_by_line_count():
    """A flat non-paint line must never be ABSORBED into the fit.

    Modelled on clip_01 f206: its "4 lines" were three real yard lines plus the
    flat bottom edge of a result banner (y=235, slope 0.001), which crossed the
    centre column exactly where yard 25 would. The ladder named it, the fit bent
    the perspective to take it, and the median residual never saw it.

    **FIRST VERSION OF THIS TEST ASSERTED THE WRONG MECHANISM** and is recorded
    because the distinction matters. It gave the solve ELEVEN lines, one a
    banner edge, and demanded a refusal. The solve certified - correctly. With
    eleven lines the impostor is 28.5 px from its model line (a 4 deg slope
    error over the +-407 px sample span), `OUTLIER_PX` drops it, and ten clean
    lines remain. Certifying there is the right answer; refusing would throw
    away a good frame for one bad segment, which is exactly the behaviour
    outlier rejection exists to prevent (residual-based refusals were 113 of
    the 150 frames being declined before it existed).

    What actually saved f206 was the LINE COUNT: drop one of four and three
    remain, which `MIN_KEPT_AFTER_REJECTION` refuses because a 5-parameter fit
    to 3 lines is near-exact by construction and its residual is not evidence.

    So the real invariant has two halves, and both are tested:
      * many lines  -> the impostor is EXCLUDED and the frame is certified
      * few lines   -> the frame is REFUSED
    """
    xs = np.array([0.2 * W, 0.5 * W, 0.8 * W])
    xc = W / 2.0
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    Hm = CAM.camera_H(POSE, W, H, AY)

    allv = _lines_from(got, xs, xc)
    real_slope_deg = abs(np.degrees(np.arctan(allv[len(allv) // 2][1])))
    assert real_slope_deg > CAM.MAX_LINE_DIR_DEG, (
        "fixture pose's yard lines are only %.2f deg off flat, so a flat "
        "impostor is indistinguishable and this test cannot run"
        % real_slope_deg)

    # ---- many lines: exclude, then certify ---------------------------- #
    many = list(allv)
    k = len(many) // 2
    many[k] = (many[k][0], 0.001)              # flat banner edge, same y
    _, _, info = CAM.solve_frame(Hm, many, None, W, H, POSE.copy(),
                                 lock_focal=True, ay=AY)
    assert info["solve_ok"], (
        "with %d lines the frame was refused (%s); outlier rejection should "
        "have dropped the one banner edge" % (len(many), info["reason"]))
    assert info["n_kept"] < len(many), (
        "the banner edge was ABSORBED into the fit: n_kept %d of %d lines. "
        "That is the f206 failure." % (info["n_kept"], len(many)))

    # ---- four lines, one of them the banner: refuse ------------------- #
    few = [allv[i] for i in (k - 1, k, k + 1, k + 2)]
    few[1] = (few[1][0], 0.001)
    _, _, info4 = CAM.solve_frame(Hm, few, None, W, H, POSE.copy(),
                                  lock_focal=True, ay=AY)
    assert not info4["solve_ok"], (
        "4 lines including a %.1f-deg-off banner edge were CERTIFIED "
        "(resid %s px, n_kept %s). This is clip_01 f206."
        % (real_slope_deg, info4.get("resid_px"), info4.get("n_kept")))
    return ("%d lines -> certified on %d; 4 lines -> refused (%s)"
            % (len(many), info["n_kept"], (info4["reason"] or "")[:38]))


def test_locked_focal_stays_locked_through_multistart():
    """A locked parameter must hold ONE value across every multistart candidate.

    CAUGHT 2026-10-17, and the reason it needs a test rather than a comment is
    that nothing about the symptom points at the cause. `bootstrap_pose` locks
    the focal and asks for a multistart; the grid handed each candidate its own
    f0, and because the focal is NOT OBSERVABLE from yard lines (see
    `test_focal_is_not_constrained_by_yard_lines`) every candidate fitted
    equally well and the winner was arbitrary. Every certified frame of clip_01
    and clip_02 came back at focal exactly 700.0, elevation -49.4 deg against
    Yash's measured -28.5, and height 13.0 yd against his 16-19.

    The yard line POSITIONS were right throughout - residual 0.69 px, absolute
    scale check +0.15 yd - because the degeneracy preserves them. So no
    position-based metric could have caught this, and that is the whole point.
    """
    yards = visible_yards(POSE)
    got = project_lines(POSE, yards)
    assert got is not None
    xs = np.array([0.2 * W, 0.5 * W, 0.8 * W])
    xc = W / 2.0
    lines = _lines_from(got, xs, xc)

    free = CAM.FREE_ALL.copy()
    free[0] = False
    want = config.GAME_FOCAL_PX
    base = POSE.copy()
    base[0] = want
    _, p, _ = CAM.fit_camera(lines, yards, W, H, base, free,
                             multistart=True, ay=AY)
    assert p is not None, "fit failed"
    assert abs(p[0] - want) < 1e-6, (
        "locked focal drifted to %.1f from %.1f: a multistart candidate kept "
        "its own f0, so 'locked' means 'chosen arbitrarily'." % (p[0], want))

    # And the same through the real cold-start entry point.
    lad = LAD.assign_ladder(np.sort(got[:, 1]))
    assert lad is not None
    _, p2, _ = CAM.bootstrap_pose(lines, lad[1], W, H, lock_focal=True, ay=AY)
    assert p2 is not None, "bootstrap_pose failed"
    exp = want * config.px_scale(W)
    assert abs(p2[0] - exp) < 1e-6, (
        "bootstrap_pose returned focal %.1f, expected the game constant %.1f"
        % (p2[0], exp))
    el = abs(np.degrees(p2[2]))
    assert 15.0 < el < 45.0, (
        "bootstrap elevation %.1f deg is outside anything measured on this "
        "footage (Yash: -28.4 to -28.5 across four clips; -49 is what focal "
        "700 produces)" % el)
    return "focal held at %.1f, elevation %.1f deg" % (p2[0], el)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    passed = failed = 0
    for t in TESTS:
        try:
            note = t()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %-48s %s" % (t.__name__, exc))
        except Exception:                                  # noqa: BLE001
            print("ERROR %-48s" % t.__name__)
            traceback.print_exc()
            return 2
        else:
            passed += 1
            print("ok    %-48s %s" % (t.__name__, note or ""))
    print("\n%d passed, %d failed, of %d" % (passed, failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
