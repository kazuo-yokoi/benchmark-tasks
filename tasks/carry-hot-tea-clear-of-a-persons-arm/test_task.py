"""Offline checks (no Isaac Sim) of the person's movements and the distances the README states.

    python -m pytest test_task.py      # or: python test_task.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _task(lot: str, trigger_at: float | None = 0.0):
    os.environ["TASK_LOT"] = lot
    import task

    importlib.reload(task)
    task._STATE["trigger_at"] = trigger_at
    return task


def _line(a, b, n=200):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a + (b - a) * k / n for k in range(n + 1)]


def _routes(task, goal):
    import demo

    pts = [task.CUP_START, *demo.ROUND, goal]
    round_ = sum((_line(a, b) for a, b in zip(pts[:-1], pts[1:])), [])
    return _line(task.CUP_START, goal), round_


def _gap(task, pts, t):
    return min(task.cup_gap(t, p) for p in pts)


def test_person_waits_for_the_start():
    """At rest both arms keep clear of every way to either place of the coaster (README: 19 cm)."""
    for lot in "ABC":
        task = _task(lot, trigger_at=None)
        for goal in (task.COASTER_XY, task.COASTER_MOVED):
            straight, round_ = _routes(task, goal)
            for t in (0.0, 5.0, 30.0):
                assert _gap(task, straight, t) > 0.19 and _gap(task, round_, t) > 0.19
                assert np.allclose(task.coaster_xy(t), task.COASTER_XY)


def test_when_the_person_starts():
    """README: each lot starts once the lifted cup is within TRIGGER_R of where the hand goes, not on a clock."""
    for lot in "ABC":
        task = _task(lot, trigger_at=None)
        q, r = np.array(task.TRIGGER_AT[lot]), task.TRIGGER_R[lot]
        start = np.array(task.CUP_START)
        toward = start + (q - start) * (1 - (r - 0.01) / np.linalg.norm(q - start))    # 1 cm inside the circle
        assert not task._triggered(toward, task.TOP + 0.001)                # not while on the table
        assert task._triggered(toward, task.TOP + 0.05)
        far = q + (start - q) / np.linalg.norm(start - q) * (r + 0.01)               # 1 cm outside it
        assert not task._triggered(far, task.TOP + 0.05)


def test_the_arms_slide_along_their_lines():
    """Each model arm only translates along its own line, at one height, its fingertips just above the table; the
    arm a lot does not use stays at rest."""
    for lot in "ABC":
        task = _task(lot)
        for t in np.arange(0.0, 12.0, 0.05):
            r_tip, l_tip = task.tips(t)
            for tip, rest, d in ((r_tip, task.RIGHT_REST, task.RIGHT_D), (l_tip, task.LEFT_REST, task.LEFT_D)):
                off = np.asarray(tip) - np.asarray(rest)
                assert abs(off[0] * d[1] - off[1] * d[0]) < 1e-9 and off @ d > -1e-9
            j = task.person_pose(t)["joints"]
            z = task.TIP_Z + (task.RAISE_A if lot == "A" else 0.0)
            assert abs(j["tip"][2] - z) < 1e-9 and abs(j["left_tip"][2] - task.TIP_Z) < 1e-9
            if lot != "C":
                assert np.allclose(l_tip, task.LEFT_REST)
            else:
                assert np.allclose(r_tip, task.RIGHT_REST)


def test_lot_a_covers_the_coaster_then_clears():
    task = _task("A")
    straight, round_ = _routes(task, task.COASTER_XY)
    K = np.asarray(task.COASTER_XY)
    ring = [K + r * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)
            for r in np.linspace(0.0, task.CENTRED - 1e-4, 6)]
    for t in np.arange(1.2, 3.6, 0.2):                          # README: the hand lies on the coaster about 2.4 s
        # wherever on the coaster the cup could count as set down, it is over or within 2 cm of the hand
        assert max(task.cup_gap(t, p) for p in ring) < task.CLEARANCE
        assert _gap(task, straight, t) < 0.0 and _gap(task, round_, t) < 0.0
    for t in (4.8, 20.0):
        assert task.cup_gap(t, task.COASTER_XY) > 0.19 and _gap(task, straight, t) > 0.19


def test_the_hands_lie_just_clear_of_what_is_under_them():
    """The hand's underside rides 3 to 6 mm above the table, and in lot A above the coaster (sim and rig alike)."""
    from scipy.spatial.transform import Rotation

    for lot, floor in (("A", None), ("B", None)):
        task = _task(lot)
        task._STATE["trigger_at"] = 0.0
        o = task.person_pose(2.0 if lot == "A" else 0.5)
        for pre in ("person_", "person_left_"):
            pos, q = o["pieces"][pre + "hand"]
            z = (task.P.mesh_points(pre + "hand") @ Rotation.from_quat(q).as_matrix().T + np.asarray(pos))[:, 2]
            under = task.COASTER_TOP if (lot == "A" and pre == "person_") else task.TOP
            assert 0.003 < z.min() - under < 0.006


def test_lot_b_pushes_the_coaster_and_blocks_the_way_round():
    task = _task("B")
    assert np.allclose(task.coaster_xy(1.0), task.COASTER_XY)
    for t in (2.0, 10.0, 50.0):
        assert np.allclose(task.coaster_xy(t), task.COASTER_MOVED)
    for t in np.arange(task.PUSHING[0], task.PUSHING[1], 0.1):    # the fingertips at the coaster while pushing it
        assert task.cup_gap(t, task.coaster_xy(t)) < task.CLEARANCE
    straight, round_ = _routes(task, task.COASTER_MOVED)
    for t in np.arange(2.7, 30.0, 0.5):
        assert _gap(task, straight, t) > 0.19                    # README: the straight way keeps 20 cm
        assert _gap(task, round_, t) < 0.0                       # README: the way round passes over the arm
        assert task.cup_gap(t, task.COASTER_MOVED) > 0.19


def test_lot_c_blocks_the_straight_way():
    task = _task("C")
    straight, round_ = _routes(task, task.COASTER_XY)
    for t in np.arange(1.3, 30.0, 0.5):
        assert _gap(task, straight, t) < 0.0                     # README: the straight way passes over the hand
        assert _gap(task, round_, t) > 0.085                     # README: the way round keeps 9 cm
        assert task.cup_gap(t, task.COASTER_XY) > 0.12


def _clear_of(pts, centre, size):
    """How far the cup's edge stays from a box lying on the table, seen from above, along ``pts``."""
    return min(np.hypot(max(abs(p[0] - centre[0]) - size[0] / 2, 0.0), max(abs(p[1] - centre[1]) - size[1] / 2, 0.0))
               for p in pts)


def test_the_tea_never_has_to_pass_over_things():
    """The right way in each lot keeps the cup clear of the pen, and the pushed coaster slides clear of it."""
    task = _task("A")
    pen = (2 * task.PEN_R, task.PEN_L)
    straight, round_ = _routes(task, task.COASTER_XY)
    assert _clear_of(straight, task.PEN_XY, pen) > task.CUP_R + 0.01                 # lot A
    assert _clear_of(round_, task.PEN_XY, pen) > task.CUP_R + 0.04                   # lot C
    straight, _ = _routes(task, task.COASTER_MOVED)
    assert _clear_of(straight, task.PEN_XY, pen) > task.CUP_R + 0.15                 # lot B
    assert _clear_of(_line(task.COASTER_XY, task.COASTER_MOVED), task.PEN_XY, pen) > task.COASTER_R + 0.07


def test_scoring_tiers():
    task = _task("A")
    task._reset()
    S = task._STATE
    for stage, want in ((0.0, 0.0), (0.25, 0.25), (0.5, 0.5)):
        S["stage"] = stage
        assert task.score({}) == want
    S["done_at"] = 12.0
    assert task.score({}) == 1.0 and task.success({})
    S.update(done_at=None, violation="cup over or against the person's arm", violation_at=5.0, stage=0.5)
    assert task.score({}) == 0.0 and not task.success({})


def test_reach_of_the_routes():
    import demo
    from yam_arena import ik, station

    task = _task("B")
    rot = ik.matrix_to_quat_xyzw(demo.tilted())
    for goal in (task.COASTER_XY, task.COASTER_MOVED):
        straight, round_ = _routes(task, goal)
        pts = [(*p, task.TOP + demo.CARRY_Z_ABOVE) for p in straight[::10] + round_[::10]]
        pts += [(*goal, task.COASTER_TOP + task.CUP_H / 2 + 0.004), (*task.CUP_START, task.TOP + task.CUP_H / 2)]
        for p in pts:
            _, err, _ = ik.solve(p, rot, base_pos=station.BASES["left"], iters=150)
            assert err < 0.002, p


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
