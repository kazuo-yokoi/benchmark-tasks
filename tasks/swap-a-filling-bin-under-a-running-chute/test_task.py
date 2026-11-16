"""Offline checks (no Isaac Sim) of the feed's timing and of the scoring rules the README states.

    python -m pytest test_task.py      # or: python test_task.py
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TASK_LOT", "A")
import task  # noqa: E402

Q0 = (0.0, 0.0, 0.0, 1.0)
A, B = "bin_a", "bin_b"


def _split(n, dropped=10):
    return [A] * n + [B] * (dropped - n)


def test_lots_feed_differently():
    """Each lot starts at a different time and runs at a different pace, so a fixed clock fits only one of them."""
    sixth = {k: task.drop_times(k)[task.LOT_SIZE - 1] for k in "ABC"}
    seventh = {k: task.drop_times(k)[task.LOT_SIZE] for k in "ABC"}
    assert len({round(v, 2) for v in sixth.values()}) == 3
    for k in "ABC":     # a swap timed for lot A (just after its sixth part) falls inside no other lot's gap
        if k != "A":
            assert not (sixth[k] < sixth["A"] + 0.25 < seventh[k] - 1.2)
    assert all(task.FEED[k][1] >= 1.8 for k in "ABC")
    assert all(task.drop_times(k)[-1] < 30 for k in "ABC")      # all parts fall well inside the 60 s run limit


def test_full_score_only_for_an_exact_lot():
    assert task.score_from(False, _split(6), 10, True, True) == 1.0
    assert task.score_from(False, _split(6, 7), 7, True, True) == 1.0
    assert task.score_from(False, _split(5), 10, True, True) == 0.5        # one too few
    assert task.score_from(False, _split(7), 10, True, True) == 0.5        # one too many
    assert task.score_from(False, _split(4), 10, True, True) == 0.0
    assert task.score_from(False, _split(8), 10, True, True) == 0.0
    assert task.score_from(False, _split(6), 10, True, False) == 0.5       # second bin still held, or not placed
    assert task.score_from(False, _split(6), 10, False, True) == 0.5


def test_waiting_and_doing_nothing_score_zero():
    assert task.score_from(False, [A] * 10, 10, False, False) == 0.0       # idle: every part in the first bin
    assert task.score_from(False, [A] * 10, 10, True, True) == 0.0         # swapped after the line ran out
    assert task.score_from(False, [A] * 6, 6, True, False) == 0.0          # swapped, nothing yet in the second


def test_a_violation_or_a_wrong_order_scores_zero():
    assert task.score_from(True, _split(6), 10, True, True) == 0.0
    assert task.score_from(False, [A] * 6 + [None] + [B] * 3, 10, True, True) == 0.0   # part 7 missed
    assert task.score_from(False, [A] * 5 + [B, A] + [B] * 3, 10, True, True) == 0.0   # bins swapped back and forth


def test_inside_a_bin():
    p0 = (0.34, 0.0, task.TOP)
    assert task.inside((0.34, 0.0, task.TOP + 0.02), p0, Q0)
    assert not task.inside((0.34 + 0.07, 0.0, task.TOP + 0.02), p0, Q0)          # past the inner wall
    assert not task.inside((0.34, 0.0, task.TOP + task.BIN_H + 0.01), p0, Q0)    # above the rim
    assert task.inside((0.34, 0.0, task.TOP + task.BIN_H + 0.01), p0, Q0, above=0.03)
    yaw45 = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    assert task.inside((0.34 + 0.075, 0.0, task.TOP + 0.02), p0, yaw45)           # corner of a turned bin


def test_a_held_bin_counts_as_placed():
    """Placing needs only the bin's position and that it stands on the table: a gripper still on its handle is fine."""
    assert task.under_tube((*task.SPOT, task.TOP), Q0)
    assert not hasattr(task, "LET_GO")


def test_placements():
    q_tilt = (math.sin(math.radians(15) / 2), 0.0, 0.0, math.cos(math.radians(15) / 2))
    assert task.on_mark((*task.FULL_MARK, task.TOP), Q0)
    assert not task.on_mark((*task.FULL_MARK, task.TOP), q_tilt)
    assert not task.on_mark((*task.FULL_MARK, task.TOP + 0.03), Q0)               # held up off the table
    assert not task.on_mark((task.FULL_MARK[0] + 0.09, task.FULL_MARK[1], task.TOP), Q0)
    assert task.under_tube((task.SPOT[0] + 0.02, task.SPOT[1], task.TOP), Q0)
    assert not task.under_tube((task.SPOT[0] + 0.04, task.SPOT[1], task.TOP), Q0)


if __name__ == "__main__":
    importlib.reload(task)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
