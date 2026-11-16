"""Scripted demo for swap-a-filling-bin-under-a-running-chute, played in the yam_arena embodiment.

    TASK_LOT=A python demo.py task.py out_a                    # full score; also lots B and C
    TASK_LOT=A python demo.py task.py out --variant slow       # one bin after the other
    TASK_LOT=B python demo.py task.py out --variant timer      # swaps on a clock set for lot A, without looking
    TASK_LOT=A python demo.py task.py out --variant idle       # does nothing

What the demo knows: the starting layout (where the bins and the FULL mark are, from the setup) and, while it runs,
the top camera's image (the frame Dream-1 is sent) and its own joint angles. It counts the parts falling out of the
tube from the image (perceive.py); it never reads the simulator's object poses or the task's counters. The bins are
where its own grippers put them.

1. The right arm takes the first bin (under the tube) by the handle tab on its right end, the left arm the empty bin
   by the tab on its left end; the left arm lifts the empty bin 1 cm and holds it beside the first, 4 cm away.
2. From the moment it holds the first bin (2.3 s, before the first part in any lot), it counts the parts falling into
   the first bin. Just after the sixth has landed, in the gap before the seventh,
   the right arm slides the full bin out to the right while the left arm slides the empty one in under the tube.
3. The left arm sets the empty bin down under the tube and lets go at once, backing out low to its own side, clear of
   the next part; the right arm sets the full bin on the FULL mark and lets go. Both go back to the ready pose.

``slow`` takes the full bin all the way to the mark and lets go before bringing the empty bin (the seventh part falls
on the table). ``timer`` does the demo's swap at the moment the sixth part has landed in lot A, without looking (in
lots B and C the bins move while a part is falling). ``idle`` does nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demokit import down, main, together, wait, wait_until  # noqa: E402

HANDLE_Z = 0.10            # tool point on a handle tab, above the table (the tab spans 7 to 12 cm)
LIFT = 0.01                # a carried bin is lifted this much
STAGE_Y = 0.175            # the empty bin waits with its centre here, beside the first (4.3 cm between them)
OUT_Y = -0.17              # the full bin first slides out to here
AWAY = (0.06, 0.08)        # m out (toward the arm's own side) and up: where a gripper comes from and goes to
SETTLE = 0.15              # s after a part is seen falling: it has landed in the bin


def handle(task, centre, side, z):
    """Tool point on the bin's tab on the ``side`` (+1: left, -1: right) end, the bin centred at ``centre``."""
    return np.array([centre[0], centre[1] + side * task.TAB_Y, z])


def take(arm, task, centre, side):
    rot = down(0.0)                      # fingers close along y, across the 12 mm tab
    g = handle(task, centre, side, task.TOP + HANDLE_Z)
    yield from arm.move(g + [0, side * AWAY[0], AWAY[1]], rot, 1.3, grip=1.0)
    yield from arm.move(g, rot, 0.6)
    yield from arm.set_grip(0.0, 0.4)


def carry(arm, task, centre, side, seconds, lift=LIFT):
    yield from arm.move(handle(task, centre, side, task.TOP + HANDLE_Z + lift), arm.rot, seconds)


def let_go(arm, side, up=AWAY[1]):
    """Open the fingers part way (the inner finger stays clear of the falling parts), lift them straight up off the
    tab (fingertips above its top), then back out to the arm's own side."""
    yield from arm.set_grip(0.7, 0.3)
    yield from arm.move(arm.pos + [0, 0, 0.045], arm.rot, 0.3)
    yield from arm.move(arm.pos + [0, side * AWAY[0], max(0.0, up - 0.045)], arm.rot, 0.4)


def script(ctx, variant):
    import task

    L, R = ctx.left, ctx.right
    spot = np.array(task.SPOT)
    stage = np.array([spot[0], STAGE_Y])
    out = np.array([spot[0], OUT_Y])
    mark = np.array(task.FULL_MARK)
    if variant == "idle":
        yield from wait(40.0)
        return

    seen = {"n": 0}
    ready = {"right": False}
    if variant != "timer":
        import perceive

        eye = perceive.DropCounter(spot[0], spot[1], task.TOP + task.BIN_H + 0.015, task.MOUTH_Z - 0.005)
        ctx.flags["counter"] = eye

        def look():
            if not ready["right"]:      # counting starts once the right arm holds the first bin still
                return
            if eye.look(ctx.now(), ctx.obs.images["top_cam"], {"left": L.q, "right": R.q}):
                seen["n"] = eye.count
                if os.environ.get("DEMO_DEBUG"):
                    print(f"seen part {eye.count} at {ctx.now():.2f}", flush=True)

        def watching(gen):
            for _ in gen:
                look()
                yield
    else:
        def watching(gen):
            yield from gen

    def prepare_left():
        yield from take(L, task, task.EMPTY_START, +1)
        yield from carry(L, task, stage, +1, 1.8)

    def prepare_right():
        yield from take(R, task, spot, -1)
        ready["right"] = True

    yield from watching(together(prepare_left(), prepare_right()))
    if variant == "timer":   # lot A's sixth part leaves the tube at 3.5 + 5 x 2.0 s
        t0, every = task.FEED["A"]
        yield from wait_until(lambda: ctx.now() >= t0 + every * (task.LOT_SIZE - 1) + 0.1 + SETTLE, timeout=60)
    else:
        yield from watching(wait_until(lambda: seen["n"] >= task.LOT_SIZE, timeout=60))
        yield from wait(SETTLE)

    if variant == "slow":
        yield from carry(R, task, out, -1, 0.8)
        yield from carry(R, task, mark, -1, 1.2)
        yield from carry(R, task, mark, -1, 0.3, lift=0.0)
        yield from let_go(R, -1)
        yield from carry(L, task, spot, +1, 1.2)
    else:
        def right():
            yield from carry(R, task, out, -1, 0.55)
            yield from carry(R, task, mark, -1, 1.2)
            yield from carry(R, task, mark, -1, 0.3, lift=0.0)
            yield from let_go(R, -1)

        def left():
            yield from wait(0.2)
            yield from carry(L, task, spot, +1, 0.7)
            yield from carry(L, task, spot, +1, 0.25, lift=0.0)
            yield from let_go(L, +1, up=0.045)         # low, away from the tube's mouth

        yield from together(right(), left())
    if variant == "slow":
        yield from carry(L, task, spot, +1, 0.3, lift=0.0)
        yield from let_go(L, +1, up=0.045)
    yield from together(L.go_home(1.5), R.go_home(1.5))
    yield from wait(4.0)


def report(task):
    S = task._STATE
    c = "".join("-" if x is None else ("a" if x == "bin_a" else "b") for x in S["caught"][:S["dropped"]])
    return {"lot": S["lot"], "dropped": S["dropped"], "caught_by": c, "first_bin_on_mark": S["a_on_mark"],
            "second_bin_under_tube": S["b_under"], "violation": S["violation"],
            "drop_times_s": [round(x, 2) for x in task.drop_times(S["lot"])][:S["dropped"]]}


if __name__ == "__main__":
    main(script, variants=("", "slow", "timer", "idle"), report=report)
