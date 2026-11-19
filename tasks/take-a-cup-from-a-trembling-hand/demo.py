"""Scripted full-score demo for take-a-cup-from-a-trembling-hand, played in the yam_arena embodiment.

    python demo.py task.py out                  # full score
    python demo.py task.py out --variant still  # as for a still cup: aims where it is seen at one instant
    python demo.py task.py out --variant low    # grips the cup round its middle, where the fingers are
    python demo.py task.py out --variant idle   # does nothing

The right gripper comes down over the cup pointing straight down, its pads open (94 mm) either side of the cup's front
and back (70 mm), centred on where the cup sways about (its mean position, not where it is at that instant), with the
pads in the free band above the person's fingers. It closes in one go, lifts the cup straight up a little and waits
for the person to let go (their fingers open once the cup's weight is taken), then lifts it clear, carries it to the
coaster, sets it down and lets go.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demokit import down, main, wait, wait_until  # noqa: E402

BAND_Z = 0.128          # tool point above the cup's base: the 68 mm finger plates reach down to 94 mm, clear of the palm (78 mm) and fingers (71 mm)
LOW_Z = 0.075           # the 'low' control: plates down to 41 mm, over the index and middle fingers
PITCH = math.radians(10.0)   # the gripper points forward and 10 deg down
GRIP_ON = 0.64          # close to 60 mm on the 70 mm cup: a firm grip (about 40 N)
LIFT = 0.030            # m commanded for the first lift: the arm's soft joints give about half of it against the person's grip
BACK = 0.11             # m back along the approach before coming in


def _frame(a, c):
    """Tool rotation with approach axis ``a`` and the pads closing along ``c``."""
    from yam_arena.demo import grip_rotation

    R0 = grip_rotation(0.0)
    ci, ai = int(np.argmax(np.abs(R0[1, :]))), int(np.argmax(np.abs(R0[2, :])))
    cs, as_ = np.sign(R0[1, ci]), np.sign(-R0[2, ai])
    a = a / np.linalg.norm(a)
    c = c - np.dot(c, a) * a
    c /= np.linalg.norm(c)
    R = np.zeros((3, 3))
    R[:, ai], R[:, ci] = as_ * a, cs * c
    k = 3 - ai - ci
    R[:, k] = np.cross(R[:, (k + 1) % 3], R[:, (k + 2) % 3])
    return R


def approach(arm, target):
    """Approach axis (forward from the arm's base toward ``target``, PITCH down) and tool rotation."""
    d = np.asarray(target[:2], float) - arm.base[:2]
    d /= np.linalg.norm(d)
    a = np.array([d[0] * math.cos(PITCH), d[1] * math.cos(PITCH), -math.sin(PITCH)])
    return a, _frame(a, np.array([-d[1], d[0], 0.0]))


def script(ctx, variant):
    import task

    S = task._STATE
    yield from wait(0.5)
    if variant == "idle":
        yield from wait(12.0)
        return
    o = S["origin"]                               # where the cup sways about (its mean position)
    arm = ctx.right                               # its approach runs along the sway, so the pads close along it
    grasp = np.array([o[0], o[1], o[2] + (LOW_Z if variant == "low" else BAND_Z)])
    a, rot = approach(arm, grasp)
    yield from arm.move(grasp - BACK * a + [0, 0, 0.03], rot, 1.8, grip=1.0)
    yield from arm.move(grasp - BACK * a, rot, 0.5)
    if variant == "still":
        # as for a cup standing still: aim where the cup is seen at one instant, close quickly and lift away
        seen = np.asarray(S["cup_pos"], float)
        grasp = np.array([seen[0], seen[1], grasp[2]])
        yield from arm.move(grasp, rot, 0.8)
        yield from arm.set_grip(GRIP_ON, 0.25)
        yield from arm.move(grasp + [0, 0, 0.15], rot, 1.0)
    else:
        yield from arm.move(grasp, rot, 1.2)
        yield from wait(0.2)
        yield from arm.set_grip(GRIP_ON, 0.4)
        yield from wait(0.2)
        # a first lift takes the cup's weight off the little finger: the person lets go
        yield from arm.move(grasp + [0, 0, LIFT], rot, 1.0)
        yield from wait_until(lambda: task._SERVO["open"], timeout=2.0)
        yield from wait(0.4)
        yield from arm.move(grasp + [0, 0, 0.15], rot, 1.2)
    # where the cup sits in the grip (it is seldom exactly on the tool point): aim the cup, not the gripper, at the
    # coaster (a camera on the cup gives the same)
    yield from wait(0.3)
    off = np.asarray(S["cup_pos"], float)[:2] - np.asarray(S["tcp"]["right"], float)[:2]
    c = task.TRAY_C - off
    place = np.array([c[0], c[1], task.TRAY_TOP + task.COASTER_T + 0.003 + BAND_Z])
    _, rot2 = approach(arm, place)
    yield from arm.move(place + [0, 0, 0.07], rot2, 1.6)
    for _ in range(2):                            # look again from just above the coaster and correct
        yield from wait(0.4)
        err = task.TRAY_C - np.asarray(S["cup_pos"], float)[:2]
        place[:2] += err
        yield from arm.move(place + [0, 0, 0.04], rot2, 0.5)
    yield from arm.move(place, rot2, 1.0)
    yield from wait(0.2)
    yield from arm.set_grip(1.0, 0.4)
    yield from arm.move(place - 0.08 * a + [0, 0, 0.05], rot2, 0.8)
    yield from arm.go_home(1.5)
    yield from wait(1.5)


def report(task):
    S = task._STATE
    return {"violation": S["violation"], "released_at": S["released_at"],
            "max_wrist_deg": round(S["max_wrist"], 1), "forearm_slide_cm": round(S["max_slide"] * 100, 2),
            "pinch_margin_mm": round(S["pinch_near"] * 1000, 1), "cup": [round(float(v), 3) for v in S["cup_pos"]],
            "on_coaster": bool(S["on_coaster"]), "sway": S.get("sway")}


if __name__ == "__main__":
    main(script, variants=("", "still", "low", "idle"), report=report)
