"""Scripted full-score demo for open-a-shaken-bottle-of-sparkling-water, played in the yam_arena embodiment.

    python demo.py task.py out                   # full score
    python demo.py task.py out --variant twist   # unscrews the cap at an ordinary pace, as for an unshaken bottle
    python demo.py task.py out --variant crack   # cracks it to the demo's 70 deg and holds it there, not watching
    python demo.py task.py out --variant idle    # does nothing

The left gripper holds the bottle round its body. The right gripper comes straight down onto the cap and turns it in
strokes of up to 100 deg, letting go and turning back between strokes. It first turns the cap until the seal opens
and the gas hisses out (70 deg); whenever the foam climbs past the shoulder it turns the cap back to 20 deg (the seal
closes and the foam collapses), and opens it again once the foam has dropped. When the gas has stopped coming, it
unscrews the cap the rest of the way, lifts it off and puts it in the dish. The foam's height is read from the task
state (a camera sees the same white column through the clear bottle).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demokit import down, main, wait, wait_until  # noqa: E402

OPEN_DEG, CLOSE_DEG = 70.0, 20.0
FOAM_HIGH, FOAM_LOW = 0.165, 0.157       # m above the base: close as the foam reaches the shoulder, reopen once it drops
STROKE = 100.0                           # deg the wrist turns per grip
TURN_DPS = 90.0                          # deg/s while venting
FAST_DPS = 200.0                         # deg/s once the gas has stopped
RETURN_DPS = 300.0                       # deg/s turning the open gripper back for the next stroke
GRIP_CAP = 0.18                          # firm on the 30 mm cap
HOLD_Z = 0.10                            # m: where the left gripper holds the bottle


def _frame(a, c):
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


def side_grip(arm, target):
    d = np.asarray(target[:2], float) - arm.base[:2]
    d /= np.linalg.norm(d)
    a = np.array([d[0], d[1], 0.0])
    return a, _frame(a, np.array([-d[1], d[0], 0.0]))


class CapHand:
    """The right arm on the cap: turns it with the wrist, re-gripping between strokes."""

    def __init__(self, arm, task):
        self.arm, self.task = arm, task
        self.yaw = -STROKE / 2
        self.gripping = False
        c = task.BASE
        self.tcp_z = c[2] + 0.0005 + task.A.CAP_Z0 + 0.016
        self.xy = c[:2]

    def pose(self, yaw_deg, up=0.0):
        return np.array([self.xy[0], self.xy[1], self.tcp_z + up]), down(math.radians(yaw_deg))

    def approach(self):
        yield from self.arm.move(*self.pose(self.yaw, up=0.04), 1.6, grip=1.0)
        yield from self.arm.move(*self.pose(self.yaw), 0.8)
        yield from self.grip()

    def grip(self):
        yield from self.arm.set_grip(GRIP_CAP, 0.25)
        self.gripping = True

    def release(self):
        yield from self.arm.set_grip(0.55, 0.2)
        self.gripping = False

    def turn_to(self, theta_deg, dps=TURN_DPS):
        """Turn the cap to ``theta_deg`` (from closed, unscrewing positive), re-gripping as needed."""
        S = self.task._STATE
        while True:
            now_deg = math.degrees(S["theta"])
            need = theta_deg - now_deg
            if abs(need) < 3.0:
                return
            lo, hi = -STROKE / 2, STROKE / 2
            room = (hi - self.yaw) if need > 0 else (self.yaw - lo)
            if room < 5.0:
                yield from self.release()
                self.yaw = lo if need > 0 else hi
                n = max(1, round(STROKE / RETURN_DPS * 20))
                for _ in range(n):
                    yield
                    self.arm.aim(*self.pose(self.yaw))
                yield from self.grip()
                continue
            step = math.copysign(min(abs(need), room), need)
            target = self.yaw + step
            n = max(1, round(abs(step) / dps * 20))
            y0 = self.yaw
            for k in range(1, n + 1):
                self.yaw = y0 + step * k / n
                self.arm.aim(*self.pose(self.yaw))
                yield
            yield from wait(0.1)
            self.yaw = target


def script(ctx, variant):
    import task

    S = task._STATE
    yield from wait(0.5)
    if variant == "idle":
        yield from wait(15.0)
        return
    L, R = ctx.left, ctx.right
    hold = task.BASE + [0, 0, HOLD_Z]
    a, rot = side_grip(L, hold)
    cap = CapHand(R, task)

    def hold_bottle():
        yield from L.move(hold - 0.10 * a + [0, 0, 0.03], rot, 1.6, grip=1.0)
        yield from L.move(hold - 0.10 * a, rot, 0.5)
        yield from L.move(hold, rot, 1.0)
        yield from L.set_grip(0.55, 0.4)

    from demokit import together
    yield from together(hold_bottle(), cap.approach())

    if variant == "twist":
        yield from cap.turn_to(math.degrees(task.fizz.THETA_OFF) + 20.0)
    elif variant == "crack":
        # opens it as far as the demo does, but holds it there without watching the foam
        yield from cap.turn_to(OPEN_DEG)
        yield from wait_until(lambda: S["vented"] or S["violation"] is not None, timeout=60.0)
        yield from cap.turn_to(math.degrees(task.fizz.THETA_OFF) + 20.0, dps=FAST_DPS)
    else:
        while not S["vented"] and S["violation"] is None:
            yield from cap.turn_to(OPEN_DEG)
            yield from wait_until(lambda: S["foam_top"] > FOAM_HIGH or S["vented"], timeout=30.0)
            if S["vented"]:
                break
            yield from cap.turn_to(CLOSE_DEG, dps=180.0)
            yield from wait_until(lambda: S["foam_top"] < FOAM_LOW, timeout=20.0)
        yield from cap.turn_to(math.degrees(task.fizz.THETA_OFF) + 20.0, dps=FAST_DPS)
    yield from wait_until(lambda: S["off"], timeout=3.0)
    # lift the cap off and put it in the dish
    p, r = cap.pose(cap.yaw, up=0.04)
    yield from R.move(p, r, 0.8)
    dish = np.array([task.DISH_C[0], task.DISH_C[1], task.TOP + task.DISH_H + 0.016 + 0.004])
    yield from R.move(dish + [0, 0, 0.06], r, 1.4)
    yield from R.move(dish, r, 0.6)
    yield from R.set_grip(1.0, 0.3)
    yield from R.move(dish + [0, 0, 0.08], r, 0.6)
    yield from L.set_grip(1.0, 0.3)
    yield from together(L.move(hold - 0.12 * a + [0, 0, 0.03], rot, 0.8), R.go_home(1.4))
    yield from L.go_home(1.4)
    yield from wait(1.5)


def report(task):
    S = task._STATE
    return {"violation": S["violation"], "vented_at": S["vented_at"], "off_at": S["off_at"],
            "max_foam_mm": round(S["max_top"] * 1000, 1), "cap_in_dish": bool(S["cap_in_dish"]),
            "P": round(S["P"], 3)}


if __name__ == "__main__":
    main(script, variants=("", "twist", "crack", "idle"), report=report)
