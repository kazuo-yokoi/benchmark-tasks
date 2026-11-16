"""Scripted demo for carry-hot-tea-clear-of-a-persons-arm, played in the yam_arena embodiment.

    TASK_LOT=A python demo.py task.py out_a                      # full score, lot A (waits, then carries on)
    TASK_LOT=B python demo.py task.py out_b                      # full score, lot B (waits, then straight to where
                                                                 #   the coaster now is)
    TASK_LOT=C python demo.py task.py out_c                      # full score, lot C (goes round the hand)
    TASK_LOT=C python demo.py task.py out --variant straight     # never looks at the person: carries straight on
    TASK_LOT=B python demo.py task.py out --variant near         # never looks at the person: always goes round
    TASK_LOT=C python demo.py task.py out --variant stop         # waits whenever blocked, never goes round
    TASK_LOT=B python demo.py task.py out --variant round        # waits whenever blocked, always goes round

The pick uses the fixed layout (where the cup is). While carrying, the left arm keeps a map of the table seen from
above (1 cm cells) of where the person's arms are, plus where a moving arm is heading (its tip's motion over the last
0.5 s, carried 0.6 s ahead), and where the coaster is. Before every step it checks its remaining route against that
map: the cup's edge must keep the task's 2 cm and a margin from the person. It takes a way that is clear all along
(the one it is on, else the straight way, else the way round through ROUND, out to the robot's left); when none is,
it carries on along a way whose next second of travel is clear, and otherwise holds the cup still where it is
(backing away along the way it came only if the person's arm is within 3 cm of it on the map), carrying on once a way
has stayed clear, and the person's arm still, for 0.4 s.
It sets the cup down where it last saw the coaster. Every variant, the habits too, sees the coaster.

The gripper holds the cup tilted 60 degrees forward from pointing down (the fingers still close across the cup, along
y): the arm then reaches the far half of the table, and from the top camera it hides only the table between its base
and the cup, never what lies ahead of the cup.

Where the map comes from is ``DEMO_SENSE``:
- ``camera`` (the demo): the person's arms and the coaster are found in the top camera's image, as Dream-1 sees it
  (perceive.py);
- ``truth`` (a check only, not a demo): the person's arms and the coaster as the simulator places them.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demokit import down, main, wait  # noqa: E402

SPEED = 0.12            # m/s, carrying
CARRY_Z_ABOVE = 0.10    # tool point above the table while carrying (the cup's base 5.75 cm up)
ROUND = [(0.24, 0.36), (0.54, 0.36)]  # the way round: out to the left, along, then in to the coaster
MARGIN = 0.01           # m, kept on top of the task's 2 cm (the map is widened by 3 cm and reads arms nearer)
BODY_X = 0.84           # the arm's tip is looked for short of this (the table's far edge)
RESUME_S = 0.4
HORIZON_S = 1.0        # s of travel ahead that must be clear to carry on (stopping is immediate)
STANDOFF = 0.08        # m, kept on top while the way is not clear all along: it waits clear of where they reach
TILT_GRIP = math.pi / 3  # the gripper tilted forward from pointing down
PREDICT_S, MOTION_S = 0.6, 0.5

# the map: x -0.10 .. 1.10 (past the table's far edge, where the person sits), y -0.60 .. 0.90, 1 cm cells
RES, X0, Y0, NX, NY = 0.01, -0.10, -0.60, 120, 150


def cell(xy):
    return int(round((xy[0] - X0) / RES)), int(round((xy[1] - Y0) / RES))


class PersonMap:
    """Where the person's arms are (boolean grid), where a moving arm is heading, and the distance field."""

    def __init__(self):
        self.hist = []   # (t, mask, tip): tip is the arm's cell farthest out from the person, over the table
        self.dist = None
        self.pred = None
        self.moving = False   # the arm's tip moved faster than 5 cm/s over the last 0.5 s

    def update(self, t, mask):
        from scipy.ndimage import distance_transform_edt, shift

        ib = int(round((BODY_X - X0) / RES))
        ii, jj = np.nonzero(mask[:ib, :])
        tip = None if len(ii) == 0 else np.array([X0 + RES * ii.min(), Y0 + RES * jj[np.argmin(ii)]])
        self.hist.append((t, mask, tip))
        while self.hist and self.hist[0][0] < t - MOTION_S - 1e-6:
            self.hist.pop(0)
        pred = mask.copy()
        self.moving = False
        t0, _, tip0 = self.hist[0]
        if tip is not None and tip0 is not None and t - t0 > 0.2:
            v = (tip - tip0) / (t - t0)            # the tip's velocity over the table
            self.moving = bool(np.linalg.norm(v) > 0.05)
            if self.moving:                        # moving: also cover where it is heading
                n = int(round(np.linalg.norm(v) * PREDICT_S / RES))
                step = v / np.linalg.norm(v)
                for k in range(1, n + 1):
                    pred |= shift(mask, tuple(np.round(step * k).astype(int)), order=0, cval=False)
        self.pred = pred
        self.dist = distance_transform_edt(~pred) * RES if pred.any() else np.full(mask.shape, 9.0)
        self.dist_now = distance_transform_edt(~mask) * RES if mask.any() else np.full(mask.shape, 9.0)

    def gap(self, xy, now=False) -> float:
        """The cup's edge to the person, seen from above, for a cup centred at ``xy``: to where they are or are heading,
        or (``now``) to where they are."""
        import task

        i, j = cell(xy)
        if not (0 <= i < NX and 0 <= j < NY):
            return 9.0
        return float((self.dist_now if now else self.dist)[i, j]) - task.CUP_R


def truth_mask(t):
    """A check only: the person's arms as the simulator places them, drawn into the map."""
    import task
    from person import outline_dist

    mask = np.zeros((NX, NY), bool)
    for tip, polys in zip(task.tips(t), task.OUTLINES):
        for poly in polys:
            lo, hi = poly.min(0) + tip, poly.max(0) + tip
            i0, j0 = (max(int(np.floor((v - o) / RES)), 0) for v, o in zip(lo, (X0, Y0)))
            i1, j1 = min(int(np.ceil((hi[0] - X0) / RES)) + 1, NX), min(int(np.ceil((hi[1] - Y0) / RES)) + 1, NY)
            if i0 >= i1 or j0 >= j1:
                continue
            gx, gy = np.meshgrid(X0 + RES * np.arange(i0, i1), Y0 + RES * np.arange(j0, j1), indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel()], 1)
            mask[i0:i1, j0:j1] |= (outline_dist(poly + tip, pts) <= 0.0).reshape(gx.shape)
    return mask


def _dump(folder, t, eye, mask, truth):
    """Debug picture: the frame with the person's pixels (red) and the robot's outline (blue), and the map: found
    (red), the simulator's truth (green)."""
    import imageio.v3 as iio

    img, m, robot = eye.last
    ov = img.copy()
    ov[robot] = (0.6 * ov[robot] + [0, 0, 100]).astype(np.uint8)
    ov[m] = (0.5 * ov[m] + [127, 0, 0]).astype(np.uint8)
    mp = np.full((NX, NY, 3), 230, np.uint8)
    mp[truth] = (60, 190, 60)
    mp[mask] = (220, 40, 40)
    mp[mask & truth] = (150, 110, 30)
    mp = np.flip(np.flip(mp, 0), 1)              # far side up, robot's left on the left, as the camera sees it
    mp = np.repeat(np.repeat(mp, 1, 0), 1, 1)
    canvas = np.full((max(ov.shape[0], NX), ov.shape[1] + NY + 4, 3), 255, np.uint8)
    canvas[:ov.shape[0], :ov.shape[1]] = ov
    canvas[:NX, ov.shape[1] + 4:] = mp
    Path(folder).mkdir(parents=True, exist_ok=True)
    iio.imwrite(f"{folder}/{t:06.2f}.png", canvas)


def _polyline(p, wps, step=0.01):
    pts, cur = [], np.asarray(p, float)
    for w in wps:
        w = np.asarray(w, float)
        n = max(1, int(np.linalg.norm(w - cur) / step))
        pts += [cur + (w - cur) * k / n for k in range(1, n + 1)]
        cur = w
    return pts


def _rest_of(poly, p):
    """What is left of the polyline ``poly`` from the point on it nearest ``p``: back onto it there first, then along
    it (so a cup that set off another way rejoins the way round instead of cutting its corner)."""
    p = np.asarray(p, float)
    best, k, near = 9.0, 0, p
    for i, (a, b) in enumerate(zip(poly[:-1], poly[1:])):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        u = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        d = float(np.linalg.norm(a + u * ab - p))
        if d < best - 1e-9:
            best, k, near = d, i, a + u * ab
    rest = ([near] if best > 0.005 else []) + [np.asarray(w, float) for w in poly[k + 1:]]
    return [w for w in rest if np.linalg.norm(w - p) > 0.005] or [np.asarray(poly[-1], float)]


def routes(p, goal):
    import task

    goal = np.asarray(goal, float)
    # both ways are fixed lines from the cup's start (the straight way and the way round), rejoined where the cup is:
    # a cup partway round never cuts across to the coaster over what lies on the table (the pen)
    return {"straight": _rest_of([task.CUP_START, goal], p), "near": _rest_of([task.CUP_START, *ROUND, goal], p)}


def decide(pm, p, goal, variant, plan, clear_since, now):
    """Which way to carry on from ``p`` ("straight", "near" for the way round, or None to hold), and since when a way
    has been clear. A stretch of a way is clear when the cup's edge keeps the task's 2 cm and MARGIN from the map
    along it. A way clear all along is taken (the one being taken kept, else the straight way first). When no way is
    clear all along, the cup carries on along a way whose next HORIZON_S of travel is clear, and holds only when that
    is blocked too: it stops before the person's arm, not as soon as the arm appears somewhere ahead."""
    import task

    if variant in ("straight", "near"):                  # the habits that never look at the person
        return variant, clear_since
    thr, reach = task.CLEARANCE + MARGIN, SPEED * HORIZON_S
    full, soon = {}, {}
    for k, r in routes(p, goal).items():
        gaps = [pm.gap(q) for q in _polyline(p, r)]       # 1 cm apart along the way
        full[k] = all(g >= thr for g in gaps)
        soon[k] = all(g >= thr + STANDOFF for g in gaps[:int(reach / 0.01) + 1])
    for ok in (full, soon):
        if variant == "stop":                              # the habit: wait whenever the way ahead is blocked,
            ok["near"] = False                             # never go round
        if variant == "round":                             # the habit: wait whenever blocked, always go round
            ok["straight"] = False
    if plan is not None and full[plan]:
        choice = plan
    elif full["straight"] or full["near"]:
        choice = "straight" if full["straight"] else "near"
    elif plan is not None and soon[plan]:                  # no way clear all along: go on while the next stretch is
        choice = plan
    else:                                                  # (a detour is begun only when it is clear all along)
        choice = "straight" if soon["straight"] else None
    if choice is None:
        return None, None
    if plan is None:                                       # holding: carry on once a way has stayed clear, and
        if pm.moving:                                      # the person's arm still, for a while
            return None, None
        clear_since = clear_since if clear_since is not None else now
        if now - clear_since < RESUME_S:
            return None, clear_since
    return choice, clear_since


def hold(pm, p, trail):
    """Where to hold the cup this step. It stays exactly where it is (a still cup spills nothing) unless the person's
    arm is within the task's 2 cm and MARGIN of it on the map (where it is, not where it may be heading); then it backs
    away smoothly along the way it came, at the carrying speed, as far as it must."""
    import task

    if pm.gap(p, now=True) >= task.CLEARANCE + MARGIN or not trail:
        return p
    step = SPEED * 0.05
    while trail and np.linalg.norm(trail[-1] - p) < 1e-6:
        trail.pop()
    if not trail:
        return p
    d = trail[-1] - p
    n = float(np.linalg.norm(d))
    if n <= step:
        return trail.pop()
    return p + d / n * step


def tilted() -> np.ndarray:
    """The gripper's rotation: pointing down, then tilted forward (about y) by TILT_GRIP."""
    c, s = math.cos(-TILT_GRIP), math.sin(-TILT_GRIP)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ down(0.0)


def choice_log(log, choice, ctx, pm, p, plan="?") -> bool:
    last = log[-1][1] if log else None
    if choice == last:
        return False
    log.append((round(ctx.now(), 2), choice, round(pm.gap(p), 3) if pm.dist is not None else None))
    return True


def script(ctx, variant):
    import task
    from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: F401

    sense = os.environ.get("DEMO_SENSE", "camera")
    L = ctx.left
    rot = tilted()
    top = task.TOP
    cx, cy = task.CUP_START
    grip_z = top + task.CUP_H / 2
    carry_z = top + CARRY_Z_ABOVE
    pm = PersonMap()
    if sense == "camera":
        import perceive

        eye = perceive.TopCamera()

    every = 4 if sense == "camera" else 1          # the camera is looked at every 0.2 s, as Dream-1's frames come
    debug = os.environ.get("DEMO_DEBUG")
    seen = {"n": 0}
    where = {"coaster": np.asarray(task.COASTER_XY, float)}    # the layout's coaster, until it is seen elsewhere
    p_now = {"p": np.array([cx, cy])}

    def look():
        t = ctx.now()
        seen["n"] += 1
        if (seen["n"] - 1) % every:
            return
        if sense == "truth":
            mask = truth_mask(t)
            where["coaster"] = np.asarray(task.coaster_xy(t), float)
        else:
            mask = eye.person_mask(ctx.obs, {"left": ctx.left.q, "right": ctx.right.q})
            if eye.coaster is not None:
                where["coaster"] = np.asarray(eye.coaster, float)
            if debug:                                   # for checking the demo only: never used to act
                _dump(debug, t, eye, mask, truth_mask(t))
        pm.update(t, mask)
        if debug and os.environ.get("DEMO_TRACE"):            # for checking the demo only
            g = {k: round(min(pm.gap(q) for q in _polyline(p_now["p"], r)), 3)
                 for k, r in routes(p_now["p"], where["coaster"]).items()}
            print("trace", round(t, 2), g, "pred", int(pm.pred.sum()), "mask", int(mask.sum()), "coaster",
                  np.round(where["coaster"], 3), flush=True)

    # 1. pick the cup up (fixed layout): from behind it and above, along the tilted gripper's axis
    back = -0.07 * rot[:, 2]
    yield from L.move([cx + back[0], cy, carry_z + 0.03], rot, 1.8, grip=1.0)
    yield from L.move([cx + back[0], cy, grip_z + back[2]], rot, 0.8)
    yield from L.move([cx, cy, grip_z], rot, 0.8)
    yield from L.set_grip(0.0, 0.8)
    yield from L.move([cx, cy, carry_z], rot, 0.8)
    # 2. carry it, watching the person
    p = np.array([cx, cy])
    clear_since, plan, log = ctx.now() - RESUME_S, None, ctx.flags.setdefault("log", [])
    trail = []                                     # where the cup has been, to back away along
    while True:
        look()
        goal = where["coaster"]
        if np.linalg.norm(p - goal) <= 0.004:
            break
        p_now["p"] = p
        choice, clear_since = decide(pm, p, goal, variant, plan, clear_since, ctx.now())
        choice_log(log, choice, ctx, pm, p, plan)
        plan = choice
        if choice is None:
            p = hold(pm, p, trail)
            L.aim([p[0], p[1], carry_z], rot)
            yield
            continue
        nxt = _polyline(p, routes(p, goal)[choice], step=SPEED * 0.05)[0]
        trail.append(p)
        p = np.asarray(nxt, float)
        L.aim([p[0], p[1], carry_z], rot)
        yield
    # 3. set it down on the coaster and let go
    yield from L.move([goal[0], goal[1], task.COASTER_TOP + task.CUP_H / 2 + 0.004], rot, 1.2)
    yield from L.set_grip(1.0, 0.6)
    yield from L.move([goal[0] + back[0], goal[1], task.COASTER_TOP + task.CUP_H / 2 + 0.004 + back[2]], rot, 0.8)
    yield from L.move([goal[0] + back[0], goal[1], task.COASTER_TOP + task.CUP_H + 0.04], rot, 0.8)
    yield from wait(1.5)
    print("plan changes:", log, flush=True)


def report(task):
    S = task._STATE
    return {"lot": S["lot"], "trigger_at_s": S["trigger_at"], "violation": S["violation"],
            "violation_at_s": S["violation_at"], "min_gap_m": round(S["min_gap"], 3),
            "min_touch_m": round(S["min_touch"], 3), "done_at_s": S["done_at"], "min_pen_m": round(S.get("min_pen", 9.0), 3),
            "off_centre_m": round(float(np.hypot(*(np.asarray(S.get("pos", np.zeros(3)))[:2]
                                                  - task.coaster_xy(S.get("t", 0.0))))), 4)}


if __name__ == "__main__":
    main(script, variants=("", "straight", "near", "stop", "round"), report=report)
