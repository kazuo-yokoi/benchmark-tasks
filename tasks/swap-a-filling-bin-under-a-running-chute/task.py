"""On the station table: a machine at the far end of the table discharges finished parts (steel bushings) down a
yellow chute that ends in a short drop tube over the middle of the table, 19 cm above it. One part falls out of the
tube every 1.8 to 2.4 seconds, into a blue bin standing under it. A second, empty blue bin stands on the left, in front
of the left arm, and a grey FULL mark is taped on the table on the right, in front of the right arm. Each bin has a
handle tab on both of its ends (its left and right walls).

A bin takes a lot of exactly six parts. Once the bin under the tube holds six, it goes onto the FULL mark and the empty
bin goes under the tube, before the seventh part falls: the machine does not stop. The lot (trial k of a run uses lot
k mod 3) sets when the first part falls and how often parts fall.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

from dynkit import Mover, Shape, UsdBody, declare_lots, lot, now, on_reset, on_step, show_when  # noqa: E402

INSTRUCTION = ("The machine drops a finished part down its chute into the bin under it every few seconds, and it "
               "doesn't stop. A bin takes a lot of exactly 6 parts: once the bin under the chute holds 6, put it on "
               "the grey FULL mark and put the empty bin under the chute. Every part must land in a bin.")

SCENE = "station"
TOP = TABLE_TOP[SCENE]

declare_lots("A", "B", "C")
# lot: (time the first part leaves the tube (s), time between parts (s))
FEED = {"A": (3.5, 2.0), "B": (4.6, 2.4), "C": (3.0, 1.8)}
N_PARTS = 10
LOT_SIZE = 6

# World frame: x forward from the arms, y to the robot's left, z up (m). Arm bases at x 0, y +-0.305.
SPOT = (0.34, 0.0)                 # under the drop tube: where parts fall and the first bin's centre starts
MOUTH_Z = TOP + 0.19               # the drop tube's mouth
TUBE_R, TUBE_LEN = 0.022, 0.14
EXIT_V = 0.5                       # m/s, a part's speed (straight down) as it leaves the mouth
PART_R, PART_H, PART_M = 0.0125, 0.02, 0.055   # steel bushing, 25 mm across, 20 mm long

# Bins (assets/bin.usda): 13.2 x 13.2 cm outside, 7 cm tall, 12 x 12 cm inside, 6 mm walls and floor, 150 g, with a
# handle tab (3 cm wide, 12 mm thick) rising 5 cm above the middle of each end wall (+y and -y).
BIN_IN, BIN_W, BIN_H = (0.12, 0.12), 0.006, 0.07
BIN_OUT = (BIN_IN[0] + 2 * BIN_W, BIN_IN[1] + 2 * BIN_W, BIN_H)
TAB = (0.03, 0.012, 0.05)
TAB_Y = BIN_IN[1] / 2 + BIN_W / 2  # tab centre line, from the bin's centre
EMPTY_START = (0.22, 0.24)
FULL_MARK = (0.22, -0.24)
MARK = (0.17, 0.17)                # the FULL mark (taped square, inside size)

UNDER_TOL = 0.03                   # m: the second bin's centre from SPOT
TILT_MAX = 10.0                    # degrees
BASE_TOL = 0.01                    # m: a bin's underside above the table
CATCH_S = 0.3                      # s inside a bin in a row: caught by it
MISS_S = 1.5                       # s after leaving the tube: a part not caught by then has missed
LEAVE_S = 0.25                     # s outside its bin in a row: the part has come out of it
OFF_TABLE = 0.05                   # m: a bin's underside this far below the table top has fallen off it
SUCCESS_HOLD_S = 1.0

ZINC, BLUE, YELLOW, DARK, MACHINE = (0.72, 0.73, 0.75), (0.15, 0.40, 0.80), (0.92, 0.72, 0.10), (0.30, 0.31, 0.33), \
    (0.42, 0.50, 0.46)

# -- the machine, its chute and the drop tube (only seen: nothing in the task goes near them) -------------------------
_TUBE_TOP = MOUTH_Z + TUBE_LEN
_OUTLET = (0.60, 0.0, TOP + 0.43)  # where the chute leaves the machine
_d = np.array([_OUTLET[0] - SPOT[0], 0.0, _OUTLET[2] - _TUBE_TOP])
_chute_len = float(np.linalg.norm(_d))
_pitch = math.atan2(_d[2], _d[0])                      # chute rises toward the machine
_q_chute = (0.0, math.sin(-_pitch / 2), 0.0, math.cos(-_pitch / 2))
_mid = ((SPOT[0] + _OUTLET[0]) / 2, 0.0, (_TUBE_TOP + _OUTLET[2]) / 2)

MACHINE_PARTS = [
    Box("machine", pos=(0.74, 0.0, TOP + 0.26), size=(0.28, 0.36, 0.52), color=MACHINE),
    Shape("chute_floor", pos=_mid, size=(_chute_len + 0.02, 0.05, 0.006), rot_xyzw=_q_chute, color=YELLOW,
          collide=False),
    Shape("chute_wall_l", pos=(_mid[0], 0.028, _mid[2] + 0.012), size=(_chute_len, 0.006, 0.03), rot_xyzw=_q_chute,
          color=YELLOW, collide=False),
    Shape("chute_wall_r", pos=(_mid[0], -0.028, _mid[2] + 0.012), size=(_chute_len, 0.006, 0.03), rot_xyzw=_q_chute,
          color=YELLOW, collide=False),
    Mover("drop_tube", kind="cylinder", pos=(*SPOT, MOUTH_Z + TUBE_LEN / 2), radius=TUBE_R, height=TUBE_LEN,
          color=YELLOW, collide=False),
]
RUN_LAMP = Mover("run_lamp", kind="sphere", pos=(0.598, 0.12, TOP + 0.47), radius=0.012, color=(0.1, 0.9, 0.2),
                 collide=False, emissive=40.0)
show_when(RUN_LAMP, lambda t: True)

BIN_A = UsdBody("bin_a", pos=(*SPOT, TOP), size=BIN_OUT, usd="assets/bin.usda", mass=0.15)
BIN_B = UsdBody("bin_b", pos=(*EMPTY_START, TOP), size=BIN_OUT, usd="assets/bin.usda", mass=0.15)
BINS = ("bin_a", "bin_b")

MARK_TAPE = [
    Box(f"mark_{k}", pos=(FULL_MARK[0] + dx, FULL_MARK[1] + dy, TOP + 0.0005), size=size, color=DARK)
    for k, (dx, dy, size) in enumerate((
        (0.0, MARK[1] / 2 + 0.004, (MARK[0] + 0.016, 0.008, 0.001)),
        (0.0, -MARK[1] / 2 - 0.004, (MARK[0] + 0.016, 0.008, 0.001)),
        (MARK[0] / 2 + 0.004, 0.0, (0.008, MARK[1], 0.001)),
        (-MARK[0] / 2 - 0.004, 0.0, (0.008, MARK[1], 0.001))))
]

# The parts wait out of sight on the floor under the table; each is put at the tube's mouth when its turn comes.
PARTS = [Cylinder(f"part{k}", pos=(0.30 + 0.06 * (k % 5), -0.2 + 0.1 * (k // 5), 0.03), radius=PART_R,
                  height=PART_H, mass=PART_M, color=ZINC) for k in range(N_PARTS)]
PART_NAMES = [p.name for p in PARTS]

OBJECTS = [*MACHINE_PARTS, RUN_LAMP, BIN_A, BIN_B, *MARK_TAPE, *PARTS]


def drop_times(name: str | None = None) -> list[float]:
    """When each part leaves the tube's mouth in lot ``name`` (default: this trial's lot)."""
    t0, every = FEED[name or lot()]
    return [t0 + every * k for k in range(N_PARTS)]


# -- geometry of the bins (pure functions, tested in test_task.py) ---------------------------------------------------

def yaw_of(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def to_bin(p, bin_pos, bin_q) -> np.ndarray:
    """Point p in the bin's frame (origin at the centre of its underside, yaw only)."""
    a = -yaw_of(bin_q)
    d = np.asarray(p, float) - np.asarray(bin_pos, float)
    return np.array([math.cos(a) * d[0] - math.sin(a) * d[1], math.sin(a) * d[0] + math.cos(a) * d[1], d[2]])


def inside(p, bin_pos, bin_q, slack: float = 0.0, above: float = 0.0) -> bool:
    """Part centre p inside the bin: within its inner outline (+ ``slack``), between its floor and its rim
    (+ ``above``)."""
    lx, ly, lz = to_bin(p, bin_pos, bin_q)
    return abs(lx) <= BIN_IN[0] / 2 + slack and abs(ly) <= BIN_IN[1] / 2 + slack and \
        BIN_W - 0.005 <= lz <= BIN_H + above


def standing(bin_pos, bin_q) -> bool:
    return tilt_of(bin_q) < TILT_MAX and float(bin_pos[2]) - TOP <= BASE_TOL


def on_mark(bin_pos, bin_q) -> bool:
    return standing(bin_pos, bin_q) and abs(float(bin_pos[0]) - FULL_MARK[0]) <= MARK[0] / 2 and \
        abs(float(bin_pos[1]) - FULL_MARK[1]) <= MARK[1] / 2


def under_tube(bin_pos, bin_q) -> bool:
    off = math.hypot(float(bin_pos[0]) - SPOT[0], float(bin_pos[1]) - SPOT[1])
    return standing(bin_pos, bin_q) and off <= UNDER_TOL


def lot_split(caught: list, dropped: int) -> int | None:
    """How many parts the first bin caught, when the parts went as the job wants: parts 1..n into the first bin, every
    later part that has left the tube into the second, and at least one there. None otherwise."""
    seq = caught[:dropped]
    if any(c is None for c in seq):
        return None
    n = 0
    while n < len(seq) and seq[n] == "bin_a":
        n += 1
    if n == len(seq) or any(c != "bin_b" for c in seq[n:]):
        return None
    return n


def score_from(violation: bool, caught: list, dropped: int, a_on_mark: bool, b_under: bool) -> float:
    """0 after a violation. Otherwise, with the parts split first bin / second bin as the job wants (lot_split): 0.5
    for the first bin on the FULL mark and 0.5 for the second under the tube, each standing on the table, when the
    first bin caught exactly 6; 0.25 each when it caught 5 or 7. 0 for any other split."""
    if violation:
        return 0.0
    n = lot_split(caught, dropped)
    if n is None:
        return 0.0
    each = 0.5 if n == LOT_SIZE else 0.25 if abs(n - LOT_SIZE) == 1 else 0.0
    return each * (int(a_on_mark) + int(b_under))


# -- the feed and the monitor -----------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset(environment=None) -> None:
    _STATE.update(t=0.0, lot=lot(), dropped=0, caught=[None] * N_PARTS, inside_run=[[0, 0] for _ in range(N_PARTS)],
                  out_run=[0] * N_PARTS, violation=None, a_on_mark=False, b_under=False,
                  first_b_at=None)


_reset()


def exit_pose(k: int):
    """Part k at the tube's mouth: lying on its side (a part tumbles as it leaves a tube), turned and spinning a
    little differently each time, falling at EXIT_V. Returns position, quaternion xyzw, velocity, spin."""
    yaw = (k * 2.399963) % math.pi                       # golden-angle steps: every part turned differently
    tilt = math.radians(70.0 + 15.0 * ((k * 0.618034) % 1.0))
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qx = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    off = 0.004 * math.cos(yaw), 0.004 * math.sin(yaw)
    pos = (SPOT[0] + off[0], SPOT[1] + off[1], MOUTH_Z - PART_R - 0.003)
    spin = (3.0 * math.cos(yaw), 3.0 * math.sin(yaw), 0.0)
    return pos, q, (0.0, 0.0, -EXIT_V), spin


def _put(environment, name, pos, quat, vel, spin) -> None:
    import torch

    base = environment.unwrapped
    origin = base.scene.env_origins
    pose = torch.tensor([[*pos, *quat]], dtype=torch.float32, device=origin.device)
    pose[:, :3] += origin
    v = torch.tensor([[*vel, *spin]], dtype=torch.float32, device=origin.device)
    base.scene[name].write_root_pose_to_sim_index(root_pose=pose)
    base.scene[name].write_root_velocity_to_sim_index(root_velocity=v)


def _flag(what: str) -> None:
    if _STATE["violation"] is None:
        _STATE["violation"] = (round(_STATE["t"], 2), what)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    t = now()
    S = _STATE
    S["t"] = t
    times = drop_times(S["lot"])
    while S["dropped"] < N_PARTS and t >= times[S["dropped"]]:
        _put(environment, PART_NAMES[S["dropped"]], *exit_pose(S["dropped"]))
        S["dropped"] += 1
    st = object_states(environment, [*BINS, *PART_NAMES[:S["dropped"]]])
    dt = float(environment.unwrapped.step_dt)
    bins = {b: (np.asarray(st[b].pos), np.asarray(st[b].quat_xyzw)) for b in BINS}
    for k in range(S["dropped"]):
        p = np.asarray(st[PART_NAMES[k]].pos)
        c = S["caught"][k]
        if c is None:
            for j, b in enumerate(BINS):
                S["inside_run"][k][j] = S["inside_run"][k][j] + 1 if inside(p, *bins[b]) else 0
                if S["inside_run"][k][j] * dt >= CATCH_S - 1e-9:
                    S["caught"][k] = b
                    if b == "bin_b" and S["first_b_at"] is None:
                        S["first_b_at"] = round(t, 2)
            if S["caught"][k] is None and t - times[k] > MISS_S:
                _flag(f"part {k + 1} missed the bins")
        else:
            out = not inside(p, *bins[c], slack=0.01, above=0.03)
            S["out_run"][k] = S["out_run"][k] + 1 if out else 0
            if S["out_run"][k] * dt >= LEAVE_S - 1e-9:
                _flag(f"part {k + 1} came out of its bin")
    for b in BINS:
        if float(bins[b][0][2]) < TOP - OFF_TABLE:
            _flag(f"{b} fell off the table")
    S["poses"] = {b: (np.round(bins[b][0] - np.array([0, 0, TOP]), 3).tolist(), round(tilt_of(bins[b][1]), 1))
                  for b in BINS}
    S["a_on_mark"] = bool(on_mark(*bins["bin_a"]))
    S["b_under"] = bool(under_tube(*bins["bin_b"]))


def violation() -> bool:
    return _STATE["violation"] is not None


def score(objects) -> float:
    """0 once a part has missed the bins or come out of the bin that caught it, or a bin has fallen off the table.
    Otherwise, if parts 1..n went into the first bin and every later part into the second: 0.5 for the first bin on
    the FULL mark and 0.5 for the second under the tube (each standing on the table, held or not) when n is 6, 0.25
    each when n is 5 or 7; else 0."""
    S = _STATE
    return score_from(violation(), S["caught"], S["dropped"], S["a_on_mark"], S["b_under"])


def success(objects) -> bool:
    """Score 1.0: parts 1-6 in the first bin, on the FULL mark; part 7 on (and every later one) in the second, under
    the tube; both standing on the table. Held 1 s."""
    return score(objects) >= 1.0


def status() -> str:
    S = _STATE
    c = "".join("-" if x is None else ("a" if x == "bin_a" else "b") for x in S["caught"][:S["dropped"]])
    return (f"lot {S['lot']} dropped {S['dropped']} caught {c} a_on_mark {S['a_on_mark']} b_under {S['b_under']} "
            f"violation {S['violation']} bins {S.get('poses')}")
