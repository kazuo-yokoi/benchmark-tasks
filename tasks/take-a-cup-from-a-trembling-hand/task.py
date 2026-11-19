"""On the station table, which stands for a bed's overbed table: a person's right forearm rests on a foam armrest at the
far side, and their hand, which has a tremor, holds an empty cup out toward the robot. A meal tray with a cork coaster
is on the robot's right.

The person is a life-size trainer forearm and hand. Two hobby servos in a pan-tilt bracket at the wrist shake the hand
at 4 to 5 Hz (the cup's rim sways about +-8 mm side to side and +-5 mm back and forth), for the whole trial. The
fingers wrap the cup's lower 7 cm on its front (the robot's side) and right, the thumb its back, the little finger
curls under its base. A third servo opens the fingers once the cup has been lifted 3 mm off the little finger for
0.25 s: the person lets go once the cup's weight is taken.

The job: take the cup and put it on the coaster, without knocking it out, catching a finger between the pads, or
pushing or pulling the person's hand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

import make_assets as A  # noqa: E402
from dynkit import UsdBody, arms, now, on_reset, on_step, robot_links  # noqa: E402

INSTRUCTION = "Take the cup the person is holding out to you and put it on the coaster on the tray."

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

# World frame: x forward from the arms, y to the robot's left, z up (m). Arm bases at x 0, y +-0.305.
CUP_Z = 0.100                                   # the cup's base above the table, as the hand holds it
ARMREST = dict(x=(0.455, 0.755), y=0.049, w=0.16, h=A.FOREARM_BOTTOM + CUP_Z)
TRAY_C, TRAY = np.array([0.26, -0.22]), (0.35, 0.26, 0.015)
COASTER_R, COASTER_T = 0.045, 0.004
TRAY_TOP = TOP + TRAY[2]

CUP_XY = np.array([0.37, 0.00])                # the cup's base centre as the hand holds it (hand-frame origin)
TREMOR_HZ = 4.5

# the tremor: the pan servo (flex, vertical axis) and the tilt servo (roll, along the forearm), a quarter cycle apart
FLEX_AMP = 0.008 / 0.084                        # rad: +-8 mm at the cup (84 mm from the pan axis)
ROLL_AMP = 0.005 / 0.075                        # rad: +-5 mm at the cup's grip band
SERVO_K, SERVO_D, SERVO_TAU = 30.0, 0.10, 0.25  # N m/rad, N m s/rad, N m: a micro servo (MG90S class) at the wrist
REL_K, REL_D, REL_TAU = 5.0, 0.20, 0.20
GRIP_DEG = 3.0                                  # the release servo holds the fingers this far past touching: a light grip
LIFT_TRIGGER, TRIGGER_S = 0.003, 0.25           # the person lets go once the cup is this far up for this long

# -- scoring constants ------------------------------------------------------------------------------------------------
TIP_DEG = 45.0          # knocked over
UPRIGHT_DEG = 10.0
WRIST_DEG = 25.0        # the wrist pointer past this: the hand was pushed or pulled (the tremor is +-5.5 deg)
FOREARM_SLIDE, FOREARM_LIFT, FOREARM_TURN = 0.015, 0.010, math.radians(8.0)
COASTER_TOL = 0.01      # m: the whole 7 cm base on the 9 cm coaster
FULL_HOLD_S = 1.0       # full credit once the cup has stood on the coaster, let go, this long
CLEAR = 0.05            # m between the cup and every finger: out of the hand
LET_GO = 0.07           # m from the cup's axis to each tool point: let go
PINCH_OPEN = 0.085      # m: a gripper closed this far is closing on whatever is between its pads
PAD_BACK, PAD_FWD, PAD_HALF_W = 0.060, 0.013, 0.034   # the finger plates: 68 mm wide, to 13 mm past the tool point
SETTLE_S = 0.3


def hand_pose():
    """The hand frame's origin (the cup's base centre, world) and yaw: the forearm straight along x."""
    return np.array([CUP_XY[0], CUP_XY[1], TOP + CUP_Z]), 0.0


def yaw_q(a: float):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def to_world(p_hand, origin, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(p_hand, float)
    return origin + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


_O, _Y = hand_pose()
HAND = UsdBody("hand", pos=tuple(map(float, _O)), size=(0.4, 0.12, 0.12), usd="assets/hand.usda", articulation=True,
               joint_pos=(("flex", 0.0), ("roll", 0.0), ("release", 0.0)), joint_damping=0.05, friction=(0.9, 0.8))
CUP = UsdBody("cup", pos=(float(_O[0]), float(_O[1]), float(_O[2]) + 0.0004), size=(0.07, 0.07, 0.11),
              usd="assets/cup.usda", mass=A.CUP_MASS, friction=(0.8, 0.7), torsional_patch=0.01)
FIXED = [
    Box("armrest", pos=(sum(ARMREST["x"]) / 2, ARMREST["y"], TOP + ARMREST["h"] / 2),
        size=(ARMREST["x"][1] - ARMREST["x"][0], ARMREST["w"], ARMREST["h"]), color=(0.36, 0.52, 0.56)),
    Box("tray", pos=(*TRAY_C, TOP + TRAY[2] / 2), size=TRAY, color=(0.80, 0.78, 0.72)),
    Cylinder("coaster", pos=(*TRAY_C, TRAY_TOP + COASTER_T / 2), radius=COASTER_R, height=COASTER_T,
             color=(0.68, 0.50, 0.33)),
]
OBJECTS = [HAND, CUP, *FIXED]

# finger and thumb centre lines in their bodies' frames (for the pinch and out-of-hand checks)
_FINGERS, _THUMB = A.hand_parts()
_K_AT = np.array([A.KNUCKLE[0], A.KNUCKLE[1], 0.035])
_H_AT = np.array([0.0, A.PALM_IN_Y + A.PALM_T / 2, 0.035])
LINK_ORIGIN = {"forearm": np.array([0.20, A.WRIST[1], 0.04]), "wrist": A.WRIST, "hand": _H_AT, "fingers": _K_AT}


def _samples(path, r, step=0.006):
    out = []
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(np.linalg.norm(b - a) / step))
        out += [(a + u * (b - a), r) for u in np.linspace(0, 1, k, endpoint=False)]
    out.append((path[-1], r))
    return out


FINGER_PTS = [(p - _K_AT, r) for _, path, r in _FINGERS for p, r in _samples(path, r)]
THUMB_PTS = [(p - _H_AT, r) for p, r in _samples(_THUMB, A.THUMB[0])]


def tilt_deg(q) -> float:
    x, y, z, w = (float(v) for v in q)
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    R = quat_matrix(q)
    return math.atan2(R[1, 0], R[0, 0])


# -- the servos ---------------------------------------------------------------------------------------------------------
_SERVO = {"n": 0, "dt": 1 / 240, "open": False, "env": None}


def servo_targets(t: float):
    w = 2 * math.pi * TREMOR_HZ
    return (FLEX_AMP * math.sin(w * t), ROLL_AMP * math.sin(w * t + math.pi / 2),
            math.radians(A.OPEN_DEG) if _SERVO["open"] else math.radians(GRIP_DEG))


def _install_servos(environment) -> None:
    """Gains per joint, and the servo targets written every physics step (240 Hz), not only every control step."""
    import torch

    base = environment.unwrapped
    hand = base.scene["hand"]
    ids = hand.find_joints(["flex", "roll", "release"], preserve_order=True)[0]
    dev = hand.device
    t = lambda v: torch.tensor([v], dtype=torch.float32, device=dev)  # noqa: E731
    hand.write_joint_stiffness_to_sim_index(stiffness=t([SERVO_K, SERVO_K, REL_K]), joint_ids=ids)
    hand.write_joint_damping_to_sim_index(damping=t([SERVO_D, SERVO_D, REL_D]), joint_ids=ids)
    hand.write_joint_effort_limit_to_sim_index(limits=t([SERVO_TAU, SERVO_TAU, REL_TAU]), joint_ids=ids)
    _SERVO["dt"] = float(base.physics_dt)
    if getattr(hand, "_tremor", False):
        return
    original = hand.write_data_to_sim

    def write_data_to_sim():
        _SERVO["n"] += 1
        tgt = servo_targets(_SERVO["n"] * _SERVO["dt"])
        hand.actuators.target_command.set_position_index(value=t(list(tgt)), joint_ids=ids)
        original()

    hand.write_data_to_sim = write_data_to_sim
    hand._tremor = True


# -- the monitor ------------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset(environment=None) -> None:
    o, y = hand_pose()
    _STATE.update(t=0.0, origin=o, yaw=y, violation=None, written=False, lifted_since=None,
                  released_at=None, cup_pos=o.copy(), cup_tilt=0.0, wrist=(0.0, 0.0), forearm0=None,
                  max_wrist=0.0, max_slide=0.0, out_of_hand=False, held=False, on_tray=False, on_coaster=False,
                  let_go=False, upright=True, pinch_near=1.0, full_since=None)
    _SERVO.update(n=0, open=False)
    if environment is not None:
        _install_servos(environment)
        _write_layout(environment)


_reset()


def _write_layout(environment) -> None:
    import torch

    base = environment.unwrapped
    origin = base.scene.env_origins
    dev = origin.device
    o, y = _STATE["origin"], _STATE["yaw"]
    hand = base.scene["hand"]
    # PhysX picks the articulation's root link itself (here the wrist bracket): pose whichever it is
    rp = hand.data.root_link_pose_w[0, :3]
    bp = hand.data.body_pos_w[0]
    root = hand.body_names[int(torch.argmin(torch.linalg.norm(bp - rp, dim=1)))]
    fa = to_world(LINK_ORIGIN[root], o, y)
    pose = torch.tensor([[*fa, *yaw_q(y)]], dtype=torch.float32, device=dev)
    pose[:, :3] += origin
    hand.write_root_pose_to_sim_index(root_pose=pose)
    hand.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
    ids = hand.find_joints(["flex", "roll", "release"], preserve_order=True)[0]
    z = torch.zeros((1, 3), dtype=torch.float32, device=dev)
    hand.write_joint_position_to_sim_index(position=z, joint_ids=ids)
    hand.write_joint_velocity_to_sim_index(velocity=z, joint_ids=ids)
    cup = base.scene["cup"]
    pose = torch.tensor([[o[0], o[1], o[2] + 0.0004, *yaw_q(y)]], dtype=torch.float32, device=dev)
    pose[:, :3] += origin
    cup.write_root_pose_to_sim_index(root_pose=pose)
    cup.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
    bodies(environment)          # read the state back: without this the hand's next step starts from stale buffers
    _STATE["written"] = True


def _flag(what: str) -> None:
    if _STATE["violation"] is None:
        _STATE["violation"] = (round(_STATE["t"], 2), what)


def bodies(environment) -> dict:
    """World pose (pos, R) of each hand link and the cup (station frame)."""
    base = environment.unwrapped
    org = base.scene.env_origins[0].detach().cpu().numpy()
    hand = base.scene["hand"]
    pos = hand.data.body_pos_w[0].detach().cpu().numpy() - org
    quat = hand.data.body_quat_w[0].detach().cpu().numpy()
    out = {n: (pos[i].astype(float), quat_matrix(quat[i])) for i, n in enumerate(hand.body_names)}
    cup = base.scene["cup"]
    cp = cup.data.root_link_pose_w[0].detach().cpu().numpy()
    out["cup"] = (cp[:3] - org, quat_matrix(cp[3:7]), cp[3:7])
    q = hand.data.joint_pos[0].detach().cpu().numpy()
    out["joints"] = {n: float(q[i]) for i, n in enumerate(hand.joint_names)}
    return out


def hand_points(b) -> list:
    """Finger and thumb centre-line points (world) with their radii."""
    pf, Rf = b["fingers"][:2]
    ph, Rh = b["hand"][:2]
    return [(pf + Rf @ p, r) for p, r in FINGER_PTS] + [(ph + Rh @ p, r) for p, r in THUMB_PTS]


def cup_surface_dist(cup_pos, cup_R, p) -> float:
    """Distance from point p to the cup's (solid) cylinder."""
    d = cup_R.T @ (np.asarray(p) - cup_pos)
    dr = math.hypot(d[0], d[1]) - A.CUP_R
    dz = -d[2] if d[2] < 0 else (d[2] - A.CUP_H if d[2] > A.CUP_H else 0.0)
    if dr <= 0 and dz == 0:
        return 0.0
    return math.hypot(max(dr, 0.0), dz)


def pinch_check(a: dict, links: dict, pts: list) -> float:
    """The smallest margin (m) of any finger point to the region between a closing gripper's pads; <= 0 is a
    finger caught between them."""
    worst = 1.0
    for side in ("left", "right"):
        arm = a[side]
        if arm["opening"] > PINCH_OPEN:
            continue
        c_ax = arm["rot"][:, 0]                   # the fingers slide along the gripper's x axis (URDF joint7/8)
        ap = arm["rot"][:, 2]
        w_ax = np.cross(ap, c_ax)
        half = arm["opening"] / 2
        for p, r in pts:
            d = p - arm["tcp"]
            along, c, w = float(d @ ap), float(d @ c_ax), float(d @ w_ax)
            m = max(-PAD_BACK - along, along - PAD_FWD, abs(w) - PAD_HALF_W - r, abs(c) - half + 0.002)
            worst = min(worst, m)
    return worst


@on_step
def _monitor(environment) -> None:
    S = _STATE
    S["t"] = now()
    if not S["written"]:
        _install_servos(environment)
        _write_layout(environment)
    b = bodies(environment)
    cp, cR, cq = b["cup"]
    S["cup_pos"] = cp
    S["cup_tilt"] = tilt = tilt_deg(cq)
    j = b["joints"]
    S["wrist"] = (math.degrees(j["flex"]), math.degrees(j["roll"]))
    fa_p, fa_R = b["forearm"]
    if S["forearm0"] is None:
        S["forearm0"] = (fa_p.copy(), math.atan2(fa_R[1, 0], fa_R[0, 0]))
    # the person lets go once the cup has been lifted off the little finger for TRIGGER_S
    pf, Rf = b["fingers"][:2]
    # the lowest point of the cup's base above its seat (hand frame z 0): the whole base must come clear, so a cup
    # tipped or dragged in the fingers still rests on the little finger
    rim = [cp + cR @ np.array([A.CUP_R * math.cos(k * math.pi / 4), A.CUP_R * math.sin(k * math.pi / 4), 0.0])
           for k in range(8)]
    seat = min((Rf.T @ (p - pf))[2] for p in rim) + 0.035
    S["seat"] = round(float(seat) * 1000, 1)
    S["fa_lift"] = round(float(fa_p[2] - S["forearm0"][0][2]) * 1000, 1)
    if not _SERVO["open"]:
        if seat > LIFT_TRIGGER:
            S["lifted_since"] = S["t"] if S["lifted_since"] is None else S["lifted_since"]
            if S["t"] - S["lifted_since"] >= TRIGGER_S - 1e-6:
                _SERVO["open"] = True
                S["released_at"] = round(S["t"], 2)
        else:
            S["lifted_since"] = None
    a = arms(environment)
    links = robot_links(environment)
    S["tcp_rel"] = np.round(cR.T @ (a["right"]["tcp"] - cp), 3).tolist()
    S["opening"] = round(a["right"]["opening"], 4)
    S["tcp"] = {s_: a[s_]["tcp"].copy() for s_ in a}
    pts = hand_points(b)
    S["out_of_hand"] = min(cup_surface_dist(cp, cR, p) - r for p, r in pts) > CLEAR
    centre = cp + cR[:, 2] * (A.CUP_H * 0.6)
    S["held"] = any(np.linalg.norm(a[s]["tcp"] - centre) < 0.05 and a[s]["opening"] < A.CUP_R * 2 + 0.006
                    for s in a)
    S["let_go"] = all(np.linalg.norm((a[s]["tcp"] - cp)[:2]) > LET_GO or a[s]["tcp"][2] - cp[2] > A.CUP_H + 0.06
                      for s in a)
    S["upright"] = tilt <= UPRIGHT_DEG
    in_tray = abs(cp[0] - TRAY_C[0]) <= TRAY[0] / 2 and abs(cp[1] - TRAY_C[1]) <= TRAY[1] / 2
    S["on_tray"] = in_tray and abs(cp[2] - TRAY_TOP) < 0.008
    S["on_coaster"] = S["on_tray"] and float(np.hypot(*(cp[:2] - TRAY_C))) <= COASTER_TOL
    full = S["on_coaster"] and S["upright"] and S["let_go"]
    S["full_since"] = (S["full_since"] if S["full_since"] is not None else S["t"]) if full else None
    if S["t"] < SETTLE_S:
        return
    if 0.5 < S["t"] < 3.0:                        # the rim's sway (hand frame, mm), before anyone touches the cup
        rim = cp + cR[:, 2] * A.CUP_H - S["origin"] - [0.0, 0.0, A.CUP_H]
        c, s_ = math.cos(-S["yaw"]), math.sin(-S["yaw"])
        u = (c * rim[0] - s_ * rim[1], s_ * rim[0] + c * rim[1], rim[2])
        sw = S.setdefault("sway", [1e3, -1e3, 1e3, -1e3, 1e3, -1e3])
        for k in range(3):
            sw[2 * k], sw[2 * k + 1] = round(min(sw[2 * k], u[k] * 1000), 1), round(max(sw[2 * k + 1], u[k] * 1000), 1)
    if tilt > TIP_DEG:
        _flag(f"cup knocked over ({tilt:.0f} deg)")
    if cp[2] < TOP + 0.03 and not (in_tray and cp[2] > TRAY_TOP - 0.005):
        _flag("cup dropped")
    if cp[2] < TRAY_TOP + 0.03 and not in_tray and not S["held"]:
        _flag("cup dropped")
    wmax = max(abs(S["wrist"][0]), abs(S["wrist"][1]))
    S["max_wrist"] = max(S["max_wrist"], wmax)
    if wmax > WRIST_DEG:
        _flag(f"hand pushed or pulled (wrist {wmax:.0f} deg)")
    p0, y0 = S["forearm0"]
    slide = float(np.hypot(*(fa_p - p0)[:2]))
    S["max_slide"] = max(S["max_slide"], slide)
    if slide > FOREARM_SLIDE:
        _flag(f"forearm dragged {slide * 100:.1f} cm")
    if fa_p[2] - p0[2] > FOREARM_LIFT:
        _flag("forearm lifted")
    turn = abs(math.atan2(math.sin(math.atan2(fa_R[1, 0], fa_R[0, 0]) - y0),
                          math.cos(math.atan2(fa_R[1, 0], fa_R[0, 0]) - y0)))
    if turn > FOREARM_TURN:
        _flag(f"forearm turned {math.degrees(turn):.0f} deg")
    m = pinch_check(a, links, pts)
    S["pinch_near"] = min(S["pinch_near"], m)
    if m <= 0.0:
        _flag("finger caught between the pads")


def violation() -> bool:
    return _STATE["violation"] is not None


def score_from(S) -> float:
    if S["violation"] is not None:
        return 0.0
    if S["on_tray"] and S["upright"] and S["let_go"]:
        held = S["full_since"] is not None and S["t"] - S["full_since"] >= FULL_HOLD_S - 1e-6
        return 1.0 if (S["on_coaster"] and held) else 0.5
    if S["out_of_hand"] and S["held"]:
        return 0.25
    return 0.0


def score(objects) -> float:
    return score_from(_STATE)


def success(objects) -> bool:
    """The cup upright on the coaster, let go, nothing knocked, pinched, pushed or pulled; held 1 s."""
    return score(objects) >= 1.0


def status() -> str:
    S = _STATE
    return (f"cup {np.round(S['cup_pos'], 3)} tilt {S['cup_tilt']:.1f} wrist "
            f"{S['wrist'][0]:.1f},{S['wrist'][1]:.1f} rel {S['released_at']} out {S['out_of_hand']} held "
            f"{S['held']} tcp {S.get('tcp_rel')} open {S.get('opening')} tray {S['on_tray']} coaster {S['on_coaster']} viol {S['violation']}")
