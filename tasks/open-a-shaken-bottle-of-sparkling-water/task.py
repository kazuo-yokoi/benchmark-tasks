"""On the station table: a 500 mL clear PET bottle of chilled sparkling water that has just been shaken (bubbles and a
layer of foam show in it), standing on a drip mat, and a small dish for the cap.

The bottle's 28 mm screw cap turns on its thread; its seal opens after 45 deg and the thread lets go after about
1.55 turns. The gas and the foam follow fizz.py: open the gas path quickly and the water gushes gas, the foam climbs
the neck and comes out over the rim; let the gas out a little at a time, closing the cap again whenever the foam
climbs, and it collapses between.

The job: open the bottle and put the cap in the dish, with no foam or water over the rim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

import fizz  # noqa: E402
import make_assets as A  # noqa: E402
from dynkit import Mover, UsdBody, arms, drive, now, on_reset, on_step  # noqa: E402

INSTRUCTION = "Open the bottle of sparkling water and put the cap in the dish."

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

# World frame: x forward from the arms, y to the robot's left, z up (m).
MAT = dict(c=(0.28, -0.12), size=(0.30, 0.20, 0.003))
BASE = np.array([0.28, -0.12, TOP + MAT["size"][2]])            # the bottle's base centre
DISH_C, DISH_R, DISH_H = np.array([0.17, -0.27]), 0.050, 0.012
CAP_JOINTS = ["cap_x", "cap_y", "cap_z", "cap_rx", "cap_ry", "twist"]

# the thread's drag (viscous, applied by the joint at every physics step): about 0.08 N m at a brisk 90 deg/s
TWIST_DAMPING = 0.05                                        # N m s/rad
LOCK_LIN, LOCK_ROT = (3000.0, 8.0), (6.0, 0.03)            # stiffness, damping while threaded

# -- scoring constants ------------------------------------------------------------------------------------------------
TIP_DEG = 30.0
CAP_CLEAR = 0.03        # m above the rim: the cap is off
LET_GO = 0.07           # m from the bottle's axis to each tool point: the bottle is released
CAP_LET_GO = 0.05       # m from the cap to each tool point: the cap is released
FULL_HOLD_S = 1.0       # full credit once the finished state has held this long

BOTTLE = UsdBody("bottle", pos=tuple(map(float, BASE + [0, 0, 0.0005])), size=(0.065, 0.065, 0.21),
                 usd="assets/bottle.usda", articulation=True, joint_pos=tuple((j, 0.0) for j in CAP_JOINTS),
                 joint_damping=0.0, friction=(1.0, 0.9), solver_iters=32)
FIXED = [
    Box("drip_mat", pos=(*MAT["c"], TOP + MAT["size"][2] / 2), size=MAT["size"], color=(0.30, 0.32, 0.36)),
    Cylinder("dish", pos=(*DISH_C, TOP + DISH_H / 2), radius=DISH_R, height=DISH_H, color=(0.93, 0.93, 0.90)),
]

# -- the foam, drawn as slices inside the bottle (seen through the clear PET) -----------------------------------------
_STATE: dict = {}
N_SLICE = 12
_SLICE_Z = np.linspace(fizz.Z_WATER, fizz.Z_RIM, N_SLICE + 1)
HIDE = (0.0, 0.0, -1.0)


def _radius_at(z: float) -> float:
    pr = fizz.PROFILE
    for (z0, r0), (z1, r1) in zip(pr[:-1], pr[1:]):
        if z0 <= z <= z1:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return pr[-1][1]


def _bottle_frame():
    p = _STATE.get("bottle_pos")
    q = _STATE.get("bottle_quat")
    if p is None:
        return BASE.copy(), (0.0, 0.0, 0.0, 1.0)
    return np.asarray(p, float), tuple(float(v) for v in q)


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slice_pose(k):
    z0, z1 = _SLICE_Z[k], _SLICE_Z[k + 1]

    def pose(t):
        top = _STATE.get("foam_top", fizz.Z_WATER)
        if top <= z0 + 0.0005:
            return HIDE, (0.0, 0.0, 0.0, 1.0)
        p, q = _bottle_frame()
        zc = (z0 + min(z1, top)) / 2
        return tuple(p + quat_matrix(q) @ np.array([0.0, 0.0, zc])), q
    return pose


FOAM = []
for _k in range(N_SLICE):
    _zc = (_SLICE_Z[_k] + _SLICE_Z[_k + 1]) / 2
    _m = Mover(f"foam{_k}", pos=HIDE, radius=max(0.006, _radius_at(_zc) - 0.0012), height=float(_SLICE_Z[1] - _SLICE_Z[0]),
               kind="cylinder", collide=False, color=(0.97, 0.97, 0.95))
    drive(_m.name, _slice_pose(_k))
    FOAM.append(_m)


def _overflow_pose(t):
    if not _STATE.get("spilled"):
        return HIDE, (0.0, 0.0, 0.0, 1.0)
    p, q = _bottle_frame()
    return tuple(p + quat_matrix(q) @ np.array([0.0, 0.0, fizz.Z_RIM - 0.012])), q


def _puddle_pose(t):
    if not _STATE.get("spilled"):
        return HIDE, (0.0, 0.0, 0.0, 1.0)
    p, _ = _bottle_frame()
    return (p[0] + 0.045, p[1] + 0.02, TOP + MAT["size"][2] + 0.001), (0.0, 0.0, 0.0, 1.0)


OVERFLOW = Mover("overflow", pos=HIDE, radius=0.021, height=0.030, kind="cylinder", collide=False,
                 color=(0.97, 0.97, 0.95))
PUDDLE = Mover("puddle", pos=HIDE, radius=0.035, height=0.002, kind="cylinder", collide=False, color=(0.70, 0.85, 0.97))
drive("overflow", _overflow_pose)
drive("puddle", _puddle_pose)

OBJECTS = [BOTTLE, *FIXED, *FOAM, OVERFLOW, PUDDLE]


def tilt_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))


# -- joint drives -----------------------------------------------------------------------------------------------------
def _set_cap_drives(environment, threaded: bool) -> None:
    import torch

    art = environment.unwrapped.scene["bottle"]
    ids = art.find_joints(CAP_JOINTS, preserve_order=True)[0]
    if threaded:
        k = [LOCK_LIN[0]] * 3 + [LOCK_ROT[0]] * 2 + [0.0]
        d = [LOCK_LIN[1]] * 3 + [LOCK_ROT[1]] * 2 + [TWIST_DAMPING]
    else:
        k = [0.0] * 6
        d = [0.02] * 3 + [0.0005] * 3
    dev = art.device
    art.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([k], dtype=torch.float32, device=dev), joint_ids=ids)
    art.write_joint_damping_to_sim_index(damping=torch.tensor([d], dtype=torch.float32, device=dev), joint_ids=ids)


# -- the monitor ------------------------------------------------------------------------------------------------------
@on_reset
def _reset(environment=None) -> None:
    _STATE.update(t=0.0, fizz=fizz.Bottle(shaken=True), theta=0.0, omega=0.0, off=False, off_at=None, spilled=False,
                  violation=None, written=False, foam_top=fizz.Z_WATER, bottle_pos=BASE.copy(),
                  bottle_quat=(0.0, 0.0, 0.0, 1.0), cap_pos=None, cap_off=False, cap_in_dish=False, upright=True,
                  released=True, on_mat=True, vented=False, vented_at=None, max_top=fizz.Z_WATER, P=fizz.P0, done_since=None)
    if environment is not None:
        _write_layout(environment)


_reset()


def _write_layout(environment) -> None:
    import torch

    base = environment.unwrapped
    origin = base.scene.env_origins
    dev = origin.device
    art = base.scene["bottle"]
    rp = art.data.root_link_pose_w[0, :3]
    bp = art.data.body_pos_w[0]
    root = art.body_names[int(torch.argmin(torch.linalg.norm(bp - rp, dim=1)))]
    rel = BASE + [0, 0, 0.0005] + (np.array([0.0, 0.0, 0.0]) if root == "bottle" else np.array([0, 0, A.CAP_Z0]))
    pose = torch.tensor([[*rel, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)
    pose[:, :3] += origin
    art.write_root_pose_to_sim_index(root_pose=pose)
    art.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
    ids = art.find_joints(CAP_JOINTS, preserve_order=True)[0]
    z = torch.zeros((1, len(ids)), dtype=torch.float32, device=dev)
    art.write_joint_position_to_sim_index(position=z, joint_ids=ids)
    art.write_joint_velocity_to_sim_index(velocity=z, joint_ids=ids)
    _set_cap_drives(environment, threaded=True)
    _ = art.data.body_pos_w[0].detach().cpu().numpy()      # read back: the next step starts from fresh buffers
    _STATE["written"] = True
    _STATE["root"] = root


def _flag(what: str) -> None:
    if _STATE["violation"] is None:
        _STATE["violation"] = (round(_STATE["t"], 2), what)


def bodies(environment) -> dict:
    base = environment.unwrapped
    org = base.scene.env_origins[0].detach().cpu().numpy()
    art = base.scene["bottle"]
    pos = art.data.body_pos_w[0].detach().cpu().numpy() - org
    quat = art.data.body_quat_w[0].detach().cpu().numpy()
    out = {n: (pos[i].astype(float), quat[i].astype(float)) for i, n in enumerate(art.body_names)}
    q = art.data.joint_pos[0].detach().cpu().numpy()
    w = art.data.joint_vel[0].detach().cpu().numpy()
    out["joints"] = {n: (float(q[i]), float(w[i])) for i, n in enumerate(art.joint_names)}
    return out


@on_step
def _monitor(environment) -> None:
    S = _STATE
    S["t"] = now()
    if not S["written"]:
        _write_layout(environment)
    b = bodies(environment)
    bp, bq = b["bottle"]
    R = quat_matrix(bq)
    S["bottle_pos"], S["bottle_quat"] = bp, bq
    theta, w = b["joints"]["twist"]
    S["theta"], S["omega"] = theta, w
    if not S["off"] and theta >= fizz.THETA_OFF:
        S["off"], S["off_at"] = True, round(S["t"], 2)
        _set_cap_drives(environment, threaded=False)
    bot = S["fizz"]
    bot.step(theta, float(environment.unwrapped.step_dt), off=S["off"])
    S["foam_top"], S["P"] = bot.top, bot.P
    S["max_top"] = max(S["max_top"], bot.top)
    if bot.vented and S["vented_at"] is None:
        S["vented_at"] = round(S["t"], 2)
    S["vented"] = bot.vented
    if bot.top > fizz.Z_RIM and (theta > fizz.THETA_SEAL or S["off"]):
        S["spilled"] = True
        _flag("foam over the rim")
    tilt = tilt_deg(R)
    S["upright"] = tilt < 10.0
    if tilt > TIP_DEG:
        _flag(f"bottle tipped ({tilt:.0f} deg)")
    in_mat = abs(bp[0] - MAT["c"][0]) <= MAT["size"][0] / 2 and abs(bp[1] - MAT["c"][1]) <= MAT["size"][1] / 2
    S["on_mat"] = in_mat and bp[2] < TOP + 0.02
    cp, _ = b["cap"]
    S["cap_pos"] = cp
    rim = bp + R @ np.array([0.0, 0.0, fizz.Z_RIM])
    S["cap_off"] = S["off"] and float(np.linalg.norm(cp - rim)) > CAP_CLEAR
    S["cap_in_dish"] = S["cap_off"] and float(np.hypot(*(cp[:2] - DISH_C))) <= DISH_R and cp[2] < TOP + DISH_H + 0.03
    a = arms(environment)
    axis_xy = bp[:2]
    S["released"] = all(float(np.hypot(*(a[s]["tcp"][:2] - axis_xy))) > LET_GO
                        or a[s]["tcp"][2] > bp[2] + A.H_BOTTLE + 0.05 for s in a)
    S["cap_released"] = all(float(np.linalg.norm(a[s]["tcp"] - cp)) > CAP_LET_GO for s in a)
    done = S["cap_in_dish"] and S["cap_released"] and S["upright"] and S["on_mat"] and S["released"]
    S["done_since"] = (S["done_since"] if S["done_since"] is not None else S["t"]) if done else None


def violation() -> bool:
    return _STATE["violation"] is not None


def score_from(S) -> float:
    if S["violation"] is not None:
        return 0.0
    if S["cap_off"]:
        if S["done_since"] is not None and S["t"] - S["done_since"] >= FULL_HOLD_S - 1e-6:
            return 1.0
        return 0.5
    return 0.25 if S["vented"] else 0.0


def score(objects) -> float:
    return score_from(_STATE)


def success(objects) -> bool:
    """Cap off, let go and in the dish, the bottle upright on its mat and let go, no foam over the rim; held 1 s."""
    return score(objects) >= 1.0


def status() -> str:
    S = _STATE
    return (f"theta {math.degrees(S['theta']):.0f} P {S['P']:.2f} foam {S['foam_top'] * 1000:.0f}mm off {S['off']} "
            f"capoff {S['cap_off']} dish {S['cap_in_dish']} vented {S['vented_at']} rel {S['released']} "
            f"viol {S['violation']}")
