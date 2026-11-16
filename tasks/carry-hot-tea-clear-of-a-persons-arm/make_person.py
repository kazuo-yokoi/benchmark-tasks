"""Writes the model arms' meshes (person_hand.usda, person_forearm.usda and the left arm's two) from MakeHuman's base
mesh.

    python make_person.py [path/to/base.obj]     # needs numpy and pxr (usd-core, or Isaac Sim's Python)

The base mesh is MakeHuman's ``makehuman/data/3dobjs/base.obj`` (CC0 1.0, released by the MakeHuman project in
September 2020: https://github.com/makehumancommunity/makehuman); without a path the script downloads it. Only its
``body`` group is used (no helpers, eyes, teeth or hair), in its default A-pose, in decimetres. Each arm is cut at the
elbow into two rigid pieces that task.py moves (kinematic bodies, no collision: task.py checks contact itself):

- ``person_hand``: the right hand; origin at the wrist, x along the middle finger, y to the hand's left, z out of
  the back of the hand (up when the palm is down), its underside flattened so that it lies flat (``flat_palm``);
- ``person_forearm``: the right forearm; origin at the wrist, x from the elbow toward the wrist, z toward the back of
  the hand;
- ``person_left_hand``, ``person_left_forearm``: the same for the left arm (cut from its own side of the mesh,
  mirrored within each piece so that task.py poses them by the right arm's rules).

The forearms are skin coloured (the T-shirt's sleeve ends above the elbow). The script also cuts the rest of the body
(``pieces``) but writes only the model arms.
"""

from __future__ import annotations

import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj"
SCALE = 0.1                                  # decimetres to metres
SKIN, SHIRT, TROUSERS, HAIR = (0.80, 0.62, 0.50), (0.30, 0.42, 0.36), (0.18, 0.20, 0.26), (0.12, 0.09, 0.07)
SLEEVE_END = 0.45                            # the shirt's sleeve ends this far from the shoulder toward the elbow
PALM_Z, PALM_SQUASH = -0.008, 0.15           # m, the hand's underside (its frame); what hangs below keeps this much


def flat_palm(pts):
    """The hand lying flat, palm down: in the A-pose the thumb and the heel of the hand hang up to 4 cm below the
    fingers' pads; squash whatever lies below the pads' level (z PALM_Z in the hand's frame) up toward it."""
    pts = pts.copy()
    low = pts[:, 2] < PALM_Z
    pts[low, 2] = PALM_Z + PALM_SQUASH * (pts[low, 2] - PALM_Z)
    return pts


def load(path):
    V, G, cur = [], {}, None
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            cur = line.split()[1]
            G.setdefault(cur, [])
        elif line.startswith("f "):
            G[cur].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    V = np.array(V)
    joint = {g[6:]: V[np.unique(np.concatenate([np.array(f) for f in fs]))].mean(0)
             for g, fs in G.items() if g.startswith("joint-") and fs}
    return V, G["body"], joint


def seg_t(p, a, b):
    """Position along segment ab (0 at a, 1 at b) of the points p, and their distance from the segment."""
    ab = b - a
    t = ((p - a) @ ab) / (ab @ ab)
    q = a + np.clip(t, 0, 1)[:, None] * ab
    return t, np.linalg.norm(p - q, axis=1)


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def seated(V, J):
    """The legs' vertices bent to sitting (MakeHuman axes: x the person's left, y up, z forward); others unchanged.
    Also returns which vertices are legs (below the hips)."""
    out = V.copy()
    hip_y = J["r-upper-leg"][1]
    legs = V[:, 1] < hip_y + 0.3                               # decimetres: from just above the hip joints down
    legs &= np.abs(V[:, 0]) < 3.2                               # not the hands of the A-posed arms
    for side in ("l", "r"):
        hip, knee = J[f"{side}-upper-leg"], J[f"{side}-knee"]
        m = legs & ((V[:, 0] > 0) if side == "l" else (V[:, 0] <= 0))
        P = V[m]
        w_hip = smooth((hip_y + 0.3 - P[:, 1]) / 0.9)             # 0 above the hip joint, 1 a little below it
        w_knee = smooth((knee[1] + 0.3 - P[:, 1]) / 0.6)          # 0 above the knee, 1 below it
        R_t, R_k = rot_x(-math.pi / 2), rot_x(math.pi / 2)        # thigh forward; shin back down
        q = P.copy()
        for k in range(len(P)):
            a = -math.pi / 2 * w_hip[k]
            v = hip + rot_x(a) @ (P[k] - hip)
            if w_knee[k] > 0:
                kn = hip + R_t @ (knee - hip)
                v_t = hip + R_t @ (P[k] - hip)
                v_k = kn + R_k @ (v_t - kn)
                v = v * (1 - w_knee[k]) + v_k * w_knee[k]
            q[k] = v
        out[m] = q
    return out, legs


def arm_masks(V, J):
    """Which vertices belong to the right arm's hand, forearm and upper arm (near the chain shoulder-elbow-wrist-
    fingertip, outboard of the chest), and how far along the upper arm each vertex is. Given the mesh mirrored
    (x negated), the same masks pick out the left arm."""
    sh, el, wr = J["r-shoulder"], J["r-elbow"], J["r-hand"]
    tip = J["r-finger-3-4"]
    t_up, d_up = seg_t(V, sh, el)
    t_fo, d_fo = seg_t(V, el, wr)
    t_ha, d_ha = seg_t(V, wr, tip + (tip - wr) * 0.15)
    outboard = V[:, 0] < sh[0] + 0.25                     # the right side is -x in MakeHuman
    hand = (d_ha < 0.75) & (t_ha > -0.05) & outboard
    fore = ((d_fo < 0.65) & (t_fo > -0.08) & (t_fo <= 1.05) & outboard) | hand
    upper = (d_up < 0.75) & (t_up > 0.12) & (t_up < 1.10) & outboard & ~(fore & (t_fo > 0.12))
    return hand, fore, upper, t_up


def pieces(V, faces, J):
    sh, el, wr = J["r-shoulder"], J["r-elbow"], J["r-hand"]
    tip = J["r-finger-3-4"]
    Vm = V * np.array([-1.0, 1.0, 1.0])                   # mirrored: the left arm where the right one is
    hand, fore, upper, t_up = arm_masks(V, J)
    lhand, lfore, lupper, lt_up = arm_masks(Vm, J)
    arm, larm = upper | fore, lupper | lfore

    def face_set(mask, flip=False):
        return [f[::-1] if flip else f for f in faces if mask[f].all()]

    # the hand's frame: x along the middle finger, z out of the back of the hand (the thumb on the hand's left)
    uh = (tip - wr) / np.linalg.norm(tip - wr)
    k = np.cross(J["r-finger-2-1"] - J["r-finger-5-1"], tip - wr)
    wh = -(k - (k @ uh) * uh)
    wh /= np.linalg.norm(wh)
    hand_frame = np.stack([uh, np.cross(wh, uh), wh], 1)
    # the forearm's frame: x from the elbow to the wrist, z toward the back of the hand
    uf = (wr - el) / np.linalg.norm(wr - el)
    wf = wh - (wh @ uf) * uf
    wf /= np.linalg.norm(wf)
    fore_frame = np.stack([uf, np.cross(wf, uf), wf], 1)
    # upper arm piece: origin at the elbow, z toward the shoulder
    z = (sh - el) / np.linalg.norm(sh - el)
    x0 = np.cross([0.0, 1.0, 0.0], z)
    x0 /= np.linalg.norm(x0)
    up_frame = np.stack([x0, np.cross(z, x0), z], 1)
    # the left arm's pieces: the mirrored mesh in the right arm's frames, mirrored back within each piece (person.py
    # poses them with the same rules as the right arm's: the hand's and forearm's y and the upper arm's x flip)
    flip_y, flip_x = np.array([1.0, -1.0, 1.0]), np.array([-1.0, 1.0, 1.0])
    # body: hips origin, x forward (MakeHuman +z), y the person's left (MakeHuman +x), z up (MakeHuman +y)
    hips = J["pelvis"]
    body_pts = (V - hips)[:, [2, 0, 1]] * SCALE
    Vs, legs = seated(V, J)
    legs &= ~arm & ~larm
    legs_pts = (Vs - hips)[:, [2, 0, 1]] * SCALE

    # colours per vertex
    def colour(mask_shirt, mask_trousers, mask_hair):
        c = np.tile(SKIN, (len(V), 1))
        c[mask_shirt] = SHIRT
        c[mask_trousers] = TROUSERS
        c[mask_hair] = HAIR
        return c

    neck_y = J["neck"][1] - 0.25
    hip_y = J["pelvis"][1] + 0.45
    band = (V[:, 1] < neck_y) & (V[:, 1] > hip_y)
    shirt = (band & ~arm & ~larm) | ((t_up < SLEEVE_END) & upper) | ((lt_up < SLEEVE_END) & lupper)
    shirt |= band & (np.abs(V[:, 0]) < 1.9) & ~fore & ~lfore                                # the shoulders
    trousers = (V[:, 1] <= hip_y) & (V[:, 1] > J["l-ankle"][1] - 0.4)
    head = J["head"]
    hair = (V[:, 1] > head[1] + 0.45) | \
        ((V[:, 1] > head[1] - 0.55) & (V[:, 2] < head[2] - 0.15) & (np.abs(V[:, 0]) < 1.0))
    col = colour(shirt, trousers, hair)
    return {
        "person_hand": (flat_palm((V - wr) @ hand_frame * SCALE), face_set(hand), col),
        "person_forearm": ((V - wr) @ fore_frame * SCALE, face_set(fore & ~hand), col),
        "person_upper_arm": ((V - el) @ up_frame * SCALE, face_set(upper), col),
        "person_left_hand": (flat_palm((Vm - wr) @ hand_frame * SCALE * flip_y), face_set(lhand, True), col),
        "person_left_forearm": ((Vm - wr) @ fore_frame * SCALE * flip_y, face_set(lfore & ~lhand, True), col),
        "person_left_upper_arm": ((Vm - el) @ up_frame * SCALE * flip_x, face_set(lupper, True), col),
        "person_body": (body_pts, [f for f in faces if (~arm[f]).all() and (~larm[f]).all() and not legs[f].all()],
                        col),
        "person_legs": (legs_pts, [f for f in faces if legs[f].all()], col),
    }, {"forearm_len": float(np.linalg.norm(wr - el) * SCALE), "hand_len": float(np.linalg.norm(tip - wr) * SCALE),
        "upper_len": float(np.linalg.norm(sh - el) * SCALE),
        "body_frame": {j: ((J[j] - hips)[[2, 0, 1]] * SCALE).round(4).tolist()
                       for j in ("r-shoulder", "l-shoulder", "neck", "head")}}


def write_usd(path, pts, faces, col):
    from pxr import Gf, Usd, UsdGeom, Vt

    used = np.unique(np.concatenate([np.array(f) for f in faces]))
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    name = path.stem
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)   # moved by task.py
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts[used]]))
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([int(remap[i]) for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", __import__("pxr").Sdf.ValueTypeNames.Color3fArray,
                                                UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in col[used]]))
    stage.GetRootLayer().Save()


USED = ("person_hand", "person_forearm", "person_left_hand", "person_left_forearm")   # the model arms (task.py)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "base.obj"
    if not src.exists():
        urllib.request.urlretrieve(URL, src)
    V, faces, J = load(src)
    parts, dims = pieces(V, faces, J)
    for name, (pts, fs, col) in parts.items():
        if name not in USED:
            continue
        write_usd(HERE / f"{name}.usda", pts, fs, col)
        print("wrote", name, len(fs), "faces")
    print(dims)


if __name__ == "__main__":
    main()
