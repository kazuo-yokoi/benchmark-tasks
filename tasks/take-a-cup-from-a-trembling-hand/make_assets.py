"""Writes the task's USD files into assets/. Run with any Python that has pxr (the Arena venv does):

    python make_assets.py

- hand.usda: the trainer forearm and right hand holding out a cup, as one articulation with a free root:
  ``forearm`` (rests on the armrest) -> ``flex`` (vertical axis at the wrist: the pan servo) -> ``wrist`` ->
  ``roll`` (along the forearm: the tilt servo, turning the hand palm up / palm down) -> ``hand`` (palm and thumb) -> ``release`` (vertical
  axis at the knuckles: the release servo) -> ``fingers`` (index to little finger, the little finger curled under the
  cup's base).
- cup.usda: a 7 cm polypropylene tumbler, 14 cm tall, 70 g.

Frame of hand.usda ("hand frame"): origin at the centre of the cup's base as the hand holds it, x toward the robot's
far side (along the forearm, to the elbow), y to the robot's left (the palm side), z up. The robot is on the -x side.
The hand is thumb up with its palm against the cup's +y side; the fingers wrap the cup's front (-x, the robot's side)
and its -y side, the thumb its back (+x).

Visual meshes are smooth tubes and rounded boxes; collisions are capsules (fingers, thumb), boxes (palm) and convex
hulls (forearm, cup).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_SEG = 32

# -- cup ------------------------------------------------------------------------------------------------------------
CUP_R, CUP_H, CUP_MASS, CUP_WALL = 0.035, 0.14, 0.070, 0.0015
CUP_COLOR = (0.55, 0.78, 0.92)

# -- hand (hand frame, m) --------------------------------------------------------------------------------------------
SKIN = (0.46, 0.29, 0.19)
SLEEVE = (0.07, 0.09, 0.17)
PALM_IN_Y = CUP_R + 0.001              # the palm's inner face, against the cup
PALM_T = 0.026
PALM_X = (-0.045, 0.050)               # knuckles .. wrist crease
PALM_Z = (-0.010, 0.078)               # little-finger edge .. index knuckle
WRIST = np.array([0.068, PALM_IN_Y + PALM_T / 2, 0.034])     # where the two wrist axes cross
KNUCKLE = np.array([PALM_X[0] + 0.004, PALM_IN_Y + 0.012])   # the release axis (vertical) in x, y
ELBOW_X = 0.335
FOREARM_BOTTOM = 0.008                 # the forearm's flat underside, resting on the armrest
FOREARM_MASS, WRIST_MASS, HAND_MASS, FINGERS_MASS = 1.20, 0.05, 0.18, 0.06
# fingers: (name, radius, height of the centre line, end angle round the cup); angles from +x, anticlockwise from above
FINGERS = [("index", 0.0085, 0.062, 268.0), ("middle", 0.0090, 0.0435, 276.0), ("ring", 0.0085, 0.0255, 270.0),
           ("little", 0.0075, 0.0085, 250.0)]
FINGER_START = 158.0                   # where each finger meets the cup, past the knuckle
THUMB = (0.0100, 0.0600, 72.0, -38.0)  # radius, height, start and end angle (round the back)
GAP = 0.0006                           # finger surfaces this far off the cup (contact without overlap)
OPEN_DEG = -38.0                       # the release joint's open position (fingers swung away from the cup)


def _q_from_to(a, b):
    """Quaternion (w, x, y, z) turning unit vector a onto b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return (0.0, *axis)
    s = math.sqrt((1 + c) * 2)
    return (s / 2, *(v / s))


def arc_points(r, z, a0, a1, n):
    return [np.array([r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), z])
            for a in np.linspace(a0, a1, n)]


def finger_path(radius, z, a_end):
    """Knuckle -> round the cup at its surface + radius, as a polyline (hand frame)."""
    rr = CUP_R + radius + GAP
    n = max(3, int(abs(a_end - FINGER_START) / 18) + 2)
    pts = arc_points(rr + 0.0009, z, FINGER_START, a_end, n)   # chords sag about 1 mm inward
    start = np.array([KNUCKLE[0], KNUCKLE[1] - 0.004, z])
    return [start] + pts


def little_path():
    r, z, a_end = FINGERS[3][1], FINGERS[3][2], FINGERS[3][3]
    pts = finger_path(r, z, 205.0)
    # the last phalanx curls in under the base
    a = math.radians(212.0)
    rim = np.array([(CUP_R + r + 0.0035) * math.cos(a), (CUP_R + r + 0.0035) * math.sin(a), -r - 0.0004])
    tip_under = np.array([-0.012, -0.018, -r - 0.0004])
    return pts + [rim, tip_under]


def thumb_path():
    r, z, a0, a1 = THUMB
    rr = CUP_R + r + GAP + 0.0009
    base = np.array([0.040, PALM_IN_Y + 0.004, 0.040])
    return [base] + arc_points(rr, z, a0, a1, int(abs(a1 - a0) / 18) + 2)


# -- meshes -----------------------------------------------------------------------------------------------------------
def revolve(profile, n=N_SEG):
    pts, faces = [], []
    m = len(profile)
    for k in range(n):
        a = 2 * math.pi * k / n
        for r, z in profile:
            pts.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(n):
        k2 = (k + 1) % n
        for i in range(m - 1):
            faces.append([k * m + i, k2 * m + i, k2 * m + i + 1, k * m + i + 1])
    return pts, faces


def tube(path, radius, n=12, cap=True):
    """A smooth tube along a polyline, closed with hemispheres (rounded fingertips)."""
    path = [np.asarray(p, float) for p in path]
    # densify
    dense = []
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(np.linalg.norm(b - a) / 0.004))
        for u in np.linspace(0, 1, k, endpoint=False):
            dense.append(a + u * (b - a))
    dense.append(path[-1])
    rings = []
    prev_n = None
    for i, p in enumerate(dense):
        t = dense[min(i + 1, len(dense) - 1)] - dense[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        if prev_n is None:
            ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            nrm = np.cross(t, ref)
        else:
            nrm = prev_n - np.dot(prev_n, t) * t
        nrm /= np.linalg.norm(nrm)
        prev_n = nrm
        bi = np.cross(t, nrm)
        rings.append([p + radius * (math.cos(2 * math.pi * j / n) * nrm + math.sin(2 * math.pi * j / n) * bi)
                      for j in range(n)])
    pts = [q for ring in rings for q in ring]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append([i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j])
    if cap:
        for end, sgn in ((0, -1), (len(dense) - 1, 1)):
            p = dense[end]
            t = dense[end] - dense[end - 1] if sgn > 0 else dense[0] - dense[1]
            t /= np.linalg.norm(t)
            c = len(pts)
            pts.append(p + radius * t)
            ring0 = end * n
            for j in range(n):
                j2 = (j + 1) % n
                faces.append([ring0 + j, ring0 + j2, c] if sgn < 0 else [ring0 + j2, ring0 + j, c])
    return [tuple(map(float, q)) for q in pts], faces


def rounded_box(lo, hi, r, n=6):
    """A box with rounded edges (superellipsoid-ish), as a closed mesh."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    pts, faces = [], []
    nu, nv = 24, 12
    for i in range(nv + 1):
        v = -math.pi / 2 + math.pi * i / nv
        for j in range(nu):
            u = 2 * math.pi * j / nu
            e = 0.25                                  # squareness
            cu, su, cv, sv = math.cos(u), math.sin(u), math.cos(v), math.sin(v)
            f = lambda x: math.copysign(abs(x) ** e, x)  # noqa: E731
            pts.append(tuple(map(float, c + h * np.array([f(cv) * f(cu), f(cv) * f(su), f(sv)]))))
    for i in range(nv):
        for j in range(nu):
            j2 = (j + 1) % nu
            faces.append([i * nu + j, i * nu + j2, (i + 1) * nu + j2, (i + 1) * nu + j])
    return pts, faces


def forearm_mesh(n=20):
    """Tapered, flat-bottomed forearm from the wrist (x = WRIST) to the elbow (+x)."""
    rings = []
    xs = np.linspace(WRIST[0] - 0.004, ELBOW_X, 14)
    for x in xs:
        u = (x - xs[0]) / (xs[-1] - xs[0])
        w = 0.032 + 0.014 * math.sin(min(1.0, u * 1.4) * math.pi / 2)    # half width
        hgt = 0.046 + 0.026 * math.sin(min(1.0, u * 1.4) * math.pi / 2)  # height
        ring = []
        for j in range(n):
            a = 2 * math.pi * j / n
            y = WRIST[1] + w * math.cos(a)
            zc = FOREARM_BOTTOM + hgt / 2
            z = zc + hgt / 2 * math.sin(a)
            z = max(z, FOREARM_BOTTOM + 0.002 * (1 - abs(math.cos(a))))
            ring.append((float(x), float(y), float(z)))
        rings.append(ring)
    pts = [p for r in rings for p in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append([i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j])
    for end, sgn in ((0, -1), (len(rings) - 1, 1)):
        c = len(pts)
        ring = np.array(rings[end])
        pts.append(tuple(map(float, ring.mean(axis=0))))
        for j in range(n):
            j2 = (j + 1) % n
            faces.append([end * n + j, end * n + j2, c] if sgn > 0 else [end * n + j2, end * n + j, c])
    return pts, faces


# -- USD helpers ------------------------------------------------------------------------------------------------------
def new_stage(name):
    from pxr import Usd, UsdGeom, UsdPhysics

    out = HERE / "assets" / f"{name}.usda"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage, out


def mesh(stage, path, pts, faces, color, collide=False, opacity=None, offset=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    off = np.asarray(offset, float)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(np.asarray(p) - off)) for p in pts]))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity is not None:
        m.CreateDisplayOpacityAttr([opacity])
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return m


def capsule(stage, path, a, b, r, offset=(0.0, 0.0, 0.0)):
    """An invisible capsule collider from a to b (body frame = hand frame minus ``offset``)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    a, b = np.asarray(a, float) - offset, np.asarray(b, float) - offset
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateAxisAttr("Z")
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(np.linalg.norm(b - a)))
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
    w, *v = _q_from_to([0, 0, 1], b - a)
    x.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(*map(float, v))))
    c.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def box_collider(stage, path, lo, hi, offset=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics

    lo, hi = np.asarray(lo, float) - offset, np.asarray(hi, float) - offset
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    x.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
    c.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def rigid(stage, path, mass, com, at):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, at)))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    m = UsdPhysics.MassAPI.Apply(x.GetPrim())
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, np.asarray(com, float) - at)))
    return x


def revolute(stage, path, b0, b1, at0, at1, axis, lo=None, hi=None):
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([Sdf.Path(b0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(b1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at1)))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    if lo is not None:
        j.CreateLowerLimitAttr(float(lo))
        j.CreateUpperLimitAttr(float(hi))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(0.0)
    return j


def hand_parts():
    """Everything the hand is made of, in the hand frame: {part: [(path polyline, radius)]}."""
    fingers = []
    for name, r, z, a_end in FINGERS[:3]:
        fingers.append((name, finger_path(r, z, a_end), r))
    fingers.append(("little", little_path(), FINGERS[3][1]))
    return fingers, thumb_path()


def hand():
    from pxr import UsdGeom, UsdPhysics

    stage, out = new_stage("hand")
    root = UsdGeom.Xform.Define(stage, "/Hand")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # forearm (the articulation's root link: free, resting on the armrest)
    fa_at = np.array([0.20, WRIST[1], 0.04])
    rigid(stage, "/Hand/forearm", FOREARM_MASS, (0.21, WRIST[1], 0.035), fa_at)
    p, f = forearm_mesh()
    mesh(stage, "/Hand/forearm/skin", p, f, SLEEVE, collide=True, offset=fa_at)
    # the pan-tilt bracket's link between the two wrist axes (no collider)
    rigid(stage, "/Hand/wrist", WRIST_MASS, WRIST, WRIST)
    p, f = rounded_box(WRIST - [0.012, 0.02, 0.02], WRIST + [0.012, 0.02, 0.02], 0.01)
    mesh(stage, "/Hand/wrist/cuff", p, f, SKIN, offset=WRIST)

    # hand: palm and thumb
    h_at = np.array([0.0, PALM_IN_Y + PALM_T / 2, 0.035])
    rigid(stage, "/Hand/hand", HAND_MASS, (0.012, h_at[1], 0.036), h_at)
    lo, hi = (PALM_X[0], PALM_IN_Y, PALM_Z[0]), (PALM_X[1] + 0.02, PALM_IN_Y + PALM_T, PALM_Z[1])
    p, f = rounded_box(lo, hi, 0.01)
    mesh(stage, "/Hand/hand/palm", p, f, SKIN, offset=h_at)
    box_collider(stage, "/Hand/hand/palm_col", (lo[0] + 0.004, lo[1], lo[2] + 0.004),
                 (PALM_X[1], hi[1], hi[2] - 0.004), offset=h_at)    # stops short of the forearm
    fingers, thumb = hand_parts()
    p, f = tube(thumb, THUMB[0])
    mesh(stage, "/Hand/hand/thumb", p, f, SKIN, offset=h_at)
    for i, (a, b) in enumerate(zip(thumb[:-1], thumb[1:])):
        capsule(stage, f"/Hand/hand/thumb_col{i}", a, b, THUMB[0], offset=h_at)

    # fingers, on the release joint at the knuckles
    k_at = np.array([KNUCKLE[0], KNUCKLE[1], 0.035])
    rigid(stage, "/Hand/fingers", FINGERS_MASS, (-0.030, 0.0, 0.035), k_at)
    for name, path, r in fingers:
        p, f = tube(path, r)
        mesh(stage, f"/Hand/fingers/{name}", p, f, SKIN, offset=k_at)
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            capsule(stage, f"/Hand/fingers/{name}_col{i}", a, b, r, offset=k_at)

    UsdGeom.Scope.Define(stage, "/Hand/joints")
    revolute(stage, "/Hand/joints/flex", "/Hand/forearm", "/Hand/wrist", WRIST - fa_at, (0, 0, 0), "Z", -40, 40)
    revolute(stage, "/Hand/joints/roll", "/Hand/wrist", "/Hand/hand", (0, 0, 0), WRIST - h_at, "X", -40, 40)
    revolute(stage, "/Hand/joints/release", "/Hand/hand", "/Hand/fingers", k_at - h_at, (0, 0, 0), "Z",
             OPEN_DEG - 2, 2)
    # the links touch each other at the wrist and knuckles: no self-collision
    from pxr import Sdf
    prim = root.GetPrim()
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


def cup():
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, out = new_stage("cup")
    root = UsdGeom.Xform.Define(stage, "/Body")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    m = UsdPhysics.MassAPI.Apply(root.GetPrim())
    m.CreateMassAttr(CUP_MASS)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.045))
    r, h, w = CUP_R, CUP_H, CUP_WALL
    prof = [(0.0, 0.0), (r - 0.003, 0.0), (r, 0.003), (r, h), (r - w, h), (r - w, 0.004), (0.0, 0.004)]
    p, f = revolve(prof)
    mesh(stage, "/Body/tumbler", p, f, CUP_COLOR, opacity=0.85)
    p, f = revolve([(0.0, 0.0), (r, 0.0), (r, h), (0.0, h)], n=24)
    c = mesh(stage, "/Body/hull", p, f, CUP_COLOR, collide=True)
    c.CreatePurposeAttr("guide")
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


def main() -> None:
    hand()
    cup()


if __name__ == "__main__":
    main()
