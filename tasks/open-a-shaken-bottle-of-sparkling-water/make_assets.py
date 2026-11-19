"""Writes the task's USD files into assets/ (any Python with pxr; the Arena venv has it):

    python make_assets.py

bottle.usda: a 500 mL clear PET carbonation bottle of chilled sparkling water with its 28 mm screw cap, as one
articulation with a free root. ``bottle`` (body, water as rigid mass) -> ``cap_x``, ``cap_y``, ``cap_z`` (prismatic) ->
``cap_rx``, ``cap_ry`` -> ``twist`` (revolute about the bottle's axis) -> ``cap``. While the thread holds, the task file
locks the five other joints stiff, so the cap only turns; once it has turned 1.55 turns the thread lets go and the cap
is free (all six joints slack). Origin: the centre of the bottle's base; +z up its axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import fizz

HERE = Path(__file__).resolve().parent
N_SEG = 40
R_BODY, H_BOTTLE = 0.0325, 0.205
CAP_R, CAP_H, CAP_Z0 = 0.0150, 0.018, 0.190          # the cap covers 190..208 mm
BOTTLE_MASS, CAP_MASS, LINK_MASS = 0.56, 0.012, 0.004
PET = (0.55, 0.70, 0.78)
WATER = (0.12, 0.38, 0.85)
CAP_BLUE = (0.10, 0.35, 0.80)
LABEL = (0.15, 0.55, 0.30)
# outside of the bottle (r, z): base, body, shoulder, neck, tamper ring, rim
OUTER = [(0.0, 0.0), (0.028, 0.0), (0.0325, 0.006), (0.0325, 0.150), (0.0315, 0.160), (0.0255, 0.172), (0.0165, 0.182),
         (0.0120, 0.186), (0.0160, 0.186), (0.0160, 0.189), (0.0125, 0.189), (0.0125, 0.205), (0.0, 0.205)]


def revolve(profile, n=N_SEG, z_shift=0.0):
    pts, faces = [], []
    m = len(profile)
    for k in range(n):
        a = 2 * math.pi * k / n
        for r, z in profile:
            pts.append((r * math.cos(a), r * math.sin(a), z + z_shift))
    for k in range(n):
        k2 = (k + 1) % n
        for i in range(m - 1):
            faces.append([k * m + i, k2 * m + i, k2 * m + i + 1, k * m + i + 1])
    return pts, faces


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
        m.CreatePurposeAttr("guide")
    return m


def link(stage, path, mass, at, com=None, inertia=None):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, at)))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    m = UsdPhysics.MassAPI.Apply(x.GetPrim())
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com if com is not None else (0, 0, 0))))
    if inertia is not None:
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    return x


def joint(stage, path, kind, b0, b1, at0, axis, lo=None, hi=None):
    from pxr import Gf, Sdf, UsdPhysics

    j = (UsdPhysics.PrismaticJoint if kind == "p" else UsdPhysics.RevoluteJoint).Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([Sdf.Path(b0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(b1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    if lo is not None:
        j.CreateLowerLimitAttr(float(lo))
        j.CreateUpperLimitAttr(float(hi))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if kind == "p" else "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(0.0)
    return j


def bottle():
    from pxr import Sdf, UsdGeom, UsdPhysics

    stage, out = new_stage("bottle")
    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    prim = root.GetPrim()
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)

    link(stage, "/Bottle/bottle", BOTTLE_MASS, (0, 0, 0), com=(0, 0, 0.075))
    p, f = revolve(OUTER)
    mesh(stage, "/Bottle/bottle/hull", p, f, PET, collide=True)
    # the clear PET is drawn only as its edges (a transparent wall would hide the water and foam in this renderer):
    # the base, a ring at the shoulder, the neck and its tamper ring
    for i, (r0, z0, r1, z1) in enumerate([(0.0, 0.0, 0.0330, 0.004), (0.0315, 0.158, 0.0330, 0.161),
                                          (0.0110, 0.182, 0.0128, 0.205), (0.0128, 0.185, 0.0162, 0.189)]):
        p, f = revolve([(r0 * 0.0, z0), (r1, z0), (r1, z1), (r0 * 0.0, z1)])
        mesh(stage, f"/Bottle/bottle/edge{i}", p, f, PET)
    for i, (r, z) in enumerate([(0.0300, 0.166), (0.0215, 0.177)]):          # thin rings marking the shoulder
        p, f = revolve([(r - 0.0006, z), (r + 0.0006, z), (r + 0.0006, z + 0.0012), (r - 0.0006, z + 0.0012)])
        mesh(stage, f"/Bottle/bottle/ring{i}", p, f, PET)
    p, f = revolve([(0.0, 0.004), (0.0310, 0.004), (0.0310, fizz.Z_WATER), (0.0, fizz.Z_WATER)])
    mesh(stage, "/Bottle/bottle/water", p, f, WATER)
    p, f = revolve([(0.0330, 0.070), (0.0330, 0.120)])
    mesh(stage, "/Bottle/bottle/label", p, f, LABEL)

    # the cap's chain: three slides and two tilts (stiff while threaded), then the twist
    z = CAP_Z0
    names = ["cap_x", "cap_y", "cap_z", "cap_rx", "cap_ry"]
    prev = "/Bottle/bottle"
    for i, nm in enumerate(names):
        lk = f"/Bottle/l_{nm}"
        link(stage, lk, LINK_MASS, (0, 0, z), inertia=(1e-6, 1e-6, 1e-6))
        kind, axis = ("p", "XYZ"[i]) if i < 3 else ("r", "XY"[i - 3])
        at0 = (0, 0, z) if prev == "/Bottle/bottle" else (0, 0, 0)
        joint(stage, f"/Bottle/joints/{nm}", kind, prev, lk, at0, axis)
        prev = lk
    link(stage, "/Bottle/cap", CAP_MASS, (0, 0, z), com=(0, 0, CAP_H / 2), inertia=(2e-6, 2e-6, 2e-6))
    p, f = revolve([(0.0, 0.0), (CAP_R, 0.0), (CAP_R, CAP_H), (0.0, CAP_H)], n=48)
    mesh(stage, "/Bottle/cap/shell", p, f, CAP_BLUE)
    mesh(stage, "/Bottle/cap/hull", p, f, CAP_BLUE, collide=True)
    joint(stage, "/Bottle/joints/twist", "r", prev, "/Bottle/cap", (0, 0, 0), "Z", -2.0,
          math.degrees(fizz.THETA_OFF) + 400.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


if __name__ == "__main__":
    bottle()
