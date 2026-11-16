"""Writes tea_cup.usda: a handleless ceramic tea cup filled with tea, as one rigid body.

    python make_cup.py        # needs pxr (usd-core, or Isaac Sim's Python)

The body's origin is the centre of the cup's base. It collides as one solid cylinder (invisible) of the cup's outside
size; what is drawn is a thin-walled cup (outside, inside, rim and foot) with the tea's surface 2 cm below the rim.
The simulator has no liquid: the tea is part of the cup's mass.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

R_OUT, HEIGHT, WALL, BASE = 0.035, 0.085, 0.004, 0.008   # m: 7 cm wide, 8.5 cm tall, 4 mm walls, 8 mm thick base
FILL_BELOW_RIM = 0.02                                   # the tea's surface, 2 cm below the rim
MASS = 0.28                                             # kg: a 120 g cup and 160 ml of tea
N = 48
CERAMIC = (0.93, 0.91, 0.86)
TEA = (0.45, 0.24, 0.08)
HERE = Path(__file__).resolve().parent


def _ring(r, z):
    return [Gf.Vec3f(r * math.cos(2 * math.pi * k / N), r * math.sin(2 * math.pi * k / N), z) for k in range(N)]


def _tube(r, z0, z1, outward=True):
    """Points and quads of an open cylinder from z0 to z1 (normals out, or in)."""
    pts = _ring(r, z0) + _ring(r, z1)
    faces = []
    for k in range(N):
        a, b, c, d = k, (k + 1) % N, N + (k + 1) % N, N + k
        faces.append((a, b, c, d) if outward else (a, d, c, b))
    return pts, faces


def _annulus(r0, r1, z, up=True):
    pts = _ring(r0, z) + _ring(r1, z)
    faces = []
    for k in range(N):
        a, b, c, d = k, (k + 1) % N, N + (k + 1) % N, N + k
        faces.append((a, d, c, b) if up else (a, b, c, d))
    return pts, faces


def _disc(r, z, up=True):
    pts = [Gf.Vec3f(0, 0, z)] + _ring(r, z)
    faces = [(0, 1 + k, 1 + (k + 1) % N) if up else (0, 1 + (k + 1) % N, 1 + k) for k in range(N)]
    return pts, faces


def _mesh(stage, path, parts, material):
    pts, counts, idx = [], [], []
    for p, faces in parts:
        off = len(pts)
        pts += p
        for f in faces:
            counts.append(len(f))
            idx += [off + i for i in f]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _material(stage, path, rgb, roughness):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def build(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/tea_cup")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.035))  # cup and tea together: below the middle

    collider = UsdGeom.Cylinder.Define(stage, "/tea_cup/collider")
    collider.CreateRadiusAttr(R_OUT)
    collider.CreateHeightAttr(HEIGHT)
    collider.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(collider).SetTranslate(Gf.Vec3d(0, 0, HEIGHT / 2))
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)       # collides, not drawn
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    ceramic = _material(stage, "/tea_cup/looks/ceramic", CERAMIC, 0.35)
    tea = _material(stage, "/tea_cup/looks/tea", TEA, 0.08)
    r_in = R_OUT - WALL
    _mesh(stage, "/tea_cup/cup", [
        _tube(R_OUT, 0.0, HEIGHT, outward=True),
        _tube(r_in, BASE, HEIGHT, outward=False),
        _annulus(r_in, R_OUT, HEIGHT, up=True),
        _disc(R_OUT, 0.0, up=False),
        _disc(r_in, BASE, up=True),
    ], ceramic)
    _mesh(stage, "/tea_cup/tea", [_disc(r_in - 0.0005, HEIGHT - FILL_BELOW_RIM, up=True)], tea)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    build(HERE / "tea_cup.usda")
    print("wrote", HERE / "tea_cup.usda")
