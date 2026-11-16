"""Writes assets/bin.usda: the open bin as one rigid body (a floor, four walls and a handle tab rising 5 cm above
each of its two end walls, the ones across the robot's left-right axis), origin at the centre of its underside. Run it with any Python that has pxr (the Arena venv does): python make_bins.py"""

from pathlib import Path

INNER_X, INNER_Y, WALL, HEIGHT, FLOOR = 0.12, 0.12, 0.006, 0.07, 0.006
TAB = (0.03, 0.012, 0.05)  # a handle tab on each end wall (+y and -y): 3 cm wide, 12 mm thick, 5 cm above the rim
COLOR = (0.15, 0.40, 0.80)


def main() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    out = Path(__file__).resolve().parent / "assets" / "bin.usda"
    out.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Bin")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim())
    ox, oy = INNER_X + 2 * WALL, INNER_Y + 2 * WALL
    parts = {
        "floor": ((0, 0, FLOOR / 2), (ox, oy, FLOOR)),
        "wall_n": ((0, INNER_Y / 2 + WALL / 2, HEIGHT / 2), (ox, WALL, HEIGHT)),
        "wall_s": ((0, -INNER_Y / 2 - WALL / 2, HEIGHT / 2), (ox, WALL, HEIGHT)),
        "wall_e": ((INNER_X / 2 + WALL / 2, 0, HEIGHT / 2), (WALL, INNER_Y, HEIGHT)),
        "wall_w": ((-INNER_X / 2 - WALL / 2, 0, HEIGHT / 2), (WALL, INNER_Y, HEIGHT)),
        "handle_n": ((0, INNER_Y / 2 + WALL / 2, HEIGHT + TAB[2] / 2), TAB),
        "handle_s": ((0, -INNER_Y / 2 - WALL / 2, HEIGHT + TAB[2] / 2), TAB),
    }
    for name, (pos, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/Bin/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*COLOR)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
