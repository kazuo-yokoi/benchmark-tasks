"""Writes the trainer hand's printable parts as STL files (millimetres) into print/, from the same geometry as the
simulation (make_assets.py). Pure Python + numpy:

    python export_stl.py

- forearm.stl: the forearm shell (print hollow, 3 walls, 10 % infill; a 1 kg steel-shot bag inside brings it to
  1.2 kg). Its wrist end takes the pan-tilt bracket.
- palm_thumb.stl: palm and thumb, one part; bolts to the tilt servo's horn.
- fingers.stl: index to little finger as one curled block; its knuckle end turns on the release servo's horn, so the
  fingers swing open together.
Each finger and the thumb are tubes along the same centre lines as the simulation's capsules, so the cup sits in the
printed hand exactly as in the simulation. Coat the fingers' inner faces with a thin layer of silicone (e.g. a
tool-handle dip) for the light friction grip.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import make_assets as A

HERE = Path(__file__).resolve().parent


def triangles(pts, faces):
    pts = np.asarray(pts, float)
    for f in faces:
        for k in range(1, len(f) - 1):
            yield pts[f[0]], pts[f[k]], pts[f[k + 1]]


def write_stl(path: Path, parts) -> int:
    tris = [t for pts, faces in parts for t in triangles(pts, faces)]
    with open(path, "wb") as fh:
        fh.write(b"trainer hand part (mm)".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            fh.write(struct.pack("<12fH", *n, *(a * 1000), *(b * 1000), *(c * 1000), 0))
    return len(tris)


def main() -> None:
    out = HERE / "print"
    out.mkdir(exist_ok=True)
    fingers, thumb = A.hand_parts()
    lo = (A.PALM_X[0], A.PALM_IN_Y, A.PALM_Z[0])
    hi = (A.PALM_X[1] + 0.02, A.PALM_IN_Y + A.PALM_T, A.PALM_Z[1])
    parts = {
        "forearm.stl": [A.forearm_mesh()],
        "palm_thumb.stl": [A.rounded_box(lo, hi, 0.01), A.tube(thumb, A.THUMB[0])],
        "fingers.stl": [A.tube(path, r) for _, path, r in fingers],
    }
    for name, meshes in parts.items():
        n = write_stl(out / name, meshes)
        print(f"wrote print/{name} ({n} triangles)")


if __name__ == "__main__":
    main()
