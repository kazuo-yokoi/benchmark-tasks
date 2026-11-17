"""Writes print/model_arm_right.stl and print/model_arm_left.stl: each model arm (forearm and hand, as task.py poses
them: the line rising 0.15 toward the elbow, the hand flat) as one closed mesh in millimetres, for 3D printing at
full size. The forearm's cut ends are capped flat. Origin at the middle fingertip, x along the arm toward the
fingertips, z up. NumPy only: python export_stl.py"""

import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

import person as P

HERE = Path(__file__).resolve().parent
ARM_TILT = 0.15


def read(name):
    text = (HERE / f"{name}.usda").read_text()
    pts = P.mesh_points(name)
    counts = [int(v) for v in re.search(r"faceVertexCounts = \[(.*?)\]", text, re.S).group(1).split(",")]
    idx = [int(v) for v in re.search(r"faceVertexIndices = \[(.*?)\]", text, re.S).group(1).split(",")]
    tris, k = [], 0
    for c in counts:
        f = idx[k:k + c]
        k += c
        tris += [(f[0], f[i], f[i + 1]) for i in range(1, c - 1)]
    return pts, np.array(tris)


def cap(pts, tris):
    """Close each open boundary loop with a fan around its centroid."""
    edges = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] += 1
    nxt = {a: b for (a, b) in edges if (b, a) not in edges}     # boundary edges, as the faces run
    pts, new, seen = list(pts), [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if len(loop) < 3:
            continue
        pts.append(np.mean([pts[i] for i in loop], 0))
        c = len(pts) - 1
        new += [(loop[(i + 1) % len(loop)], loop[i], c) for i in range(len(loop))]
    return np.array(pts), np.vstack([tris, new]) if new else tris


def arm(pre):
    u = P.unit(np.r_[1.0, 0.0, -ARM_TILT])
    W = -P.HAND * u
    e = W - P.FOREARM * u
    Rh, Rf = P.frames(W, e, u)
    out_p, out_t = [], []
    for name, R in ((pre + "hand", Rh), (pre + "forearm", Rf)):
        p, t = read(name)
        p, t = cap(p, t)
        v = p[t]
        if np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() < 0:   # the left pieces are mirrored
            t = t[:, ::-1]
        out_t.append(t + sum(len(q) for q in out_p))
        out_p.append(W + p @ R.T)
    return np.vstack(out_p) * 1000.0, np.vstack(out_t)


def write_stl(path, pts, tris):
    with open(path, "wb") as f:
        f.write(b"model arm, mm".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *pts[a], *pts[b], *pts[c], 0))


def main():
    (HERE / "print").mkdir(exist_ok=True)
    for side, pre in (("right", "person_"), ("left", "person_left_")):
        pts, tris = arm(pre)
        write_stl(HERE / "print" / f"model_arm_{side}.stl", pts, tris)
        lo, hi = pts.min(0), pts.max(0)
        print(side, len(tris), "triangles, size (mm)", np.round(hi - lo, 1))


if __name__ == "__main__":
    main()
