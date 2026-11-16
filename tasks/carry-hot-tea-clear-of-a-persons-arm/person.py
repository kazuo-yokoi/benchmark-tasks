"""person: the model arms' pieces posed along their lines (NumPy only).

The meshes come from make_person.py (MakeHuman's CC0 base mesh): ``person_hand`` and ``person_forearm`` (the right
arm) and the same two for the left (``person_left_*``). :func:`frames` turns where a model arm's wrist and elbow are,
and which way its hand points, into the hand's and forearm's rotations, in the pieces' own axes.
"""

from __future__ import annotations

import math

import numpy as np

HAND, FOREARM = 0.177, 0.226                        # m: wrist to middle fingertip, elbow to wrist (make_person.py)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def quat_xyzw(R) -> tuple[float, float, float, float]:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)
    return tuple(float(v) for v in q)


def hand_frame(heading):
    """Hand axes (columns: along the fingers, the hand's left, the back of the hand) for a heading."""
    u = unit(heading)
    v = np.cross([0.0, 0.0, 1.0], u)
    v = unit(v) if np.linalg.norm(v) > 1e-6 else np.array([0.0, 1.0, 0.0])
    return np.stack([u, v, np.cross(u, v)], 1)


def frames(W, e, heading):
    """Rotations of the hand (origin at the wrist ``W``) and the forearm (from the elbow ``e`` to the wrist, z toward
    the back of the hand) of a model arm whose hand points along ``heading``."""
    H = hand_frame(heading)
    w = H[:, 2]
    xf = unit(np.asarray(W, float) - np.asarray(e, float))
    zf = unit(w - (w @ xf) * xf)
    return H, np.stack([xf, np.cross(zf, xf), zf], 1)


def mesh_points(name):
    """The vertices of a piece's mesh (``<name>.usda`` next to this file), in the piece's own axes."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent / f"{name}.usda").read_text()
    body = re.search(r"point3f\[\] points = \[(.*?)\]", text, re.S).group(1)
    return np.array([[float(v) for v in t.split(",")] for t in re.findall(r"\(([^()]*)\)", body)])


def hull(xy):
    """Convex hull (counter-clockwise, no repeats) of 2-D points."""
    pts = sorted(set(map(tuple, np.round(np.asarray(xy, float), 6))))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def outline_dist(poly, pts):
    """Signed distance (m; negative inside) from points (N x 2) to a convex counter-clockwise polygon."""
    pts = np.atleast_2d(np.asarray(pts, float))
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a                                                    # edges
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    d = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(1)
    inside = ((ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]) >= 0).all(1)
    return np.where(inside, -d, d)
