"""perceive: counts the parts falling out of the drop tube, from the top camera's image alone (demo.py).

The image is the one Dream-1 is sent: the top camera's frame scaled to 320 x 176 and JPEG-encoded at quality 85.
The demo watches a small window of it: the column of air between the tube's mouth and the rim of the bin under it,
found by projecting that column with the top camera's calibration (rig pose and lens). A falling part crosses the
window in two or three frames and leaves it as it was, so the demo counts a part each time the window suddenly
differs from how it looked just before, and then settles back.

While an arm that has just moved is in or next to the window (its outline drawn from its joint angles: the URDF's
forward kinematics, the links as 6.5 cm spheres, widened by NEAR_PX pixels for its shadow), the demo does not count,
so the robot's own movement is never taken for a part; an arm that keeps still changes nothing in the image. The demo
keeps its arms away from the window while parts may fall. Nothing here reads the simulator.
"""

from __future__ import annotations

import io

import numpy as np

W, H = 320, 176            # Dream-1's frame size
ROBOT_R = 0.065            # m, the robot's links drawn as spheres this big
DIFF = 60                  # summed |RGB| change for a pixel to count as changed
MIN_PIX = 4                # changed pixels for the window to count as disturbed
REFRACTORY = 0.8           # s: parts come at most every 1.8 s, so one fall is never counted twice
MOVED = 1e-3               # rad: a joint command changing by more than this in a step is a moving arm
NEAR_PX = 12               # pixels around the window where a moving arm (or its shadow) stops the count
STILL_S = 0.3              # s: an arm is masked until it has kept still this long
BG_RATE = 0.3              # how fast the window's picture of the empty air follows slow changes (light, shadows)


def as_dream_sees(rgb: np.ndarray) -> np.ndarray:
    """The frame as Dream-1's client sends it: 320 x 176, JPEG quality 85."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).convert("RGB").resize((W, H), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).astype(np.int16)


class TopCamera:
    """The top camera's calibration (pose and lens, from the rig)."""

    def __init__(self):
        from yam_arena import station

        cam = station.rig("calibrated").top_cam
        self.pos = np.asarray(cam.pos, float)
        x, y, z, w = cam.rot_xyzw
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                           [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                           [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        self.lens = cam.lens

    def project(self, pts):
        """Pixels (u, v) and depths of world points (N, 3) in a W x H frame (OpenGL camera: looking along -z)."""
        L = self.lens
        sx, sy = W / L.width, H / L.height
        pc = (np.asarray(pts, float) - self.pos) @ self.R
        z = -pc[:, 2]
        u = L.fx * sx * pc[:, 0] / z + L.cx * sx - 0.5
        v = -L.fy * sy * pc[:, 1] / z + L.cy * sy - 0.5
        return u, v, z

    def robot_outline(self, joints: dict) -> np.ndarray:
        """Boolean (H, W): where the robot's arms are in the image, from its joint angles (``{"left": q, ...}``)."""
        from yam_arena import YAM_URDF, station
        from yam_arena.constants import ARM_JOINTS
        from yam_arena.urdf import forward_kinematics

        pts = []
        for side, q in joints.items():
            fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q[:6])})
            org = [np.asarray(T[:3, 3]) + np.asarray(station.BASES[side]) for T in fk.values()]
            for a, b in zip(org[:-1], org[1:]):
                n = max(2, int(np.linalg.norm(b - a) / 0.02))
                pts += [a + (b - a) * k / n for k in range(n + 1)]
        u, v, z = self.project(np.array(pts))
        out = np.zeros((H, W), bool)
        yy, xx = np.mgrid[0:H, 0:W]
        rad = ROBOT_R * self.lens.fx * (W / self.lens.width) / z
        for ui, vi, ri in zip(u, v, rad):
            x0, x1 = int(max(0, ui - ri)), int(min(W, ui + ri + 1))
            y0, y1 = int(max(0, vi - ri)), int(min(H, vi + ri + 1))
            if x0 < x1 and y0 < y1:
                out[y0:y1, x0:x1] |= (xx[y0:y1, x0:x1] - ui) ** 2 + (yy[y0:y1, x0:x1] - vi) ** 2 <= ri * ri
        return out


class DropCounter:
    """Counts parts falling through the column from (x, y, z_low) to (x, y, z_high), ``half`` wide."""

    def __init__(self, x: float, y: float, z_low: float, z_high: float, half: float = 0.015):
        self.cam = TopCamera()
        corners = [(x + dx, y + dy, z) for dx in (-half, half) for dy in (-half, half) for z in (z_low, z_high)]
        u, v, _ = self.cam.project(corners)
        self.u0, self.u1 = int(np.floor(u.min())), int(np.ceil(u.max())) + 1
        self.v0, self.v1 = int(np.floor(v.min())), int(np.ceil(v.max())) + 1
        self.bg = None
        self.count = 0
        self.times: list[float] = []
        self.busy = False
        self.last_t = -1e9
        self.prev_q: dict = {}
        self.moved_at: dict = {}

    def window(self) -> tuple[int, int, int, int]:
        return self.u0, self.v0, self.u1, self.v1

    def look(self, t: float, rgb: np.ndarray, joints: dict) -> bool:
        """Take one top-camera frame at time t; True if a new part was counted."""
        img = as_dream_sees(rgb)[self.v0:self.v1, self.u0:self.u1]
        for side, q in joints.items():
            q = np.asarray(q, float)
            if side not in self.prev_q or np.abs(q - self.prev_q[side]).max() > MOVED:
                self.moved_at[side] = t
            self.prev_q[side] = q.copy()
        moving = {s: q for s, q in joints.items() if t - self.moved_at.get(s, -1e9) <= STILL_S}
        free = np.ones(img.shape[:2], bool)
        if moving:
            outline = self.cam.robot_outline(moving)
            near = outline[max(0, self.v0 - NEAR_PX):self.v1 + NEAR_PX, max(0, self.u0 - NEAR_PX):self.u1 + NEAR_PX]
            if near.any():   # an arm moving next to the window: its shadow may cross it, so don't count now
                self.bg = img.astype(float)
                self.busy = False
                return False
        if self.bg is None:
            self.bg = img.astype(float)
            return False
        changed = (np.abs(img - self.bg).sum(axis=2) > DIFF) & free
        disturbed = int(changed.sum()) >= MIN_PIX
        new = False
        if disturbed and not self.busy and t - self.last_t >= REFRACTORY:
            self.count += 1
            self.times.append(round(t, 2))
            self.last_t = t
            new = True
        self.busy = disturbed
        if not disturbed:   # the air is clear: follow slow changes of the background there
            self.bg[free] = (1 - BG_RATE) * self.bg[free] + BG_RATE * img[free]
            self.bg[~free] = img[~free]
        return new
