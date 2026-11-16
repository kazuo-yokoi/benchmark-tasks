"""perceive: where the person's arms and the coaster are on the table, from the top camera's image alone (demo.py).

The image is the one Dream-1 is sent: the top camera's frame scaled to 320 x 176 and JPEG-encoded at quality 85.
Two open models find the person in it (no simulator state, no colour rules):

1. Grounding DINO (IDEA-Research/grounding-dino-base, Apache-2.0), an open-vocabulary detector, boxes
   "a human arm", "a hand", "a person" and "a round coaster";
2. SAM 2.1 (facebook/sam2.1-hiera-small, Apache-2.0) cuts the person's pixels out of each box.

The robot's own arms are left out: their outline in the image is drawn from the robot's own joint angles (the URDF's
forward kinematics, the links as 6.5 cm spheres), and a box mostly inside it, and any pixel inside it, is dropped.

Each person pixel is then carried down its camera ray to the plane 4 cm above the table (where a forearm resting
or reaching over the table is), using the top camera's calibration (rig pose and lens), and drawn into the demo's
map of the table, widened by 3 cm for the error of that height. Only regions at least a hand's size are kept (not
specks), and each map is the union of the last three looks (0.6 s), so that a frame in which the detector misses the
arm does not clear it. Where the robot's own arm hides the table, the map keeps what it last saw there.

The coaster is the box labelled "coaster" whose centre, carried down to the coaster's top, is a coaster's size (6 to
16 cm wide) and lies outside the robot's outline; its centre is where the coaster is, averaged over the last finds while it
has not moved. A box with a person pixel in or at it is not used (a hand on the coaster hides part of it). When no
such box is found (a hand, the cup or the gripper over it), the last position stands.
"""

from __future__ import annotations

import io

import numpy as np

QUERIES = "a human arm. a hand. a person. a round coaster."
COASTER_W = (0.06, 0.16)    # m: how wide a coaster box can be on the table
COASTER_STILL = 0.012       # m: a coaster found closer than this to where it was is taken not to have moved
COASTER_MEAN = 10           # while it has not moved, its place is the mean of the last this many finds
PAD = 2                     # px: a coaster box with a person pixel within this of it is not used
BOX_SCORE, TEXT_SCORE = 0.28, 0.25
ROBOT_R = 0.065              # m, the robot's links drawn as spheres this big (the covers are bulky)
ROBOT_BOX = 0.5              # a box this much inside the robot's outline is the robot
W, H = 320, 176              # Dream-1's frame size
PLANE_DZ = 0.04
WIDEN = 0.03
MIN_CELLS = 150              # map cells (1 cm): a hand, widened by 3 cm, covers about 300
MEMORY = 3                   # looks (0.2 s apart) whose maps are joined
DINO = "IDEA-Research/grounding-dino-base"
SAM = "facebook/sam2.1-hiera-small"


def as_dream_sees(rgb: np.ndarray) -> np.ndarray:
    """The frame as Dream-1's client sends it: 320 x 176, JPEG quality 85."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).convert("RGB").resize((W, H), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


class Segmenter:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_p = AutoProcessor.from_pretrained(DINO)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(self.device).eval()
        self.sam_p = Sam2Processor.from_pretrained(SAM)
        self.sam = Sam2Model.from_pretrained(SAM).to(self.device).eval()

    def boxes(self, img: np.ndarray, exclude: np.ndarray | None = None):
        """[(box x0 y0 x1 y1, score, label)], leaving out boxes mostly inside ``exclude`` (the robot)."""
        torch = self.torch
        inputs = self.det_p(images=img, text=QUERIES, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inputs)
        res = self.det_p.post_process_grounded_object_detection(out, inputs.input_ids, threshold=BOX_SCORE,
                                                              text_threshold=TEXT_SCORE,
                                                              target_sizes=[img.shape[:2]])[0]
        keep = []
        labels = res.get("text_labels", res.get("labels"))
        for box, score, label in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            x0, x1 = np.clip(box[[0, 2]], 0, img.shape[1] - 1)
            y0, y1 = np.clip(box[[1, 3]], 0, img.shape[0] - 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            if exclude is not None:
                inside = exclude[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
                if inside.size and inside.mean() > ROBOT_BOX:
                    continue
            keep.append(([float(x0), float(y0), float(x1), float(y1)], float(score), str(label)))
        return keep

    def mask(self, img: np.ndarray, exclude: np.ndarray | None = None):
        """Boolean (H, W): the person's pixels (none inside ``exclude``), and the boxes labelled coaster."""
        torch = self.torch
        found = self.boxes(img, exclude)
        coasters = [(b, s) for b, s, lab in found if "coaster" in lab]
        boxes = [b for b, _, lab in found if "coaster" not in lab]
        m = np.zeros(img.shape[:2], bool)
        if not boxes:
            return m, coasters
        inputs = self.sam_p(images=img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_p.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        for k in range(masks.shape[0]):
            m |= masks[k, 0].numpy().astype(bool)
        if exclude is not None:
            m &= ~exclude
        return m, coasters


class TopCamera:
    """The top camera's calibration (pose and lens, from the rig) and the segmenter."""

    def __init__(self):
        from yam_arena import station
        from yam_arena.task_file import TABLE_TOP

        cam = station.rig("calibrated").top_cam
        self.pos = np.asarray(cam.pos, float)
        x, y, z, w = cam.rot_xyzw
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                           [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                           [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        self.lens = cam.lens
        self.plane_z = TABLE_TOP["station"] + PLANE_DZ
        self.seg = Segmenter()
        self.last = None
        self.recent = []
        self.fused = None
        self.coaster = None
        self.coaster_seen = []          # where the coaster was found since it last moved (averaged)

    def rays(self, us, vs):
        """World directions of pixels (u, v) of a W x H frame (OpenGL camera: x right, y up, looking along -z)."""
        L = self.lens
        sx, sy = W / L.width, H / L.height
        xc = (us + 0.5 - L.cx * sx) / (L.fx * sx)
        yc = -(vs + 0.5 - L.cy * sy) / (L.fy * sy)
        d = np.stack([xc, yc, -np.ones_like(xc)], axis=-1)
        return d @ self.R.T

    def project(self, pts):
        """Pixels (u, v) and depths of world points (N, 3) in a W x H frame."""
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

    def to_table(self, mask: np.ndarray, z: float | None = None):
        """Table points (x, y) under the masked pixels, on the plane PLANE_DZ above the table (or at height z)."""
        vs, us = np.nonzero(mask)
        if len(us) == 0:
            return np.zeros((0, 2))
        return self.uv_to_table(us.astype(float), vs.astype(float), z)

    def uv_to_table(self, us, vs, z: float | None = None):
        d = self.rays(np.asarray(us, float), np.asarray(vs, float))
        s = ((self.plane_z if z is None else z) - self.pos[2]) / d[:, 2]
        ok = s > 0
        return (self.pos[None, :2] + s[ok, None] * d[ok, :2])

    def person_mask(self, obs, joints: dict) -> np.ndarray:
        """The demo's map (demo.NX x demo.NY cells) of where the person is, from the latest top-camera frame and the
        robot's own joint angles (to leave its arms out)."""
        import demo
        from scipy.ndimage import binary_dilation

        img = as_dream_sees(obs.images["top_cam"])
        robot = self.robot_outline(joints)
        m, coasters = self.seg.mask(img, exclude=robot)
        self._find_coaster(coasters, robot, m)
        self.last = (img, m, robot)
        grid = _cells(self.to_table(m))
        r = int(round(WIDEN / demo.RES))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        if grid.any():
            grid = binary_dilation(grid, structure=(xx ** 2 + yy ** 2) <= r * r)
        if self.fused is not None:          # under the robot's own arm: keep what was seen there last
            hidden = binary_dilation(_cells(self.to_table(robot)), iterations=2)
            grid |= self.fused & hidden
        grid = arm_sized(grid)
        self.recent = (self.recent + [grid])[-MEMORY:]
        self.fused = np.logical_or.reduce(self.recent)
        return self.fused

    def _find_coaster(self, coasters, robot, person) -> None:
        """Update ``self.coaster`` (x, y) from the boxes labelled coaster."""
        import task

        best = None
        for (x0, y0, x1, y1), score in coasters:
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            if robot[int(v), int(u)]:
                continue
            a, b = max(int(x0) - PAD, 0), max(int(y0) - PAD, 0)
            if person[b:int(y1) + PAD + 1, a:int(x1) + PAD + 1].any():
                continue                # a hand on or at the coaster: its box is not the coaster's outline
            ends = self.uv_to_table([x0, x1, u, u], [v, v, y0, y1], task.COASTER_TOP)
            if len(ends) < 4:
                continue
            w = max(np.linalg.norm(ends[0] - ends[1]), np.linalg.norm(ends[2] - ends[3]))
            if not (COASTER_W[0] <= w <= COASTER_W[1]):
                continue
            c = self.uv_to_table([u], [v], task.COASTER_TOP)[0]
            near = 0.0 if self.coaster is None else float(np.linalg.norm(c - self.coaster))
            if best is None or near < best[0]:
                best = (near, c)
        if best is None:
            return
        if self.coaster is None or best[0] > COASTER_STILL:
            self.coaster_seen = []      # it moved: start again from here
        self.coaster_seen = (self.coaster_seen + [best[1]])[-COASTER_MEAN:]
        self.coaster = np.mean(self.coaster_seen, axis=0)


def _cells(pts) -> np.ndarray:
    """The demo's map with the cells under the table points ``pts`` set."""
    import demo

    grid = np.zeros((demo.NX, demo.NY), bool)
    if len(pts):
        i = np.round((pts[:, 0] - demo.X0) / demo.RES).astype(int)
        j = np.round((pts[:, 1] - demo.Y0) / demo.RES).astype(int)
        ok = (i >= 0) & (i < demo.NX) & (j >= 0) & (j < demo.NY)
        grid[i[ok], j[ok]] = True
    return grid


def arm_sized(grid: np.ndarray) -> np.ndarray:
    """Keep the map's regions at least an arm's hand in size (MIN_CELLS, widened), dropping specks the detector finds
    elsewhere."""
    from scipy.ndimage import label

    lab, n = label(grid)
    if n == 0:
        return grid
    sizes = np.bincount(lab.ravel())
    keep = [k for k in range(1, n + 1) if sizes[k] >= MIN_CELLS]
    return np.isin(lab, keep)
