"""
Minimal reader for COLMAP *text* models (cameras.txt, images.txt, points3D.txt).

Dependency-free on purpose (numpy only): it is used by the backend, by the Colab depth step
(which has no pycolmap) and by the tests. COLMAP conventions apply throughout:
  * a pose is world -> camera:  X_cam = R * X_world + t   (qvec = (w, x, y, z), tvec = t)
  * camera frame is OpenCV-style: x right, y down, z forward
  * the world frame and its scale are arbitrary (SfM is only defined up to a similarity)
"""
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

# name -> (model_id, number of params) for the models COLMAP can write
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8,
    "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5, "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
}


class Camera(NamedTuple):
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


class Image(NamedTuple):
    image_id: int
    qvec: np.ndarray          # (4,) w, x, y, z  (world -> camera)
    tvec: np.ndarray          # (3,)
    camera_id: int
    name: str
    xys: np.ndarray           # (N, 2) observed keypoints, pixels
    point3d_ids: np.ndarray   # (N,) -1 where the keypoint has no 3D point


class Point3D(NamedTuple):
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track_length: int


class Model(NamedTuple):
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points3d: Dict[int, Point3D]


def _data_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path: str) -> Dict[int, Camera]:
    cameras = {}
    for line in _data_lines(path):
        parts = line.split()
        cid = int(parts[0])
        cameras[cid] = Camera(cid, parts[1], int(parts[2]), int(parts[3]),
                              np.array([float(p) for p in parts[4:]], dtype=np.float64))
    return cameras


def read_images_text(path: str) -> Dict[int, Image]:
    """images.txt alternates a pose line and a keypoint line (which may be empty)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    images = {}
    i = 0
    while i < len(lines):
        if not lines[i].strip():          # blank keypoint line of an image without observations
            i += 1
            continue
        head = lines[i].split()
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        xys = np.array(pts, dtype=np.float64).reshape(-1, 3) if pts else np.zeros((0, 3))
        iid = int(head[0])
        images[iid] = Image(
            iid,
            np.array([float(v) for v in head[1:5]]),
            np.array([float(v) for v in head[5:8]]),
            int(head[8]),
            " ".join(head[9:]),
            xys[:, :2],
            xys[:, 2].astype(np.int64),
        )
        i += 2
    return images


def read_points3d_text(path: str) -> Dict[int, Point3D]:
    points = {}
    for line in _data_lines(path):
        p = line.split()
        pid = int(p[0])
        points[pid] = Point3D(pid, np.array([float(p[1]), float(p[2]), float(p[3])]),
                              np.array([int(p[4]), int(p[5]), int(p[6])], dtype=np.uint8),
                              float(p[7]), (len(p) - 8) // 2)
    return points


def load_model(directory: str) -> Model:
    return Model(
        read_cameras_text(os.path.join(directory, "cameras.txt")),
        read_images_text(os.path.join(directory, "images.txt")),
        read_points3d_text(os.path.join(directory, "points3D.txt")),
    )


# ---------------------------------------------------------------- geometry helpers ----------

def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def camera_center(image: Image) -> np.ndarray:
    """Camera centre in world coordinates: C = -R^T t."""
    return -qvec_to_rotmat(image.qvec).T @ image.tvec


def focal_px(camera: Camera) -> float:
    """Mean focal length in pixels (first param, or mean of fx, fy)."""
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE",
                        "RADIAL_FISHEYE"):
        return float(camera.params[0])
    return float((camera.params[0] + camera.params[1]) / 2.0)


def horizontal_fov_deg(camera: Camera) -> float:
    return float(np.degrees(2 * np.arctan(camera.width / (2 * focal_px(camera)))))


def keyframe_index_from_name(name: str) -> Optional[int]:
    """'keyframe_0007_original_0123.jpg' -> 7 (the package image naming convention)."""
    base = os.path.basename(name)
    if base.startswith("keyframe_"):
        try:
            return int(base.split("_")[1])
        except (IndexError, ValueError):
            return None
    return None


def frame_index_from_name(name: str) -> Optional[int]:
    """'keyframe_0007_original_0123.jpg' -> 123 (the video frame the keyframe came from)."""
    import re
    m = re.search(r"original_(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def sparse_depth_samples(model: Model, image: Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observed keypoints of `image` that have a triangulated 3D point, as
    (uv pixels (M, 2), depth z in the camera frame (M,)); depth is in COLMAP's arbitrary units.
    """
    valid = image.point3d_ids >= 0
    ids = image.point3d_ids[valid]
    uv = image.xys[valid]
    if ids.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    R = qvec_to_rotmat(image.qvec)
    xyz = np.array([model.points3d[i].xyz for i in ids if i in model.points3d])
    keep = np.array([i in model.points3d for i in ids])
    uv = uv[keep]
    z = (xyz @ R.T + image.tvec)[:, 2]
    front = z > 1e-9
    return uv[front], z[front]


def similarity_align(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama: scale s, rotation R, translation t minimising |dst - (s R src + t)|.
    Used to compare an SfM trajectory (arbitrary frame/scale) against a reference trajectory.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t
