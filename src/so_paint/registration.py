"""Camera-to-robot registration and model-detected landmark reconstruction.

A vision model supplies pixel detections. Metric scale comes from correspondences whose
robot-frame position is known -- in practice the brush tip observed at poses the arm
actually visited, which needs no external target. Never estimate a robot-frame transform
from semantic confidence alone: the fit residual and held-out poses are the evidence.
"""

import cv2
import numpy as np

from .cameras import camera_matrices
from .models import CameraConfig


def calibrate(camera: CameraConfig, robot_points, image_points, max_error_px=2):
    xyz, uv = np.asarray(robot_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2) or len(xyz) < 6:
        raise ValueError("Need at least 6 matched robot XYZ / pixel XY fiducials")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Fiducial coordinates must be finite")
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-7) < 2:
        raise ValueError("Fiducials must span a plane or volume")
    k, _ = camera_matrices(camera)
    distortion = np.asarray(camera.distortion)
    ok, rv, tv = cv2.solvePnP(xyz, uv, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Camera pose could not be solved")
    reproj, _ = cv2.projectPoints(xyz, rv, tv, k, distortion)
    errors = np.linalg.norm(reproj[:, 0, :] - uv, axis=1)
    if errors.max() > max_error_px:
        raise ValueError(f"Reprojection error {errors.max():.2f}px exceeds {max_error_px}px")
    r, _ = cv2.Rodrigues(rv)
    if np.any((r @ xyz.T + tv)[2] <= 0):
        raise ValueError("Calibration places landmarks behind the camera")
    eye = (-r.T @ tv).ravel()
    forward, up = r.T[:, 2], -r.T[:, 1]
    c = camera.model_copy(
        update={
            "eye": tuple(eye),
            "target": tuple(eye + forward),
            "up": tuple(up),
            "calibrated": True,
        }
    )
    return c, {
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "max_px": float(errors.max()),
        "note": "Fit residual only. Verify independent landmarks and brush hover before contact.",
    }


def pixel_to_plane(camera: CameraConfig, pixel, z):
    if camera.source != "simulation" and not camera.calibrated:
        raise ValueError(f"Camera {camera.name} is not registered to the robot base")
    k, w = camera_matrices(camera)
    ray = w[:3, :3].T @ np.linalg.solve(k, np.r_[pixel, 1])
    eye = np.asarray(camera.eye)
    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray is parallel to the work surface")
    distance = (z - eye[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Work surface is behind the camera")
    return eye + distance * ray


def reconstruct(cameras, observations, max_disagreement_m=0.003):
    """Input: name -> {z_m, pixels: {camera_name: [u,v]}}. Pixels are undistorted."""
    by_name = {c.name: c for c in cameras}
    result = {}
    for name, landmark in observations.items():
        positions = [
            pixel_to_plane(by_name[c], p, landmark["z_m"]) for c, p in landmark["pixels"].items()
        ]
        if not positions:
            raise ValueError(f"No camera observations for {name}")
        median = np.median(positions, axis=0)
        disagreement = float(np.max(np.linalg.norm(np.array(positions) - median, axis=1)))
        if disagreement > max_disagreement_m:
            raise ValueError(f"{name}: cameras disagree by {disagreement * 1000:.1f} mm")
        result[name] = {
            "xyz": median.tolist(),
            "max_disagreement_m": disagreement,
            "views": len(positions),
            "depth_source": "configured surface height",
        }
    return result


def validate_camera(camera, robot_points, image_points, max_error_px=2):
    """Evaluate held-out observations without fitting or changing the camera."""
    xyz, uv = np.asarray(robot_points, dtype=float), np.asarray(image_points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or uv.shape != (len(xyz), 2):
        raise ValueError("Need at least 3 held-out robot XYZ / pixel XY pairs")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Held-out coordinates must be finite")
    k, transform = camera_matrices(camera)
    rotation, _ = cv2.Rodrigues(transform[:3, :3])
    predicted, _ = cv2.projectPoints(xyz, rotation, transform[:3, 3],
                                    k, np.asarray(camera.distortion))
    errors = np.linalg.norm(predicted[:, 0] - uv, axis=1)
    in_front = bool(np.all((transform[:3, :3] @ xyz.T + transform[:3, 3:4])[2] > 0))
    return {"passed": bool(in_front and errors.max() <= max_error_px), "max_px": float(errors.max()),
            "rms_px": float(np.sqrt(np.mean(errors ** 2))), "errors_px": errors.tolist(),
            "threshold_px": max_error_px,
            "note": "Independent observations only; vary position, height and wrist orientation. "
                    "A pass validates these observations, not contact depth or all arm poses."}
