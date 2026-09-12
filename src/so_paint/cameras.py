"""Configurable camera views; camera pixels never silently become robot metres."""

import sys
import time
from math import ceil

import cv2
import numpy as np

from .models import CameraConfig


def camera_matrices(c: CameraConfig):
    eye, target, up = np.array(c.eye), np.array(c.target), np.array(c.up)
    forward = target - eye
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError(f"{c.name}: camera eye equals target")
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError(f"{c.name}: up vector is parallel to viewing direction")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = -rotation @ eye
    k = np.array([[c.focal_px, 0, c.width / 2], [0, c.focal_px, c.height / 2], [0, 0, 1]])
    if c.intrinsic_matrix is not None:
        k = np.asarray(c.intrinsic_matrix)
        if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("Invalid camera intrinsic matrix")
    return k, world_to_camera


def project(points, c):
    k, w = camera_matrices(c)
    cam = (w @ np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))].T).T[:, :3]
    pixels = (k @ cam.T).T
    return pixels[:, :2] / pixels[:, 2:3], cam[:, 2]


def render(c, world, arm, q, planned=None):
    img = np.full((c.height, c.width, 3), (47, 53, 60), np.uint8)

    def polygon(points, color):
        px, depth = project(points, c)
        if np.all(depth > 0.01):
            cv2.fillConvexPoly(img, np.rint(px).astype(np.int32), color, cv2.LINE_AA)

    polygon([[0.02, -0.15, 0], [0.36, -0.15, 0], [0.36, 0.15, 0], [0.02, 0.15, 0]], (100, 109, 113))
    px, depth = project(world.corners, c)
    if np.all(depth > 0.01):
        h = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [511, 0], [511, 511], [0, 511]]), px.astype(np.float32)
        )
        canvas = cv2.warpPerspective(world.canvas, h, (c.width, c.height))
        mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), h, (c.width, c.height))
        img[mask > 0] = canvas[mask > 0]
    for s in world.stations:
        theta = np.linspace(0, 2 * np.pi, 36)
        for radius, z, rgb in [
            (s.radius_m + 0.003, s.rim_z, (35, 40, 45)),
            (s.radius_m, s.center[2], s.color_rgb),
        ]:
            points = np.c_[
                s.center[0] + radius * np.cos(theta),
                s.center[1] + radius * np.sin(theta),
                np.full(36, z),
            ]
            polygon(points, rgb)
    if planned is not None:
        pp, depth = project(planned, c)
        if np.all(depth > 0.01):
            cv2.polylines(
                img, [np.rint(pp).astype(np.int32)], False, (70, 240, 200), 1, cv2.LINE_AA
            )
    tip, axis, links = arm.fk(q)
    px, depth = project(links[:-1], c)
    for i in range(len(px) - 1):
        if min(depth[i : i + 2]) > 0.01:
            cv2.line(
                img,
                tuple(np.rint(px[i]).astype(int)),
                tuple(np.rint(px[i + 1]).astype(int)),
                (232, 177, 65),
                5,
                cv2.LINE_AA,
            )
    shaft, depth = project([tip - axis * arm.settings.brush_length_m, tip], c)
    if np.all(depth > 0.01):
        cv2.line(
            img,
            tuple(np.rint(shaft[0]).astype(int)),
            tuple(np.rint(shaft[1]).astype(int)),
            (220, 230, 235),
            2,
            cv2.LINE_AA,
        )
    return img


def open_device(c, warmup=2):
    """Open a camera fresh. USB webcams hand back a stale or black first frame, and a
    device that was just released is not always ready, so warm it up and check."""
    if sys.platform == "darwin":
        from .camera_process import CameraProcess

        device = CameraProcess(c.device, c.width, c.height)
    else:
        device = cv2.VideoCapture(c.device)
        device.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        device.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
    if not device.isOpened():
        device.release()
        raise RuntimeError(f"Camera {c.name} (device {c.device!r}) did not open")
    try:
        for _ in range(warmup):
            device.read()
    except Exception:
        device.release()
        raise
    return device


def read_frame(c, device):
    """One RGB frame from an open device, undistorted if the camera is calibrated."""
    ok, image = device.read()
    if not ok:
        raise RuntimeError(f"Camera {c.name} did not return a frame")
    if c.calibrated and c.intrinsic_matrix is not None:
        image = cv2.undistort(image, np.array(c.intrinsic_matrix), np.array(c.distortion))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def capture(c, world, arm, q, planned=None):
    """A fresh connection every time. Holding a USB camera open across a session is what
    makes the next open fail, so each observation connects, grabs and releases."""
    if c.source == "simulation":
        return render(c, world, arm, q, planned)
    failure = None
    for attempt in range(3):
        if attempt:
            time.sleep(0.3)
        try:
            device = open_device(c)
            try:
                return read_frame(c, device)
            finally:
                device.release()
        except RuntimeError as error:
            failure = error
    raise RuntimeError(f"{failure} after 3 attempts")


def composite(frames):
    tile_w, tile_h, label_h = 640, 480, 42
    columns = min(2, len(frames))
    out = np.full(
        (ceil(len(frames) / columns) * (tile_h + label_h), columns * tile_w, 3), 22, np.uint8
    )
    for i, (name, frame) in enumerate(frames.items()):
        x, y = (i % columns) * tile_w, (i // columns) * (tile_h + label_h)
        h, w = frame.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        scaled = cv2.resize(frame, (round(w * scale), round(h * scale)))
        hh, ww = scaled.shape[:2]
        xx, yy = x + (tile_w - ww) // 2, y + label_h + (tile_h - hh) // 2
        out[yy : yy + hh, xx : xx + ww] = scaled
        cv2.putText(
            out,
            name,
            (x + 16, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 234, 240),
            1,
            cv2.LINE_AA,
        )
    return out
