"""URDF forward kinematics and continuity-seeded constrained brush-tip IK.

The SO-101 has five arm axes. We constrain position and brush direction, leaving
rotation around the brush free. Gripper opening is not a sixth positioning axis.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .models import Settings

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


class Arm:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mount = Rotation.from_euler("xyz", settings.brush_mount_rpy)
        root = ET.parse(Path(__file__).parent / "assets/so101.urdf").getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        self.chain = []
        lo, hi = [], []
        for name in NAMES + ["gripper_frame_joint"]:
            j = joints[name]
            o = j.find("origin")
            origin = transform(
                np.fromstring(o.attrib["xyz"], sep=" "), np.fromstring(o.attrib["rpy"], sep=" ")
            )
            axis = np.fromstring(j.find("axis").attrib["xyz"], sep=" ")
            self.chain.append((origin, axis))
            if j.attrib["type"] != "fixed":
                lo.append(float(j.find("limit").attrib["lower"]))
                hi.append(float(j.find("limit").attrib["upper"]))
        self.lower, self.upper = np.array(lo), np.array(hi)

    def fk(self, q):
        t = np.eye(4)
        links = [t[:3, 3].copy()]
        for i, (origin, axis) in enumerate(self.chain):
            t = t @ origin
            if i < 5:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
                t = t @ r
            links.append(t[:3, 3].copy())
        tip = (t @ np.r_[self.settings.brush_tip_offset, 1])[:3]
        direction = t[:3, :3] @ self.mount.apply(self.settings.brush_axis)
        direction /= np.linalg.norm(direction)
        return tip, direction, np.array(links + [tip])

    def rotation(self, q):
        t = np.eye(4)
        for i, (origin, axis) in enumerate(self.chain):
            t = t @ origin
            if i < 5:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
                t = t @ r
        return Rotation.from_matrix(t[:3, :3]) * self.mount

    def ik(self, target, seed=None, rotation=None, mode="brush_axis"):
        target = np.asarray(target)
        seed = np.zeros(5) if seed is None else np.asarray(seed)

        rotation = Rotation.from_euler("xyz", [np.pi, 0, 0]) if rotation is None else rotation
        desired_axis = rotation.apply(self.settings.brush_axis)
        desired_axis /= np.linalg.norm(desired_axis)

        def residual(q):
            tip, direction, _ = self.fk(q)
            orientation = (
                (rotation.inv() * self.rotation(q)).as_rotvec()
                if mode == "full"
                else direction - desired_axis
            )
            return np.r_[(tip - target) * 20, orientation * 0.4, (q - seed) * 0.0001]

        candidates = [seed]
        if seed is None or np.allclose(seed, 0):
            candidates += [np.array([0, -0.8, 0.8, 0.8, 0]), np.array([0, 0.8, -0.8, -0.8, 0])]
        best = None
        for candidate in candidates:
            sol = least_squares(
                residual,
                np.clip(candidate, self.lower, self.upper),
                bounds=(self.lower, self.upper),
                max_nfev=100,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            if best is None or np.linalg.norm(residual(sol.x)) < best[0]:
                best = (np.linalg.norm(residual(sol.x)), sol.x)
        q = best[1]
        tip, direction, _ = self.fk(q)
        error = float(np.linalg.norm(tip - target))
        tilt = float(
            np.rad2deg((rotation.inv() * self.rotation(q)).magnitude())
            if mode == "full"
            else np.rad2deg(np.arccos(np.clip(np.dot(direction, desired_axis), -1, 1)))
        )
        if (
            error > self.settings.position_tolerance_m
            or tilt > self.settings.brush_tilt_tolerance_deg
        ):
            raise ValueError(
                f"Unreachable brush pose {target.tolist()}: "
                f"position error {error * 1000:.2f} mm, tilt {tilt:.1f} degrees"
            )
        return q
