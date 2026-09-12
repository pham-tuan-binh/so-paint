"""Preflight whole batches, solve IK continuously and time-scale to motion limits."""

from uuid import uuid4

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .models import Pose, Sample, Trajectory


def continuous_waypoints(poses):
    """Coalesce straight, same-orientation runs without rounding corners or holds.

    The first waypoint remains an explicit approach. A removed waypoint lies exactly
    on the retained segment, so all original collision and pose checks still apply.
    """
    result = []
    for pose in poses:
        while len(result) >= 2:
            a, b = result[-2:]
            ab = np.array(b.xyz()) - a.xyz()
            bc = np.array(pose.xyz()) - b.xyz()
            lengths = np.linalg.norm(ab) * np.linalg.norm(bc)
            if (
                b.hold_s > 0
                or lengths < 1e-12
                or np.dot(ab, bc) <= 0
                or np.linalg.norm(np.cross(ab, bc)) > lengths * 1e-7
                or not np.allclose(a.rpy(), b.rpy(), atol=1e-9, rtol=0)
                or not np.allclose(b.rpy(), pose.rpy(), atol=1e-9, rtol=0)
            ):
                break
            result.pop()
        result.append(pose)
    return result


class Planner:
    def __init__(self, arm, world, settings):
        self.arm, self.world, self.s = arm, world, settings

    def plan(self, poses: list[Pose], start_q, revision, mode="brush_axis"):
        if not 1 <= len(poses) <= 128:
            raise ValueError("Provide 1 to 128 brush-tip poses per batch")
        if mode not in ("full", "brush_axis"):
            raise ValueError("orientation_mode must be full or brush_axis")
        q = np.array(start_q, dtype=float)
        samples = []
        elapsed, max_error, max_angle = 0.0, 0.0, 0.0
        previous_tip = self.arm.fk(q)[0]
        phase, station = self.world.classify(previous_tip)
        samples.append(
            Sample(t=0, joints=q.tolist(), tip=tuple(previous_tip), phase=phase, station=station)
        )
        for pose in continuous_waypoints(poses):
            start = self.arm.fk(q)[0]
            goal = np.array(pose.xyz())
            r0, r1 = self.arm.rotation(q), Rotation.from_euler("xyz", pose.rpy())
            # In axis mode preserve current roll about the brush to avoid arbitrary spinning.
            if mode == "brush_axis":
                old = r0.apply(self.s.brush_axis)
                new = r1.apply(self.s.brush_axis)
                delta, _ = Rotation.align_vectors([new], [old])
                r1 = delta * r0
            slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
            distance = float(np.linalg.norm(goal - start))
            u = np.linspace(0, 1, max(12, int(distance / 0.002) + 1))
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            qs = [q.copy()]
            for b in blend[1:]:
                q = self.arm.ik(start + (goal - start) * b, q, slerp(float(b)), mode)
                self.world.classify(self.arm.fk(q)[0])
                qs.append(q.copy())
            spline = CubicSpline(u, qs, bc_type=((1, np.zeros(5)), (1, np.zeros(5))))
            dense_u = np.linspace(0, 1, max(100, len(u) * 3))
            joint_v = float(np.abs(spline(dense_u, 1)).max())
            joint_a = float(np.abs(spline(dense_u, 2)).max())
            # Conservative time scaling; each retained corner or dwell is a rest-to-rest move.
            contact = min(start[2], goal[2]) < self.s.hover_z
            speed = self.s.paint_speed if contact else self.s.max_tip_speed
            duration = (
                max(
                    0.2,
                    1.875 * distance / speed,
                    joint_v / self.s.max_joint_speed,
                    np.sqrt(joint_a / self.s.max_joint_accel),
                )
                * 1.03
            )
            ticks = int(np.ceil(duration * self.s.sample_hz))
            duration = ticks / self.s.sample_hz
            if elapsed + duration + pose.hold_s > self.s.max_batch_duration_s + 1e-8:
                raise ValueError(
                    f"Batch needs {elapsed + duration + pose.hold_s:.1f}s, "
                    f"exceeding {self.s.max_batch_duration_s:g}s. "
                    "Split it at a lifted brush pose; no motion was executed."
                )
            for t in np.linspace(0, duration, ticks + 1)[1:]:
                frac = t / duration
                qt = spline(frac)
                if np.any(qt < self.arm.lower) or np.any(qt > self.arm.upper):
                    raise ValueError("Interpolated trajectory exceeds a joint limit")
                tip, direction, links = self.arm.fk(qt)
                b = 10 * frac**3 - 15 * frac**4 + 6 * frac**5
                error = float(np.linalg.norm(tip - (start + (goal - start) * b)))
                desired = slerp(float(b))
                angle = (
                    np.rad2deg((desired.inv() * self.arm.rotation(qt)).magnitude())
                    if mode == "full"
                    else np.rad2deg(
                        np.arccos(
                            np.clip(
                                np.dot(direction, desired.apply(self.s.brush_axis))
                                / np.linalg.norm(self.s.brush_axis),
                                -1,
                                1,
                            )
                        )
                    )
                )
                if error > self.s.position_tolerance_m or angle > self.s.brush_tilt_tolerance_deg:
                    raise ValueError(
                        "Interpolated pose exceeds accuracy tolerance; use closer waypoints"
                    )
                # Coarse table guard on moving joint centres; this is not mesh collision checking.
                if np.min(links[2:-1, 2]) < self.s.table_z + 0.003:
                    raise ValueError("A moving arm joint is too close to the table")
                phase, station = self.world.classify(tip)
                # Low lateral travel through a station's wall is forbidden even between samples.
                self._check_swept(samples[-1].tip, tip)
                samples.append(
                    Sample(
                        t=elapsed + float(t),
                        joints=qt.tolist(),
                        tip=tuple(tip),
                        phase=phase,
                        station=station,
                    )
                )
                max_error, max_angle = max(max_error, error), max(max_angle, float(angle))
            elapsed += duration
            for _ in range(int(np.ceil(pose.hold_s * self.s.sample_hz))):
                elapsed += 1 / self.s.sample_hz
                samples.append(samples[-1].model_copy(update={"t": elapsed}))
            q = np.array(samples[-1].joints)
        if elapsed > self.s.max_batch_duration_s + 1e-8:
            raise ValueError(
                f"Rounded batch duration exceeds {self.s.max_batch_duration_s:g}s"
            )
        # Keep the review boundary above contact. No accidental extra paint during inspection.
        if samples[-1].tip[2] < self.s.hover_z - 0.0015:
            raise ValueError(f"End the batch with the brush lifted to z >= {self.s.hover_z} m")
        times = np.array([s.t for s in samples])
        velocities = (
            np.diff(np.array([s.joints for s in samples]), axis=0) / np.diff(times)[:, None]
        )
        if np.abs(velocities).max() > self.s.max_joint_speed * 1.01:
            raise ValueError("Trajectory exceeds joint speed limit")
        if len(velocities) > 1:
            accel = (
                np.diff(velocities, axis=0)
                / ((np.diff(times)[1:] + np.diff(times)[:-1]) / 2)[:, None]
            )
            if np.abs(accel).max() > self.s.max_joint_accel * 1.03:
                raise ValueError("Trajectory exceeds joint acceleration limit")
        return Trajectory(
            id=uuid4().hex[:12],
            start_revision=revision,
            samples=samples,
            duration_s=elapsed,
            max_tip_error_m=max_error,
            max_orientation_error_deg=max_angle,
            orientation_mode=mode,
            waypoints=poses,
        )

    def _check_swept(self, a, b):
        a, b = np.array(a), np.array(b)
        for f in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.0005) + 1)):
            self.world.classify(a + (b - a) * f)
