"""Two operations, deterministic motion, and a visual feedback boundary.

One session drives either the simulation or a physical SO-101 through LeRobot. Both
share the planner, the geometry checks and the review boundary. What differs is where
the joint state comes from: simulation advances its own state, hardware reports encoder
measurements, and the two are never presented interchangeably.
"""

import copy
import json
import threading
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .brush import Brush
from .cameras import camera_matrices, capture, composite, project
from .kinematics import NAMES, Arm
from .models import Pose, Sample, Settings, Trajectory
from .planner import Planner
from .recording import Recorder
from .world import World

# The URDF limits are the model's, and a real arm's mechanical travel is wider, so a
# measured pose can sit outside them with nothing wrong. That blocks planning, but it is
# the arm's own problem to fix: `recover` nudges it back in. Past this much excursion the
# calibration or the motor-to-URDF mapping is wrong and a person should look at it.
JOINT_LIMIT_SLACK_RAD = 0.02
RECOVERABLE_EXCURSION_RAD = 0.35
RECOVERY_MARGIN_RAD = 0.03
# Recovery runs well under the painting speed, and stays a nudge: an arm in an odd pose
# with the brush possibly touching something is no place for a large joint swing.
RECOVERY_SPEED_FRACTION = 0.2
MAX_RECOVERY_JOINT_DEG = 20.0


class Workbench:
    def __init__(self, settings=None, output_dir=None, record=True, driver=None):
        self.settings = settings or Settings()
        self.backend = self.settings.backend
        self.world = World(self.settings)
        self.arm = Arm(self.settings)
        centre = self.world.paper_xy.mean(axis=0)
        self.q = (
            self.arm.ik([*centre, self.settings.hover_z])
            if self.backend == "simulation"
            else np.zeros(len(NAMES))
        )
        # A hardware session replaces this nominal pose with the first encoder reading.
        # move_to cannot run before look_at, so nothing is ever planned from it.
        self.joints_source = "simulated" if self.backend == "simulation" else "not measured yet"
        self.planner = Planner(self.arm, self.world, self.settings)
        self.brush = Brush()
        self.revision = 0
        self.observed_revision = -1
        self.time_s = 0.0
        self.last_report = None
        self.last_trajectory = None
        self.lock = threading.RLock()
        self.driver = driver
        if self.backend == "lerobot" and self.driver is None:
            from .hardware import LeRobotArm

            # Constructing the driver connects nothing; the port opens on first use.
            self.driver = LeRobotArm(self.settings)
        self.measured = None
        self.out_of_limits = {}
        self.executing = None
        self.cancel = threading.Event()
        # The session clock for hardware pacing and telemetry; replaced only by tests.
        self.now, self.sleep = time.monotonic, time.sleep
        self.output = Path(output_dir or Path("runs") / uuid4().hex[:10]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "settings.json").write_text(self.settings.model_dump_json(indent=2))
        self.recording_path = self.output / "session.rrd"
        self.recorder = Recorder(self.recording_path, self.world, self.settings) if record else None

    # --- hardware state -------------------------------------------------------------

    def _ensure_connected(self):
        if self.driver is None:
            raise RuntimeError(f"Backend {self.backend} has no driver")
        if not self.driver.connected:
            self.driver.connect()

    def _adopt(self, reading):
        """Take encoder feedback as the session's joint state.

        A pose just outside the model's limits is recorded, not raised: the session can
        still look, and `recover` commands the arm back in. A large excursion is refused,
        because that is a wrong mapping rather than a misplaced arm.
        """
        q = np.asarray(reading.joints_rad, dtype=float)
        low, high = self.arm.lower - JOINT_LIMIT_SLACK_RAD, self.arm.upper + JOINT_LIMIT_SLACK_RAD
        outside = {
            name: float(value)
            for name, value, over in zip(NAMES, q, (q < low) | (q > high))
            if over
        }
        excursion = float(np.max(np.maximum(self.arm.lower - q, q - self.arm.upper), initial=0.0))
        if excursion > RECOVERABLE_EXCURSION_RAD:
            raise RuntimeError(
                f"Measured joints are {np.rad2deg(excursion):.0f} degrees outside the URDF "
                f"limits: {outside}. That is too far to be a misplaced arm, so the "
                "motor-to-URDF mapping or the LeRobot calibration is wrong. Run "
                "uv run so-paint calibrate and do not command motion."
            )
        self.measured = reading
        self.out_of_limits = outside
        self.q = np.clip(q, self.arm.lower, self.arm.upper)
        self.joints_source = "measured (outside the model limits)" if outside else "measured"
        return bool(outside)

    def status(self):
        """A snapshot that stays answerable while a hardware batch is running."""
        return {
            "backend": self.backend,
            "revision": self.revision,
            "observed_revision": self.observed_revision,
            "review_required": self.observed_revision != self.revision,
            "output": str(self.output),
            "robot": self.settings.robot.model_dump(),
            "hardware_connected": bool(self.driver is not None and self.driver.connected),
            "joints_source": self.joints_source,
            "out_of_limits_rad": self.out_of_limits,
            "session_time_s": self.time_s,
            "executing": self.executing,
        }

    def request_cancel(self):
        """Ask a running hardware batch to stop in place. Callable during execution."""
        executing = self.executing
        self.cancel.set()
        return {
            "cancelling": executing,
            "note": "The controller stops in place and reports partial execution."
            if executing
            else "Nothing is executing.",
        }

    # --- operations -----------------------------------------------------------------

    def look_at(self):
        if self.executing:
            raise ValueError(
                "A hardware batch is still executing. Use status, or cancel to stop it."
            )
        with self.lock:
            if self.backend == "lerobot":
                self._ensure_connected()
                self._adopt(self.driver.read())
            frames, errors, metadata = {}, {}, {}
            for c in self.settings.observation_cameras():
                try:
                    frames[c.name] = capture(c, self.world, self.arm, self.q)
                except RuntimeError as e:
                    errors[c.name] = str(e)
                    continue
                camera_data = {
                    "source": c.source,
                    "pixel_size": [frames[c.name].shape[1], frames[c.name].shape[0]],
                    "metric_calibration": c.source == "simulation" or c.calibrated,
                }
                if camera_data["metric_calibration"]:
                    k, w = camera_matrices(c)
                    camera_data.update(
                        {
                            "intrinsics": k.tolist(),
                            "robot_to_camera": w.tolist(),
                            "paper_pixels": project(self.world.corners, c)[0].tolist(),
                            "brush_tip_pixel": project([self.arm.fk(self.q)[0]], c)[0][0].tolist(),
                            "station_pixels": {
                                s.name: project([s.center], c)[0][0].tolist()
                                for s in self.world.stations
                            },
                        }
                    )
                metadata[c.name] = camera_data
            if not frames:
                raise RuntimeError(f"No camera returned a frame: {errors}")
            # Stable evidence: each observation owns its full-resolution raw frames.
            observation_dir = self.output / "observations" / uuid4().hex
            observation_dir.mkdir(parents=True)
            for index, (name, frame) in enumerate(frames.items()):
                raw_path = observation_dir / f"camera-{index}.png"
                Image.fromarray(frame).save(raw_path)
                metadata[name]["image"] = str(raw_path)
            combined = composite(frames)
            Image.fromarray(combined).save(observation_dir / "look_at.png")
            overlays = {}
            for camera in self.settings.observation_cameras():
                frame = frames.get(camera.name)
                if frame is None:
                    continue
                calibrated = metadata[camera.name]["metric_calibration"]
                overlays[camera.name] = self._overlay(frame, camera) if calibrated else frame
            Image.fromarray(composite(overlays)).save(observation_dir / "alignment.png")
            simulated = self.backend == "simulation"
            if simulated:
                self._save_image("painting.png", self.world.canvas)
            else:
                # Hardware paint is only observable in the camera views. This raster is the
                # commanded prediction, kept under a name that cannot be read as evidence.
                self._save_image("predicted_painting.png", self.world.expected)
            if self.recorder:
                self.recorder.observation(
                    frames,
                    self.time_s,
                    self.arm,
                    self.q,
                    source="simulated" if simulated else "measured",
                )
            scene_image = (
                self.recorder.screenshot(self.output / "scene.png") if self.recorder else None
            )
            # Partial camera failure does not satisfy a review boundary.
            if not errors:
                self.observed_revision = self.revision
            tip, axis, _ = self.arm.fk(self.q)
            robot = {
                "gripper": "closed",
                "gripper_state_source": "simulation" if simulated else "measured",
                "tip_xyz": tip.tolist(),
                "tip_rpy": self.arm.rotation(self.q).as_euler("xyz").tolist(),
                "brush_axis": axis.tolist(),
                "joints_rad": dict(zip(NAMES, self.q.tolist())),
                "joints_source": self.joints_source,
                "tip_source": "forward kinematics of "
                + ("simulated joints" if simulated else "measured joints, not observed optically"),
            }
            if self.measured is not None:
                robot.update(
                    {
                        "measured_joints_rad": dict(
                            zip(NAMES, np.asarray(self.measured.joints_rad).tolist())
                        ),
                        "measured_gripper_pct": self.measured.gripper_pct,
                        "gripper_closed_reference_pct": self.settings.robot.gripper_closed_pct,
                        "feedback_latency_ms": self.measured.latency_s * 1000,
                        "out_of_limits_rad": self.out_of_limits,
                    }
                )
            if self.out_of_limits:
                robot["blocked"] = (
                    f"These joints sit outside the model's limits: {self.out_of_limits} rad. "
                    "Planning needs a start pose inside them. Call recover to command the arm "
                    "back in yourself -- preview it first, and check the brush is clear of the "
                    "table. Do not ask the user to move the arm by hand."
                )
            result = {
                "backend": self.backend,
                "scene_image": scene_image,
                "scene_image_source": "Rerun headless 3D render" if scene_image else None,
                "renderer_error": self.recorder.render_error if self.recorder else None,
                "revision": self.revision,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "image": str(observation_dir / "look_at.png"),
                "alignment_image": str(observation_dir / "alignment.png"),
                "painting_image": str(self.output / "painting.png") if simulated else None,
                "predicted_painting_image": None
                if simulated
                else str(self.output / "predicted_painting.png"),
                "cameras": metadata,
                "camera_errors": errors,
                "workspace": self.world.metadata(),
                "brush": {
                    "color": self.brush.color,
                    "remaining_m": self.brush.remaining_m,
                    "state_source": "estimated from commanded dwell and stroke length",
                    "estimated_tip_offset": self.settings.brush_tip_offset,
                    "estimated_mount_rpy": self.settings.brush_mount_rpy,
                    "estimated_length_m": self.settings.brush_length_m,
                    "estimated_width_m": self.settings.brush_width_m,
                },
                "robot": robot,
                "last_motion": self.last_report,
                "recording": str(self.recording_path) if self.recorder else None,
                "next": (
                    "Inspect camera/overlay alignment and painting; then move_to with "
                    f"the next <={self.settings.max_batch_duration_s:g}s batch."
                ),
            }
            result["observation_file"] = str(observation_dir / "observation.json")
            (observation_dir / "observation.json").write_text(json.dumps(result, indent=2))
            (self.output / "observation.json").write_text(json.dumps(result, indent=2))
            return result

    def recover(self, *, preview=False, elbow_lift=False):
        """Command the arm back inside the model's joint limits, in joint space.

        Planning needs a start pose the model can represent, so a joint sitting past its
        URDF limit blocks every batch. This is the way out, and it is the agent's to use:
        a slow, minimal nudge of the offending joints back to just inside their limits,
        and a lift to hover height if that leaves the brush low -- because a pose with the
        tip under the table cannot be planned out of either. It reaches for nothing else.
        Preview it and check the reported tip travel first: a brush resting against
        something still drags.
        """
        if self.backend == "simulation":
            raise ValueError("Recovery is a hardware operation; simulated joints stay in range")
        with self.lock:
            if self.executing:
                raise ValueError(f"Batch {self.executing['id']} is still executing")
            self._ensure_connected()
            reading = self.driver.read()
            self._adopt(reading)
            q = np.asarray(reading.joints_rad, dtype=float)
            target = np.clip(
                q, self.arm.lower + RECOVERY_MARGIN_RAD, self.arm.upper - RECOVERY_MARGIN_RAD
            )
            tip_now = self.arm.fk(q)[0]
            low_brush = bool(tip_now[2] < self.settings.hover_z - 0.002)
            if not self.out_of_limits and not low_brush:
                return {
                    "moved": False,
                    "joints_rad": dict(zip(NAMES, q.tolist())),
                    "tip_xyz": tip_now.tolist(),
                    "next": "Already inside the joint limits with the brush clear; look_at.",
                }
            if elbow_lift:
                # Explicit recovery option for a clear upward arc, reviewed in preview.
                target = q.copy()
                target[2] = max(self.arm.lower[2] + RECOVERY_MARGIN_RAD,
                                q[2] - np.deg2rad(MAX_RECOVERY_JOINT_DEG))
                tips = np.array([self.arm.fk(q + (target-q)*f)[0]
                                 for f in np.linspace(0, 1, 41)])
                if np.any(np.diff(tips[:, 2]) < -1e-5) or tips[-1, 2] - tips[0, 2] < .02:
                    raise ValueError("Elbow recovery does not provide a monotonic upward lift")
                lift = "elbow-only upward arc; inspect horizontal travel in preview"
            else:
                target, lift = self._recovery_target(q, target)
            if float(np.abs(target - q).max()) < 1e-4:
                return {
                    "moved": False,
                    "joints_rad": dict(zip(NAMES, q.tolist())),
                    "tip_xyz": tip_now.tolist(),
                    "lift": lift,
                    "next": "Recovery cannot improve this pose on its own. The brush is "
                    "probably resting against something: free it, then recover again.",
                }
            trajectory = self._joint_ramp(q, target)
            tips = np.array([s.tip for s in trajectory.samples])
            report = {
                "id": trajectory.id,
                "backend": self.backend,
                "operation": "recover",
                "preview": preview,
                "moved": not preview,
                "joints_outside_rad": dict(self.out_of_limits),
                "measured_joints_rad": dict(zip(NAMES, q.tolist())),
                "target_joints_rad": dict(zip(NAMES, target.tolist())),
                "largest_correction_deg": float(np.rad2deg(np.abs(target - q).max())),
                "duration_s": trajectory.duration_s,
                "session_start_s": self.time_s,
                "motion_frames": str(self.recorder.execution.frame_index) if self.recorder else None,
                "tip_before_xyz": tips[0].tolist(),
                "tip_after_xyz": tips[-1].tolist(),
                "tip_travel_mm": float(np.linalg.norm(tips[-1] - tips[0]) * 1000),
                "min_tip_z_m": float(tips[:, 2].min()),
                "tip_clear": bool(tips[-1][2] >= self.settings.hover_z - 0.002),
                "lift": lift,
                "limitations": "A joint-space nudge with no IK, no collision checking and no "
                "knowledge of what the brush is resting on.",
            }
            report["backlash_warning"] = self._backlash_warning(
                float(np.rad2deg(np.abs(target - q).max()))
            )
            if preview:
                report["next"] = "Re-run without preview to execute, then look_at."
                return report
            self.cancel.clear()
            self.executing = {
                "id": trajectory.id,
                "planned_duration_s": trajectory.duration_s,
                "started_utc": datetime.now(UTC).isoformat(),
            }
        return self._execute(trajectory, report)

    def _backlash_warning(self, joint_travel_deg):
        """Flag a move small enough for the arm's slack to swallow it."""
        backlash = self.settings.robot.backlash_deg
        if self.backend == "simulation" or joint_travel_deg >= backlash:
            return None
        return (
            f"This move turns no joint more than {joint_travel_deg:.1f} degrees, inside the "
            f"{backlash} degrees of backlash configured for this arm. The arm may not move at "
            "all, or may take up slack in the wrong direction first, and the result will not "
            "tell you anything. Make a decisive move instead of repeating this one."
        )

    def _recovery_target(self, measured, clipped):
        """Where recovery should end: inside the limits, and with the brush off the table.

        Straight up from where the tip already is, because leaving it low only trades one
        unplannable pose for another. Takes the highest lift that still fits inside the
        recovery motion limit, rather than any swing IK happens to return.
        """
        tip = self.arm.fk(clipped)[0]
        if tip[2] >= self.settings.hover_z:
            return clipped, None
        # Probe heights at <=5 mm spacing: a high hover plane must not eliminate
        # small recoverable lifts near a folded wrist. Keep the joint limit unchanged.
        count = max(6, int(np.ceil((self.settings.hover_z - tip[2]) / 0.005)) + 1)
        for height in np.linspace(self.settings.hover_z, tip[2], count)[:-1]:
            try:
                candidate = self.arm.ik(
                    [tip[0], tip[1], height], clipped, self.arm.rotation(clipped)
                )
            except ValueError:
                candidate = None
            if candidate is not None and (
                np.rad2deg(np.abs(candidate - measured).max()) <= MAX_RECOVERY_JOINT_DEG
            ):
                return candidate, f"lifted the brush to z {height:.3f} m"
            # A folded wrist can prevent an orientation-preserving lift. Permit its
            # angle to change while holding pan/roll and the tip's horizontal position.
            cap = np.deg2rad(MAX_RECOVERY_JOINT_DEG)
            lower = np.maximum(self.arm.lower[1:4], measured[1:4] - cap)
            upper = np.minimum(self.arm.upper[1:4], measured[1:4] + cap)
            if np.any(lower >= upper):
                continue

            def joints(values):
                result = clipped.copy()
                result[1:4] = values
                return result

            goal = np.array([tip[0], tip[1], height])
            fit = least_squares(
                lambda v, goal=goal: np.r_[
                    (self.arm.fk(joints(v))[0] - goal) * 100,
                    (v - clipped[1:4]) * .01,
                ],
                np.clip(clipped[1:4], lower, upper), bounds=(lower, upper),
            )
            candidate = joints(fit.x)
            path = np.array([
                self.arm.fk(clipped + (candidate - clipped) * f)[0]
                for f in np.linspace(0, 1, 41)
            ])
            if (
                np.linalg.norm(path[-1] - goal) <= self.settings.position_tolerance_m
                and np.min(np.diff(path[:, 2])) >= -0.00001
                and np.max(np.linalg.norm(path[:, :2] - tip[:2], axis=1)) <= .003
                and np.rad2deg(np.abs(candidate - measured).max()) <= MAX_RECOVERY_JOINT_DEG + 1e-8
            ):
                return candidate, f"lifted the brush to z {height:.3f} m with wrist angle free"
        return clipped, (
            f"left the brush at z {tip[2]:.3f} m: no lift fits within "
            f"{MAX_RECOVERY_JOINT_DEG:.0f} degrees of joint motion. Joint limits are fixed, "
            "but the next plan may still fail on geometry. Free the brush before retrying "
            "rather than forcing a larger move."
        )

    def _joint_ramp(self, start, goal):
        """A rest-to-rest joint interpolation. No IK, no Cartesian path, no reaching."""
        delta = goal - start
        speed = self.settings.max_joint_speed * RECOVERY_SPEED_FRACTION
        duration = max(0.5, 1.875 * float(np.abs(delta).max()) / speed)
        ticks = int(np.ceil(duration * self.settings.sample_hz))
        duration = ticks / self.settings.sample_hz
        samples = []
        for t in np.linspace(0, duration, ticks + 1):
            u = t / duration
            q = start + delta * (10 * u**3 - 15 * u**4 + 6 * u**5)
            samples.append(
                Sample(t=float(t), joints=q.tolist(), tip=tuple(self.arm.fk(q)[0]), phase="travel")
            )
        tip = self.arm.fk(goal)[0]
        roll, pitch, yaw = self.arm.rotation(goal).as_euler("xyz")
        return Trajectory(
            id=uuid4().hex[:12],
            start_revision=self.revision,
            samples=samples,
            duration_s=duration,
            max_tip_error_m=0.0,
            max_orientation_error_deg=0.0,
            orientation_mode="brush_axis",
            waypoints=[Pose(x=tip[0], y=tip[1], z=tip[2], roll=roll, pitch=pitch, yaw=yaw)],
        )

    def move_to(self, poses: list[Pose], *, preview=False, orientation_mode="brush_axis"):
        with self.lock:
            if self.executing:
                raise ValueError(
                    f"Batch {self.executing['id']} is still executing. Use status, or cancel."
                )
            if self.out_of_limits:
                raise ValueError(
                    f"Cannot plan from a start pose outside the model's joint limits: "
                    f"{self.out_of_limits} rad. Call recover to bring the arm back in, then "
                    "look_at. The arm is yours to move; do not hand this to the user."
                )
            if not preview and self.observed_revision != self.revision:
                raise ValueError(
                    "Call look_at and inspect all cameras before the next executed batch"
                )
            trajectory = self.planner.plan(poses, self.q, self.revision, orientation_mode)
            brush = copy.deepcopy(self.brush)
            # Validate ALL brush operations before changing simulated state.
            for a, b in zip(trajectory.samples, trajectory.samples[1:]):
                brush.advance(a, b, self.world)
            self.last_trajectory = trajectory
            path = self.output / f"trajectory-{trajectory.id}.json"
            path.write_text(trajectory.model_dump_json(indent=2))
            report = {
                "id": trajectory.id,
                "backend": self.backend,
                "preview": preview,
                "duration_s": trajectory.duration_s,
                "sample_count": len(trajectory.samples),
                "trajectory": str(path),
                "max_ik_error_mm": trajectory.max_tip_error_m * 1000,
                "max_orientation_error_deg": trajectory.max_orientation_error_deg,
                "orientation_mode": orientation_mode,
                "limitations": "Kinematic validation and coarse tip/rim/table guards; no mesh collision, brush force, or fluid physics.",
            }
            joints = np.array([sample.joints for sample in trajectory.samples])
            report["backlash_warning"] = self._backlash_warning(
                float(np.rad2deg(np.abs(joints - joints[0]).max()))
            )
            if preview:
                if self.recorder:
                    self.recorder.plan(trajectory, self.time_s)
                report["next"] = (
                    "Call look_at to inspect waypoint overlays; submit the same poses with preview=false to execute."
                )
                return report
            if self.recorder:
                self.recorder.plan(trajectory, self.time_s)
            if self.backend == "simulation":
                return self._simulate(trajectory, report)
            self.cancel.clear()
            self.executing = {
                "id": trajectory.id,
                "planned_duration_s": trajectory.duration_s,
                "started_utc": datetime.now(UTC).isoformat(),
            }
        # Physical motion runs outside the session lock so status and cancel stay answerable.
        return self._execute(trajectory, report)

    def _simulate(self, trajectory, report):
        before = self.world.canvas.copy()
        next_camera_t = 0.0
        for a, b in zip(trajectory.samples, trajectory.samples[1:]):
            self.brush.advance(a, b, self.world)
            if b.phase == "paint" and a.phase == "paint":
                rgb = next(s.color_rgb for s in self.world.stations if s.name == self.brush.color)
                self.world.deposit(a.tip, b.tip, expected=True, color=rgb)
                self.world.deposit(a.tip, b.tip, color=rgb)
            if self.recorder:
                timestamp = self.time_s + b.t
                self.recorder.execution.command(self.arm, b.joints, timestamp)
                self.recorder.execution.feedback(
                    self.arm,
                    b.joints,
                    timestamp,
                    source="simulated",
                    target_joints=b.joints,
                    gripper_rad=0,
                )
                if b.t >= next_camera_t or b is trajectory.samples[-1]:
                    # During accelerated simulation, only synthesize camera frames. Real
                    # devices belong to an independently timestamped hardware capture loop.
                    for camera in self.settings.cameras:
                        if camera.source == "simulation":
                            frame = capture(camera, self.world, self.arm, b.joints)
                            self.recorder.execution.camera_frame(
                                camera.name, frame, timestamp, source="simulation"
                            )
                    next_camera_t = b.t + 1 / self.settings.record_camera_hz
        self.q = np.array(trajectory.samples[-1].joints)
        self.revision += 1
        if self.recorder:
            self.recorder.stream.flush()
        self.time_s += trajectory.duration_s
        delta = np.any(self.world.canvas != before, axis=2)
        mismatch = np.mean(
            np.abs(self.world.canvas.astype(float) - self.world.expected.astype(float))
        )
        report.update(
            {
                "revision": self.revision,
                "changed_canvas_pixels": int(delta.sum()),
                "planned_vs_painted_mean_abs_rgb": float(mismatch),
                "feedback_scope": "Approximate synthetic deposition vs commanded strokes, not semantic match to the user's painting.",
                "next": "look_at: compare real positioning, color, stroke footprint and coverage before replanning.",
            }
        )
        self.last_report = report
        self._save_image(f"after-{trajectory.id}.png", self.world.canvas)
        (self.output / f"report-{trajectory.id}.json").write_text(json.dumps(report, indent=2))
        return report

    def _execute(self, trajectory, report):
        """Run a prevalidated batch on the physical arm, then commit measured state."""
        from .hardware import CameraStream, run_trajectory

        execution_log = self.recorder.execution if self.recorder else None
        physical = (
            [c for c in self.settings.cameras if c.source != "simulation"]
            if self.settings.record_cameras_during_motion
            else []
        )
        base, origin = self.time_s, self.now()
        try:
            with CameraStream(
                physical,
                execution_log,
                lambda: base + max(0.0, self.now() - origin),
                self.settings.record_camera_hz,
            ) as stream:
                record, measured = run_trajectory(
                    self.driver,
                    trajectory,
                    self.settings,
                    arm=self.arm,
                    execution_log=execution_log,
                    session_time_s=base,
                    origin=origin,
                    cancel=self.cancel,
                    now=self.now,
                    sleep=self.sleep,
                )
            record["camera_frames_logged"] = stream.frames
            record["camera_errors"] = stream.errors
            if not physical:
                record["camera_logging"] = (
                    "off: record_cameras_during_motion is false, so the cameras stay free for "
                    "look-at. Motion is recorded from encoders only."
                )
            elif stream.stalled:
                record["camera_errors"]["stream"] = (
                    "The motion capture thread did not stop in time and may still hold a "
                    "camera. Set record_cameras_during_motion to false if look-at now fails."
                )
        finally:
            with self.lock:
                self.executing = None
        with self.lock:
            executed = trajectory.samples[: record["commanded_samples"] + 1]
            painted = 0
            for a, b in pairwise(executed):
                self.brush.advance(a, b, self.world)
                if b.phase == "paint" and a.phase == "paint":
                    rgb = next(
                        s.color_rgb for s in self.world.stations if s.name == self.brush.color
                    )
                    # Only the prediction. Real deposition is evidence from the cameras.
                    self.world.deposit(a.tip, b.tip, expected=True, color=rgb)
                    painted += 1
            try:
                self._adopt(measured)
            except RuntimeError as exc:
                # The record of what was executed is kept whatever the feedback says. An
                # implausible measurement is reported here and raised again by look_at,
                # which is what the session needs before anything else moves.
                self.joints_source = "measurement rejected"
                report["measured_state_error"] = str(exc)
            self.revision += 1
            self.time_s += record["elapsed_s"]
            if self.recorder:
                self.recorder.stream.flush()
            report.update(record)
            if report.get("operation") == "recover":
                clear = self.arm.fk(self.q)[0][2] >= self.settings.hover_z - 0.002
                report["next"] = (
                    "look_at, then carry on."
                    if clear and not self.out_of_limits
                    else "Not clear yet: recovery moves at most a nudge at a time. Run recover "
                    "again to continue, and look at the images to see what is in the way."
                )
            report.update(
                {
                    "revision": self.revision,
                    "predicted_paint_segments": painted,
                    "still_out_of_limits_rad": dict(self.out_of_limits),
                    "feedback_scope": "Encoder feedback and the commanded-paint prediction. "
                    "Whether paint reached the paper is only visible in the camera views.",
                    "next": "look_at: compare the measured tip pose and the real paper against "
                    "the prediction. Never resend a partially executed batch without observing.",
                }
            )
            self.last_report = report
            if report.get("operation") != "recover":
                self._save_image(f"predicted-{trajectory.id}.png", self.world.expected)
            (self.output / f"report-{trajectory.id}.json").write_text(json.dumps(report, indent=2))
            return report

    def reconfigure(self, settings):
        """Replace setup estimates between batches, retaining joints, paint and session clock."""
        with self.lock:
            if self.executing:
                raise ValueError("Cannot reload while a batch is executing")
            if settings.backend != self.backend:
                raise ValueError(
                    f"Cannot switch backend from {self.backend} to {settings.backend} in a "
                    "running session; stop the service and start it again"
                )
            if self.backend != "simulation" and settings.robot != self.settings.robot:
                raise ValueError(
                    "Cannot change the robot connection or motor mapping in a running session; "
                    "stop the service, edit the config, and start it again"
                )
            world, arm = World(settings), Arm(settings)
            if np.any(self.q < arm.lower) or np.any(self.q > arm.upper):
                raise ValueError("Current joints are outside the new limits")
            world.classify(arm.fk(self.q)[0])
            if self.brush.color and self.brush.color not in {s.name for s in world.stations}:
                raise ValueError("Cannot remove the currently loaded color; clean the brush first")
            world.canvas = self.world.canvas.copy()
            world.expected = self.world.expected.copy()
            planner = Planner(arm, world, settings)
            path = self.output / f"session-config-{self.revision + 1}.rrd"
            recorder = Recorder(path, world, settings) if self.recorder else None
            old_recorder = self.recorder
            self.settings, self.world, self.arm, self.planner = settings, world, arm, planner
            self.recorder, self.recording_path = recorder, path
            self.revision += 1
            self.observed_revision = -1
            self.last_trajectory = None
            (self.output / "settings.json").write_text(settings.model_dump_json(indent=2))
            (self.output / f"settings-{self.revision}.json").write_text(
                settings.model_dump_json(indent=2)
            )
            if old_recorder:
                old_recorder.close()
            return {"revision": self.revision, "next": "look-at to review the updated setup"}

    def _overlay(self, frame, camera):
        """Draw where the model thinks the paper, stations and last waypoints are.

        These lines are the model's prediction projected into the view. Agreeing with the
        overlay in another view proves nothing if both came from the same wrong transform:
        compare them against the raw image.
        """
        overlay = frame.copy()
        paper = np.rint(project(self.world.corners, camera)[0]).astype(np.int32)
        cv2.polylines(overlay, [paper], True, (70, 240, 200), 1, cv2.LINE_AA)
        text = (cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        for station in self.world.stations:
            px = tuple(np.rint(project([station.center], camera)[0][0]).astype(int))
            cv2.drawMarker(overlay, px, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(overlay, station.name, (px[0] + 7, px[1] - 7), *text)
        if self.last_trajectory:
            points = [p.xyz() for p in self.last_trajectory.waypoints]
            for i, px in enumerate(np.rint(project(points, camera)[0]).astype(int)):
                cv2.circle(overlay, tuple(px), 3, (80, 230, 255), 1)
                cv2.putText(
                    overlay, str(i), tuple(px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 230, 255), 1
                )
        return overlay

    def _save_image(self, name, array):
        Image.fromarray(array).save(self.output / name)

    def close(self):
        self.cancel.set()
        if self.driver is not None:
            self.driver.disconnect()
        if self.recorder:
            self.recorder.close()
