"""Physical SO-101 control through LeRobot.

Importing this module imports nothing from LeRobot and touches no device: `connect()`
is the only thing that opens the serial port. The arm's saved LeRobot calibration is
required, never created here, and never re-recorded automatically -- so-paint prompts
the user to run LeRobot's own interactive calibration instead (see `calibration.py`).

Joint arrays are always the five arm axes in URDF radians. The conversion to LeRobot's
calibrated motor degrees is explicit (`calibration.to_motor_degrees`) and the gripper is
held at its measured closed position, so no code path assumes zero motor units is closed.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import calibration
from .cameras import open_device, read_frame
from .kinematics import NAMES

# The brush geometry depends on where the gripper actually holds it.
GRIPPER_TOLERANCE_PCT = 5.0
# How long to keep watching feedback after the last setpoint before reporting the pose.
SETTLE_TIMEOUT_S = 1.5
# A position servo trails its setpoint while it is moving, by roughly the distance the
# setpoint travels in a few control periods. That lag is not a fault, so the tracking
# limit is a steady-state figure plus an allowance that scales with commanded speed.
FOLLOWING_LAG_S = 0.2
# Backlash makes a joint travel the wrong way on every direction change, so wrong-way
# motion alone means nothing. The difference is that slack runs out: it plateaus at the
# size of the gap while the command keeps going, whereas a sign error tracks the command
# all the way. So look for wrong-way travel that is bigger than the slack allowance AND
# still keeping pace with the command.
INVERSION_TRACKING_RATIO = 0.5


@dataclass(frozen=True)
class Reading:
    """One encoder sample. `sampled_at_s` is a monotonic acquisition-time estimate."""

    joints_rad: np.ndarray
    gripper_pct: float
    sampled_at_s: float
    latency_s: float


class LeRobotArm:
    """SO-101 follower driven through LeRobot. Constructing this connects nothing."""

    backend = "lerobot"

    def __init__(self, settings, *, now=time.monotonic):
        self.settings = settings
        self.robot = settings.robot
        self._now = now
        self._device = None
        self._torque = False
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._device is not None and self._device.is_connected

    def describe(self, limits=None) -> dict:
        report = calibration.describe(self.robot, limits)
        report["connected"] = self.connected
        report["motors_held"] = self.connected and self._torque
        return report

    def connect(self, *, torque=True):
        """Open the port and adopt the saved calibration. Never runs LeRobot's calibration.

        LeRobot's own `connect()` enables torque last, which makes the arm snap to whatever
        goal position its registers still held. The order here is deliberately different:
        torque off, adopt the saved calibration, make the present measured pose the goal,
        then configure and energize -- so connecting holds the arm where it already is.
        Like LeRobot, this assumes the arm is resting when you connect, because torque is
        released first: support it if it is holding a pose.

        `torque=False` is a measurement-only connection. It changes no motor state (except
        adopting the saved calibration, which is required for the units to mean anything)
        and leaves the motors unenergized, so `send` refuses.
        """
        with self._lock:
            if self.connected:
                return
            path = calibration.calibration_file(self.robot)
            if path is None or not path.is_file():
                raise RuntimeError(
                    f"No LeRobot calibration for this arm ({path}). "
                    "Run: uv run so-paint calibrate"
                )
            # Validate the file before energizing anything.
            calibration.load(path)
            try:
                # SO101FollowerConfig is the registered SO-101 config: the plain
                # SOFollowerConfig base has no id or calibration_dir.
                from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            except ImportError as exc:  # pragma: no cover - depends on the install extras
                raise RuntimeError(
                    "LeRobot is not installed. Run: uv sync --extra hardware"
                ) from exc
            device = SO101Follower(
                SO101FollowerConfig(
                    port=self.robot.port,
                    # The calibration file's own location wins, so an explicit
                    # robot.calibration_path is used verbatim.
                    id=path.stem,
                    calibration_dir=path.parent,
                    use_degrees=True,
                    max_relative_target=self.robot.max_relative_target_deg,
                    disable_torque_on_disconnect=self.robot.disable_torque_on_disconnect,
                )
            )
            if not device.calibration:
                raise RuntimeError(f"LeRobot did not load a calibration from {path}")
            device.bus.connect()
            try:
                if not device.bus.is_calibrated:
                    # The motors hold different calibration registers than the saved file.
                    # Write the saved values, as LeRobot's own non-interactive path does,
                    # with torque off first: changing a homing offset shifts the servo
                    # target, which would move an energized arm.
                    device.bus.disable_torque()
                    device.bus.write_calibration(device.calibration)
                    if not device.bus.is_calibrated:
                        raise RuntimeError(
                            "The motors did not accept the saved calibration. Recalibrate: "
                            f"{calibration.commands(self.robot)['calibrate']}"
                        )
                if torque:
                    device.bus.disable_torque()
                    # Hold the present pose instead of a stale goal from an earlier session.
                    device.bus.sync_write(
                        "Goal_Position", device.bus.sync_read("Present_Position")
                    )
                    device.configure()
            except Exception:
                # Leave the motors released: a half-configured arm should not hold torque.
                device.bus.disconnect(disable_torque=True)
                raise
            self._device = device
            self._torque = torque

    def _require(self):
        if not self.connected:
            raise RuntimeError("The arm is not connected")
        return self._device

    def read(self) -> Reading:
        """Read encoder positions. Returns measurements, never a commanded value."""
        device = self._require()
        with self._lock:
            started = self._now()
            observation = device.get_observation()
            finished = self._now()
        degrees = {name: observation[f"{name}.pos"] for name in NAMES}
        return Reading(
            joints_rad=calibration.from_motor_degrees(degrees, self.robot),
            gripper_pct=float(observation["gripper.pos"]),
            sampled_at_s=(started + finished) / 2,
            latency_s=finished - started,
        )

    def send(self, q_rad) -> dict:
        """Command one joint setpoint, holding the gripper at its measured closed position."""
        device = self._require()
        if not self._torque:
            raise RuntimeError("This is a measurement-only connection; the motors are not held")
        if self.robot.gripper_closed_pct is None:
            raise RuntimeError(
                "robot.gripper_closed_pct has not been measured. "
                "Run: uv run so-paint calibrate"
            )
        action = {
            f"{motor}.pos": value
            for motor, value in calibration.to_motor_degrees(q_rad, self.robot).items()
        }
        action["gripper.pos"] = float(self.robot.gripper_closed_pct)
        with self._lock:
            return device.send_action(action)

    def stop_in_place(self) -> Reading:
        """Make the present measured position the goal, so the arm stops where it is."""
        reading = self.read()
        self.send(reading.joints_rad)
        return reading

    def disconnect(self):
        with self._lock:
            device, self._device = self._device, None
            self._torque = False
            if device is not None and device.is_connected:
                # Torque is left as configured: releasing it drops the arm and the brush.
                device.bus.disconnect(
                    disable_torque=self.robot.disable_torque_on_disconnect
                )


def check_start_state(driver, trajectory, robot) -> Reading:
    """Refuse to execute from a pose the arm is not actually in. Sends nothing."""
    reading = driver.read()
    planned = np.array(trajectory.samples[0].joints)
    error_deg = float(np.max(np.abs(np.rad2deg(reading.joints_rad - planned))))
    if error_deg > robot.start_tolerance_deg:
        raise ValueError(
            f"The measured arm pose differs from the planned start by {error_deg:.1f} degrees "
            f"(limit {robot.start_tolerance_deg}). Nothing was sent. Call look-at to resync, "
            "and check whether the arm was moved or lost torque."
        )
    if robot.gripper_closed_pct is not None:
        gripper_error = abs(reading.gripper_pct - robot.gripper_closed_pct)
        if gripper_error > GRIPPER_TOLERANCE_PCT:
            raise ValueError(
                f"The gripper reads {reading.gripper_pct:.1f}, {gripper_error:.1f} from the "
                f"saved closed position {robot.gripper_closed_pct:.1f}. The brush may have "
                "moved. Nothing was sent; re-measure with so-paint calibrate."
            )
    return reading


def run_trajectory(
    driver,
    trajectory,
    settings,
    *,
    arm,
    execution_log=None,
    session_time_s=0.0,
    origin=None,
    cancel=None,
    now=time.monotonic,
    sleep=time.sleep,
):
    """Replay a prevalidated joint trajectory on hardware and report what actually happened.

    The trajectory is already checked by the planner; this only paces the setpoints, logs
    commands and encoder feedback on the session clock, and stops in place on cancellation,
    tracking error, stale feedback or a device fault. Motion is never silently retried.

    Returns the execution record and the final encoder reading. `origin` is the monotonic
    instant that `session_time_s` corresponds to, shared with the camera capture loop.
    """
    robot = settings.robot
    samples = trajectory.samples
    start = check_start_state(driver, trajectory, robot)
    origin = now() if origin is None else origin

    def session_time(monotonic_s):
        return session_time_s + max(0.0, monotonic_s - origin)

    def log_feedback(reading, target):
        if execution_log is None:
            return
        # Physical gripper angle is not measurable from these units; report the units.
        execution_log.feedback(
            arm,
            reading.joints_rad,
            session_time(reading.sampled_at_s),
            source="measured",
            target_joints=target,
        )
        execution_log.gripper(
            reading.gripper_pct, session_time(reading.sampled_at_s), source="measured"
        )

    log_feedback(start, samples[0].joints)
    commanded = 0
    command_times = []
    max_error_deg = 0.0
    max_tip_error_m = 0.0
    reason = None
    inverted = {}
    last = start
    lowest_tip = arm.fk(start.joints_rad)[0]
    lowest_sample = {"tip_xyz": lowest_tip.tolist(), "time_s": session_time(start.sampled_at_s),
                     "joints_rad": start.joints_rad.tolist()}
    previous = samples[0]
    for sample in samples[1:]:
        if cancel is not None and cancel.is_set():
            reason = "cancelled by request"
            break
        delay = origin + sample.t - now()
        if delay > 0:
            sleep(delay)
        try:
            driver.send(sample.joints)
            sent_at = now()
            command_times.append(sent_at)
            if execution_log is not None:
                execution_log.command(arm, sample.joints, session_time(sent_at))
            commanded += 1
            last = driver.read()
        except Exception as exc:  # noqa: BLE001 - any device fault must stop the motion
            reason = f"device fault: {type(exc).__name__}: {exc}"
            break
        log_feedback(last, sample.joints)
        feedback_tip = arm.fk(last.joints_rad)[0]
        if feedback_tip[2] < lowest_tip[2]:
            lowest_tip = feedback_tip
            lowest_sample = {"tip_xyz": feedback_tip.tolist(),
                             "time_s": session_time(last.sampled_at_s),
                             "joints_rad": last.joints_rad.tolist()}
        if last.latency_s > robot.feedback_timeout_s:
            reason = f"stale feedback: {last.latency_s * 1000:.0f} ms round trip"
            break
        error_deg = float(np.max(np.abs(np.rad2deg(last.joints_rad - np.array(sample.joints)))))
        max_error_deg = max(max_error_deg, error_deg)
        max_tip_error_m = max(
            max_tip_error_m,
            float(np.linalg.norm(arm.fk(last.joints_rad)[0] - arm.fk(sample.joints)[0])),
        )
        # A joint going the wrong way is usually backlash taking up, and waiting fixes
        # that. It is a sign fault only once the command is far larger than the slack and
        # the joint is still travelling away from it.
        backlash = np.deg2rad(robot.backlash_deg)
        commanded_delta = np.array(sample.joints) - np.array(samples[0].joints)
        measured_delta = last.joints_rad - start.joints_rad
        inverted = {
            name: {
                "commanded_deg": float(np.rad2deg(commanded_delta[i])),
                "measured_deg": float(np.rad2deg(measured_delta[i])),
            }
            for i, name in enumerate(NAMES)
            if abs(commanded_delta[i]) > backlash
            and abs(measured_delta[i]) > backlash
            and abs(measured_delta[i]) >= INVERSION_TRACKING_RATIO * abs(commanded_delta[i])
            and commanded_delta[i] * measured_delta[i] < 0
        }
        if inverted:
            reason = (
                f"{', '.join(inverted)} kept travelling away from the command, past the "
                f"{robot.backlash_deg} degrees of backlash allowed for. Slack runs out; this "
                "did not. Check the measured direction with so-paint calibrate, set "
                "robot.joint_signs, and do not retry until it is fixed."
            )
            break
        step_deg = float(
            np.max(np.abs(np.rad2deg(np.array(sample.joints) - np.array(previous.joints))))
        )
        allowance = (
            robot.max_tracking_error_deg
            + robot.backlash_deg
            + step_deg / (sample.t - previous.t) * FOLLOWING_LAG_S
        )
        if error_deg > allowance:
            reason = (
                f"tracking error {error_deg:.1f} degrees exceeded {allowance:.1f} "
                f"({robot.max_tracking_error_deg} plus {robot.backlash_deg} of backlash plus "
                f"{FOLLOWING_LAG_S}s of following lag) at t={sample.t:.2f}s. If the "
                "arm was still moving, raise robot.backlash_deg or max_tracking_error_deg "
                "rather than making the move smaller."
            )
            break
        previous = sample
    settled = False
    if reason is None:
        # Watch the arm arrive before reporting a final pose; the last setpoint is not it.
        target = np.array(samples[-1].joints)
        deadline = now() + SETTLE_TIMEOUT_S
        while now() < deadline:
            try:
                last = driver.read()
            except Exception as exc:  # noqa: BLE001 - a fault here is still a fault
                reason = f"device fault while settling: {type(exc).__name__}: {exc}"
                break
            log_feedback(last, samples[-1].joints)
            if float(np.max(np.abs(np.rad2deg(last.joints_rad - target)))) <= 1.0:
                settled = True
                break
            sleep(1 / settings.sample_hz)
    stopped_in_place = False
    if reason is not None:
        try:
            last = driver.stop_in_place()
            stopped_in_place = True
            log_feedback(last, None)
        except Exception as exc:  # noqa: BLE001 - report, never mask, a failed stop
            reason = f"{reason}; stop-in-place also failed: {type(exc).__name__}: {exc}"
    planned = len(samples) - 1
    tip, axis, _ = arm.fk(last.joints_rad)
    record = {
        "backend": "lerobot",
        "completed": reason is None,
        "stop_reason": reason,
        "stopped_in_place": stopped_in_place,
        "settled": settled,
        "commanded_samples": commanded,
        "planned_samples": planned,
        "target_command_hz": settings.sample_hz,
        "measured_command_hz": (
            (len(command_times) - 1) / (command_times[-1] - command_times[0])
            if len(command_times) > 1 and command_times[-1] > command_times[0] else None
        ),
        "max_command_gap_s": (
            float(np.max(np.diff(command_times))) if len(command_times) > 1 else None
        ),
        "executed_fraction": commanded / planned if planned else 1.0,
        "elapsed_s": float(now() - origin),
        "max_tracking_error_deg": max_error_deg,
        "inverted_joints": inverted,
        "max_measured_tip_error_mm": max_tip_error_m * 1000,
        "lowest_encoder_tip": lowest_sample,
        "feedback_source": "motor encoders read through LeRobot",
        "measured_joints_rad": dict(zip(NAMES, last.joints_rad.tolist())),
        "measured_tip_xyz": tip.tolist(),
        "measured_brush_axis": axis.tolist(),
        "measured_gripper_pct": last.gripper_pct,
        "limitations": "Encoder feedback and kinematics only: no force, contact or "
        "collision sensing. The tip pose is derived from joints, not observed optically.",
    }
    return record, last


@dataclass
class CameraStream:
    """Capture physical cameras during motion on their own clock (see docs/TELEMETRY.md).

    Each frame is logged with its own acquisition time, independently of `look_at`, and
    labeled as a physical frame. Capture failures are collected, never raised into motion.
    """

    cameras: list
    execution_log: object
    clock: object
    hz: float
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    errors: dict = field(default_factory=dict)
    frames: int = 0
    stalled: bool = False

    def __enter__(self):
        if self.cameras and self.execution_log is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            # A thread stuck inside a camera read still owns the device, which is what
            # makes the next look_at fail. Say so rather than leaving it a mystery.
            self._thread.join(timeout=10)
            self.stalled = self._thread.is_alive()
            self._thread = None

    def _run(self):
        """Log frames while the arm moves, reopening any camera that drops out.

        A USB camera that fails mid-batch is released immediately rather than held in a
        broken state, because the next look_at has to be able to open it.
        """
        devices = {}
        try:
            while not self._stop.is_set():
                for camera in self.cameras:
                    try:
                        if camera.name not in devices:
                            devices[camera.name] = open_device(camera)
                        frame = read_frame(camera, devices[camera.name])
                        self.execution_log.camera_frame(
                            camera.name, frame, self.clock(), source="physical"
                        )
                        self.frames += 1
                        self.errors.pop(camera.name, None)
                    except Exception as exc:  # noqa: BLE001 - logging must not stop motion
                        self.errors[camera.name] = f"{type(exc).__name__}: {exc}"
                        broken = devices.pop(camera.name, None)
                        if broken is not None:
                            broken.release()
                self._stop.wait(1 / self.hz)
        finally:
            for device in devices.values():
                device.release()
