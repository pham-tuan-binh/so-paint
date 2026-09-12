"""LeRobot calibration handling and physical execution, with no device present."""

import json
import subprocess
import sys

import numpy as np
import pytest

from so_paint import calibration
from so_paint.hardware import Reading, run_trajectory
from so_paint.kinematics import NAMES, Arm
from so_paint.models import RobotConfig, Settings
from so_paint.recipes import at, brush_stroke, load_color
from so_paint.workbench import Workbench

# Body ranges wide enough to cover the URDF limits; LeRobot records wrist_roll full turn.
SPANS = {"shoulder_pan": 2700, "shoulder_lift": 2400, "elbow_flex": 2300, "wrist_flex": 2250}


def calibration_data(**overrides):
    motors = {}
    for i, name in enumerate(calibration.MOTORS):
        span = SPANS.get(name, 4094 if name == "wrist_roll" else 900)
        motors[name] = {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": -37,
            "range_min": calibration.HOMING_RAW - span // 2,
            "range_max": calibration.HOMING_RAW + span // 2,
        }
    for name, patch in overrides.items():
        motors[name].update(patch)
    return motors


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    """Point LeRobot's calibration cache at a temporary, plausible calibration."""

    def write(**overrides):
        path = tmp_path / "calibration" / "robots" / "so_follower" / "painter.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(calibration_data(**overrides), indent=2))
        return path

    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "calibration"))
    return write


@pytest.fixture
def ports(monkeypatch):
    """Control which serial devices appear present, instead of reading the real /dev."""

    def set(*devices):
        monkeypatch.setattr(calibration, "serial_candidates", lambda: list(devices))

    set()
    return set


def robot_config(**overrides):
    return RobotConfig(
        **{"port": "/dev/cu.usbmodem-test", "id": "painter", "gripper_closed_pct": 41.5, **overrides}
    )


def test_calibration_is_found_in_the_lerobot_cache(calibrated):
    path = calibrated()
    robot = robot_config()
    assert calibration.calibration_file(robot) == path
    motors = calibration.load(path)
    assert [m["id"] for m in motors.values()] == [1, 2, 3, 4, 5, 6]
    report = calibration.describe(robot, (Arm(Settings()).lower, Arm(Settings()).upper))
    assert report["calibrated"] and report["warnings"] == []
    assert all(joint["covers_urdf_limit"] for joint in report["joints"].values())
    assert report["gripper"]["closed_pct"] == 41.5
    # An arm with no calibration is reported as such, never assumed usable.
    assert not calibration.describe(robot_config(id="other"))["calibrated"]


def test_calibration_report_flags_short_and_implausible_travel(calibrated):
    arm = Arm(Settings())
    path = calibrated(
        shoulder_pan={"range_min": 1400, "range_max": 2000},
        wrist_flex={"range_min": 20, "range_max": 4080},
    )
    report = calibration.describe(robot_config(), (arm.lower, arm.upper))
    warnings = " ".join(report["warnings"])
    # Travel that stops short of the model's limits, and travel far wider than them.
    assert not report["joints"]["shoulder_pan"]["covers_urdf_limit"]
    assert "shoulder_pan" in warnings and "silently limited" in warnings
    assert "wrist_flex" in warnings and "the URDF range" in warnings
    assert calibration.load(path)["shoulder_pan"]["range_max"] == 2000


def test_a_lopsided_homing_pose_is_a_note_not_a_warning(calibrated):
    """Parking off mid-range shifts nothing: LeRobot centres degrees on the sweep."""
    arm = Arm(Settings())
    calibrated(shoulder_lift={"range_min": 400, "range_max": 3200})
    report = calibration.describe(robot_config(), (arm.lower, arm.upper))
    assert report["joints"]["shoulder_lift"]["covers_urdf_limit"]
    assert report["warnings"] == []
    assert "shoulder_lift" in " ".join(report["notes"])
    assert abs(report["joints"]["shoulder_lift"]["homing_pose_offset_deg"]) > 5


def test_calibration_rejects_implausible_files(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"shoulder_pan": {}}))
    with pytest.raises(TypeError):
        calibration.load(path)
    data = calibration_data()
    data["gripper"]["range_max"] = data["gripper"]["range_min"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="empty recorded range"):
        calibration.load(path)
    data = calibration_data()
    data["elbow_flex"]["id"] = 5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="motor ids"):
        calibration.load(path)


def test_motor_degrees_round_trip_with_signs_and_offsets():
    robot = robot_config(joint_signs=[1, -1, 1, -1, 1], joint_offsets_deg=[0, 3, 0, -2.5, 0])
    q = np.array([0.1, -0.4, 0.3, 0.2, -1.0])
    degrees = calibration.to_motor_degrees(q, robot)
    np.testing.assert_allclose(calibration.from_motor_degrees(degrees, robot), q, atol=1e-12)
    plain = calibration.to_motor_degrees(q, robot_config())
    np.testing.assert_allclose([plain[n] for n in NAMES], np.rad2deg(q), atol=1e-12)


def test_lerobot_backend_requires_a_finished_calibration():
    with pytest.raises(ValueError, match="so-paint calibrate"):
        Settings(backend="lerobot")
    with pytest.raises(ValueError, match="gripper_closed_pct"):
        Settings(backend="lerobot", robot=RobotConfig(port="/dev/x", id="painter"))
    assert Settings(backend="lerobot", robot=robot_config()).backend == "lerobot"


def test_importing_the_driver_does_not_import_lerobot():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, so_paint.hardware, so_paint.workbench;"
                "assert not [m for m in sys.modules if m.startswith('lerobot')]"
            ),
        ],
        check=True,
    )


class Clock:
    """A virtual session clock, so pacing a batch does not take a batch's time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, seconds)


class FakeArm:
    """A stand-in SO-101: records setpoints and echoes them back as encoder feedback."""

    def __init__(
        self,
        clock,
        *,
        gripper=41.5,
        drift=0.0,
        invert=(),
        latency=0.0,
        fault_at=None,
        cancel_after=None,
        wild_after=None,
        lag_ticks=0,
    ):
        self.lag_ticks = lag_ticks
        self.wild_after = wild_after
        self.invert = invert
        self.origin = None
        self.clock = clock
        self.cancel_after = cancel_after
        self.cancel = None
        self.joints = None
        self.gripper = gripper
        self.drift = drift
        self.latency = latency
        self.fault_at = fault_at
        self.commands = []
        self.stops = []
        self.connected = True
        self.disconnected = False

    def connect(self, *, torque=True):
        self.connected = True

    def read(self):
        # With lag, the encoders report a setpoint from a few control periods ago: what a
        # position servo actually does while it is moving.
        joints = self.joints
        if self.lag_ticks and len(self.commands) > self.lag_ticks:
            joints = self.commands[-1 - self.lag_ticks]
        return Reading(
            joints_rad=np.array(joints, dtype=float),
            gripper_pct=self.gripper,
            sampled_at_s=self.clock(),
            latency_s=self.latency,
        )

    def send(self, q_rad):
        if self.fault_at is not None and len(self.commands) >= self.fault_at:
            raise OSError("serial write failed")
        self.commands.append(np.array(q_rad, dtype=float))
        if self.wild_after is not None and len(self.commands) >= self.wild_after:
            # Encoders reporting a pose the model cannot hold: a broken mapping, not a
            # misplaced arm.
            self.joints = np.full(5, 3.0)
            return
        if self.cancel_after is not None and len(self.commands) >= self.cancel_after:
            # Stand in for another client calling cancel while the batch runs.
            self.cancel.set()
        target = np.array(q_rad, dtype=float)
        if self.invert:
            # A sign error maps the commanded travel to the opposite direction, mirrored
            # about wherever the arm started, so the joint tracks -delta exactly.
            origin = self.origin if self.origin is not None else np.array(self.joints, float)
            self.origin = origin
            for i in self.invert:
                target[i] = origin[i] - (target[i] - origin[i])
        self.joints = target + self.drift

    def stop_in_place(self):
        self.stops.append(np.array(self.joints, dtype=float))
        return self.read()

    def disconnect(self):
        self.connected = False
        self.disconnected = True


@pytest.fixture
def hardware(tmp_path, calibrated):
    calibrated()
    settings = Settings(backend="lerobot", robot=robot_config(), rerun_screenshots=False)

    def build(**kwargs):
        clock = Clock()
        driver = FakeArm(clock, **kwargs)
        bench = Workbench(settings, tmp_path / "run", record=False, driver=driver)
        bench.now, bench.sleep = clock, clock.sleep
        driver.cancel = bench.cancel
        driver.joints = bench.arm.ik([*bench.world.paper_xy.mean(axis=0), settings.hover_z])
        return bench, driver

    return build


def test_hardware_look_at_reports_measured_state(hardware):
    bench, driver = hardware()
    driver.joints = bench.q + 0.01
    try:
        observation = bench.look_at()
    finally:
        bench.close()
    assert observation["backend"] == "lerobot"
    assert observation["robot"]["joints_source"] == "measured"
    assert observation["robot"]["gripper_state_source"] == "measured"
    assert observation["robot"]["measured_gripper_pct"] == 41.5
    # A hardware session never presents the synthetic raster as an observation.
    assert observation["painting_image"] is None
    assert observation["predicted_painting_image"].endswith("predicted_painting.png")
    np.testing.assert_allclose(
        [observation["robot"]["joints_rad"][n] for n in NAMES], driver.joints
    )
    assert driver.disconnected


def test_hardware_batch_executes_and_commits_measured_joints(hardware):
    bench, driver = hardware()
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert report["completed"] and report["stop_reason"] is None
        assert report["backend"] == "lerobot"
        assert report["commanded_samples"] == report["planned_samples"]
        assert report["measured_command_hz"] == pytest.approx(50)
        assert report["max_command_gap_s"] <= 1 / 30 + 1e-9
        assert report["feedback_source"] == "motor encoders read through LeRobot"
        assert len(driver.commands) == report["planned_samples"]
        assert bench.revision == 1 and bench.joints_source == "measured"
        assert bench.brush.color == "red"
        bench.look_at()
        report = bench.move_to(brush_stroke(bench.world, [(0.16, 0.01), (0.20, 0.01)]))
        assert report["completed"] and report["predicted_paint_segments"] > 0
        # Real paint is only visible in the cameras: the raster stays a prediction.
        assert np.all(bench.world.canvas == 250)
        assert np.any(bench.world.expected != 250)
        assert bench.observed_revision != bench.revision
        with pytest.raises(ValueError, match="look_at"):
            bench.move_to([at((0.18, 0, 0.028))])
    finally:
        bench.close()


def test_hardware_refuses_to_execute_from_an_unmeasured_pose(hardware):
    bench, driver = hardware()
    try:
        bench.look_at()
        driver.joints = driver.joints + np.deg2rad(9)
        with pytest.raises(ValueError, match="differs from the planned start"):
            bench.move_to(load_color(bench.world, "red"))
        assert driver.commands == [] and bench.revision == 0
        driver.joints = bench.q.copy()
        driver.gripper = 60.0
        with pytest.raises(ValueError, match="brush may have moved"):
            bench.move_to(load_color(bench.world, "red"))
        assert driver.commands == []
        assert bench.executing is None
    finally:
        bench.close()


def test_tracking_error_stops_in_place_and_reports_partial_execution(hardware):
    bench, driver = hardware(drift=np.deg2rad(15))
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"]
        assert "tracking error" in report["stop_reason"]
        assert report["stopped_in_place"] and driver.stops
        assert 0 < report["commanded_samples"] < report["planned_samples"]
        assert 0 < report["executed_fraction"] < 1
        assert "Never resend a partially executed batch" in report["next"]
        # The committed state is what the encoders reported, not the last setpoint.
        np.testing.assert_allclose([bench.q[i] for i in range(5)], driver.joints, atol=1e-9)
    finally:
        bench.close()


def test_implausible_feedback_keeps_the_record_and_stops_the_session(hardware):
    bench, _ = hardware(wild_after=3)
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"]
        assert "outside the URDF limits" in report["measured_state_error"]
        assert bench.joints_source == "measurement rejected"
        assert json.loads((bench.output / f"report-{report['id']}.json").read_text())
        with pytest.raises(RuntimeError, match="mapping"):
            bench.look_at()
    finally:
        bench.close()


def test_the_arm_recovers_itself_from_a_pose_outside_the_model_limits(hardware):
    bench, driver = hardware()
    try:
        bench.look_at()
        # A real arm's travel is wider than the URDF's, so this is a normal situation.
        driver.joints = bench.q.copy()
        driver.joints[2] = bench.arm.upper[2] + 0.04  # elbow past its limit, tip under the table
        observation = bench.look_at()
        assert observation["robot"]["joints_source"] == "measured (outside the model limits)"
        assert "recover" in observation["robot"]["blocked"]
        assert "do not ask the user" in observation["robot"]["blocked"].lower()
        assert "elbow_flex" in observation["robot"]["out_of_limits_rad"]
        # Planning is blocked, and the message points at the arm, not at the user.
        with pytest.raises(ValueError, match="Call recover"):
            bench.move_to(load_color(bench.world, "red"))

        preview = bench.recover(preview=True)
        assert preview["moved"] is False and not driver.commands
        # A nudge, never a swing: recovery stays inside its own motion limit.
        assert 0 < preview["largest_correction_deg"] <= 20
        assert preview["tip_travel_mm"] > 0 and "min_tip_z_m" in preview

        report = bench.recover()
        assert report["completed"] and report["moved"]
        assert report["still_out_of_limits_rad"] == {}
        assert driver.joints[2] <= bench.arm.upper[2]
        assert bench.out_of_limits == {} and bench.joints_source == "measured"
        # Clipping the elbow alone would leave the tip below the table, which is just as
        # unplannable, so recovery lifts it clear as well.
        assert "lifted the brush" in report["lift"]
        # Recovery is a nudge at a time, so it is repeatable until the brush is clear.
        for _ in range(6):
            if report.get("tip_clear"):
                break
            report = bench.recover()
            assert report["moved"]
        assert report["tip_clear"]
        assert bench.arm.fk(bench.q)[0][2] >= bench.settings.hover_z - 0.002
        assert bench.recover()["moved"] is False
        # And the session carries on normally.
        bench.look_at()
        assert bench.move_to(load_color(bench.world, "red"))["completed"]
    finally:
        bench.close()


def test_a_pose_far_outside_the_limits_is_refused_rather_than_nudged(hardware):
    bench, driver = hardware()
    try:
        bench.look_at()
        driver.joints = bench.q.copy()
        driver.joints[2] = bench.arm.upper[2] + 0.6
        with pytest.raises(RuntimeError, match="too far to be a misplaced arm"):
            bench.look_at()
        with pytest.raises(RuntimeError, match="calibrate"):
            bench.recover()
        assert driver.commands == []
    finally:
        bench.close()


def test_device_fault_and_stale_feedback_stop_the_batch(hardware):
    bench, _ = hardware(fault_at=5)
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"] and "device fault" in report["stop_reason"]
        assert report["commanded_samples"] == 5
    finally:
        bench.close()
    bench, _ = hardware(latency=2.0)
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"] and "stale feedback" in report["stop_reason"]
    finally:
        bench.close()


def test_a_joint_running_backwards_is_named_not_just_a_tracking_error(hardware):
    bench, _ = hardware(invert=(1,))
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"]
        assert "shoulder_lift" in report["inverted_joints"]
        assert "kept travelling away from the command" in report["stop_reason"]
        assert "robot.joint_signs" in report["stop_reason"]
        assert report["stopped_in_place"]
    finally:
        bench.close()


def test_following_lag_and_backlash_are_allowed_but_a_standing_error_is_not(hardware):
    # A servo trailing its setpoint by a few control periods keeps going...
    bench, _ = hardware(lag_ticks=3)
    try:
        bench.look_at()
        assert bench.move_to(load_color(bench.world, "red"))["completed"]
    finally:
        bench.close()
    # ...and so does an offset the size of the SO-101's gear slack.
    bench, _ = hardware(drift=np.deg2rad(5))
    try:
        bench.look_at()
        assert bench.move_to(load_color(bench.world, "red"))["completed"]
        assert bench.settings.robot.backlash_deg == 6
    finally:
        bench.close()
    # ...while an error far past every allowance still stops the batch.
    bench, _ = hardware(drift=np.deg2rad(40))
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"] and "tracking error" in report["stop_reason"]
        assert "backlash" in report["stop_reason"] and "following lag" in report["stop_reason"]
    finally:
        bench.close()


def test_motion_logging_leaves_the_look_at_camera_alone(hardware):
    bench, _ = hardware()
    try:
        bench.look_at()
        report = bench.move_to(load_color(bench.world, "red"))
        # One USB camera has one owner, so nothing opens it while the arm moves.
        assert report["camera_frames_logged"] == 0
        assert "stay free for look-at" in report["camera_logging"]
        assert report["camera_errors"] == {}
        assert bench.settings.record_cameras_during_motion is False
    finally:
        bench.close()


def test_a_move_inside_the_backlash_is_flagged_before_it_runs(hardware):
    bench, _ = hardware()
    try:
        observation = bench.look_at()
        tip = observation["robot"]["tip_xyz"]
        # A millimetre nudge: the arm's slack would swallow it whole.
        nudge = bench.move_to([at((tip[0] + 0.001, tip[1], tip[2]))], preview=True)
        assert "backlash" in nudge["backlash_warning"]
        assert "decisive move" in nudge["backlash_warning"]
        # A real move says nothing about backlash.
        assert bench.move_to(load_color(bench.world, "red"), preview=True)["backlash_warning"] is None
    finally:
        bench.close()


def test_cancel_stops_a_batch_in_place(hardware):
    bench, driver = hardware(cancel_after=4)
    try:
        bench.look_at()
        assert bench.status()["executing"] is None
        assert bench.status()["backend"] == "lerobot"
        report = bench.move_to(load_color(bench.world, "red"))
        assert not report["completed"] and report["stop_reason"] == "cancelled by request"
        assert report["commanded_samples"] == 4 and report["stopped_in_place"]
        assert bench.status()["review_required"]
        # A cancellation does not leak into the next batch.
        bench.look_at()
        driver.cancel_after = None
        assert bench.move_to(load_color(bench.world, "red"))["completed"]
    finally:
        bench.close()


def test_run_trajectory_reports_what_it_sent(hardware):
    bench, driver = hardware()
    try:
        bench.look_at()
        trajectory = bench.planner.plan([at((0.18, 0.0, 0.03))], bench.q, 0)
        record, reading = run_trajectory(
            driver,
            trajectory,
            bench.settings,
            arm=bench.arm,
            now=bench.now,
            sleep=bench.sleep,
        )
        assert record["completed"] and record["settled"]
        assert record["max_tracking_error_deg"] == 0
        np.testing.assert_allclose(reading.joints_rad, trajectory.samples[-1].joints, atol=1e-9)
        assert json.dumps(record)
    finally:
        bench.close()


def calibrate_args(**overrides):
    from argparse import Namespace

    defaults = {"port": None, "id": None, "check": False, "recalibrate": False}
    return Namespace(**{**defaults, **overrides})


def test_the_port_and_id_resolve_themselves(tmp_path, calibrated, ports):
    from so_paint.models import RobotConfig

    calibrated()
    blank = RobotConfig()
    # One saved calibration and one connected device need no naming and no --port.
    ports("/dev/cu.usbmodem-only")
    assert calibration.resolve_id(blank) == ("painter", "the only LeRobot calibration on this machine")
    port, why = calibration.resolve_port(blank)
    assert (port, why) == ("/dev/cu.usbmodem-only", "the only serial device present")
    assert calibration.describe(blank)["calibrated"]

    # A configured port that is present wins; a configured port that is gone does not
    # silently stand, and the substitution is stated.
    configured = RobotConfig(port="/dev/cu.usbmodem-only")
    assert calibration.resolve_port(configured)[1] == "configured in the workspace file"
    ports("/dev/cu.usbmodem-renamed")
    port, why = calibration.resolve_port(configured)
    assert port == "/dev/cu.usbmodem-renamed" and "is not connected" in why

    # Two devices, or two calibrated arms, are ambiguities that only the user can settle.
    ports("/dev/cu.usbmodem-a", "/dev/cu.usbmodem-b")
    assert calibration.resolve_port(blank)[0] is None
    assert "lerobot-find-port" in calibration.resolve_port(blank)[1]
    (tmp_path / "calibration" / "robots" / "so_follower" / "second.json").write_text(
        json.dumps(calibration_data())
    )
    identifier, why = calibration.resolve_id(blank)
    assert identifier is None and "painter, second" in why
    assert calibration.calibration_file(blank) is None
    assert calibration.resolve_id(RobotConfig(id="second"))[0] == "second"


def test_calibrate_command_prompts_for_the_missing_steps(tmp_path, monkeypatch, ports):
    from so_paint.cli import run_calibration

    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "empty"))
    result = run_calibration(calibrate_args(), Settings(), tmp_path / "workspace.json", None)
    prompt = " ".join(result["prompt"])
    # Nothing to calibrate on, so it reports instead of launching anything.
    assert result["run"]["started"] is False
    assert "No serial port" in result["run"]["reason"]
    assert not result["calibrated"]
    assert "no serial device is connected" in prompt
    assert "lerobot-find-port" in prompt
    # No arm is calibrated, so the steps name a default id rather than demanding one.
    assert "id 'painter'" in prompt
    assert "lerobot-calibrate --robot.type=so101_follower" in prompt
    assert "move the arm to the middle of its range" in prompt
    assert "Glue the brush" in prompt
    assert result["lerobot"]["installed"] is True


def test_calibrate_runs_lerobot_from_a_terminal_and_reuses_what_exists(
    tmp_path, calibrated, ports, monkeypatch
):
    import subprocess

    from so_paint.cli import run_calibration

    ports("/dev/cu.usbmodem-test")
    monkeypatch.setattr("sys.stdin", type("Tty", (), {"isatty": staticmethod(lambda: True)})())
    launched = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_: launched.append(command) or subprocess.CompletedProcess(command, 0),
    )
    config = tmp_path / "workspace.json"

    # Uncalibrated: calibrating is what the command does, without any extra flag.
    result = run_calibration(calibrate_args(), Settings(), config, None)
    assert result["run"]["started"] is True and result["run"]["exit_code"] == 0
    assert launched[0][:2] == ["uv", "run"] and "lerobot-calibrate" in launched[0]
    assert "--robot.id=painter" in launched[0]
    assert "--robot.port=/dev/cu.usbmodem-test" in launched[0]

    # Already calibrated: reused, never re-recorded, unless asked explicitly.
    calibrated()
    launched.clear()
    reused = run_calibration(calibrate_args(), Settings(robot=robot_config()), config, None)
    assert launched == [] and reused["calibrated"]
    assert "already calibrated" in reused["run"]["reason"]
    assert run_calibration(
        calibrate_args(recalibrate=True), Settings(robot=robot_config()), config, None
    )["run"]["started"]
    assert len(launched) == 1

    # --check looks and touches nothing.
    launched.clear()
    checked = run_calibration(calibrate_args(check=True), Settings(), config, None)
    assert launched == [] and "run" not in checked


def test_calibrate_finishes_the_setup_and_preserves_the_workspace(
    tmp_path, calibrated, ports, monkeypatch
):
    import so_paint.hardware as hardware_module
    from so_paint.cli import run_calibration

    class StubArm:
        """A connected arm, without a device: the read is the only thing exercised."""

        def __init__(self, settings):
            self.settings = settings

        def connect(self, *, torque=True):
            assert torque is False, "measuring the gripper must not energize the motors"

        def read(self):
            return Reading(np.zeros(5), 41.5, 0.0, 0.0)

        def disconnect(self):
            pass

    monkeypatch.setattr(hardware_module, "LeRobotArm", StubArm)
    calibrated()
    ports("/dev/cu.usbmodem-test")
    config = tmp_path / "workspace.json"
    settings = Settings()
    settings.workspace.paper_corners_xy = [(0.15, 0.03), (0.22, 0.03), (0.22, -0.03), (0.15, -0.03)]
    settings.brush_tip_offset = (0.11, 0.0, -0.09)
    config.write_text(settings.model_dump_json(indent=2))

    # One command takes an already-calibrated arm the rest of the way: it measures the
    # closed gripper itself and saves what it resolved.
    result = run_calibration(calibrate_args(), settings, config, None)
    assert result["measured"]["gripper_pct"] == 41.5
    assert result["config_written"]["backend"] == "lerobot"
    assert result["config_written"]["still_needed"] == []
    saved = Settings.model_validate_json(config.read_text())
    assert saved.backend == "lerobot" and saved.robot.gripper_closed_pct == 41.5
    assert saved.robot.id == "painter" and saved.robot.port == "/dev/cu.usbmodem-test"
    # Geometry the agent had already estimated survives untouched.
    assert saved.brush_tip_offset == (0.11, 0.0, -0.09)
    np.testing.assert_allclose(saved.workspace.paper_corners_xy, settings.workspace.paper_corners_xy)

    # --check writes nothing.
    config.write_text(settings.model_dump_json(indent=2))
    checked = run_calibration(calibrate_args(check=True), settings, config, None)
    assert "config_written" not in checked and "measured" not in checked
    assert Settings.model_validate_json(config.read_text()).backend == "simulation"


def test_an_incomplete_setup_does_not_claim_the_hardware_backend(tmp_path, calibrated, ports):
    from so_paint.cli import run_calibration

    calibrated()
    ports()  # nothing connected, so there is no port to save
    config = tmp_path / "workspace.json"
    config.write_text(Settings().model_dump_json(indent=2))
    written = run_calibration(calibrate_args(), Settings(), config, None)["config_written"]
    assert written["backend"] == "simulation"
    assert "robot.port" in written["still_needed"]
    assert Settings.model_validate_json(config.read_text()).backend == "simulation"


def test_backend_and_robot_changes_are_rejected_mid_session(hardware):
    bench, _ = hardware()
    try:
        bench.look_at()
        with pytest.raises(ValueError, match="Cannot switch backend"):
            bench.reconfigure(Settings(rerun_screenshots=False))
        moved = Settings(
            backend="lerobot",
            robot=robot_config(port="/dev/cu.usbmodem-other"),
            rerun_screenshots=False,
        )
        with pytest.raises(ValueError, match="robot connection"):
            bench.reconfigure(moved)
    finally:
        bench.close()


def test_elbow_recovery_preview_is_upward_and_bounded(hardware):
    from so_paint.kinematics import Arm
    from so_paint.models import Settings

    bench, driver = hardware()
    try:
        bench.look_at()
        bench.arm = Arm(Settings(brush_tip_offset=(.08397, .01303, .01451)))
        driver.joints = np.array([.02608, -1.30190, 1.46761, .40814, .03452])
        report = bench.recover(preview=True, elbow_lift=True)
        assert not driver.commands
        assert report['largest_correction_deg'] <= 20 + 1e-8
        assert report['tip_after_xyz'][2] > report['tip_before_xyz'][2] + .02
        assert report['min_tip_z_m'] >= report['tip_before_xyz'][2] - 1e-8
    finally:
        bench.close()
