"""LeRobot motor calibration: discovery, parsing and the user-facing calibration prompt.

Everything here is file and arithmetic work. Reading a calibration never opens the
serial port, energizes a motor or imports LeRobot. `so-paint calibrate` renders this
into the steps the user runs with LeRobot's own tools; so-paint never writes the
calibration file itself, so an already-calibrated arm keeps its saved calibration.
"""

import json
import os
from pathlib import Path

import numpy as np

from .kinematics import NAMES

# LeRobot's CLI selects the SO-101 with --robot.type=so101_follower, but the follower
# class is named so_follower, and that class name is the calibration subdirectory.
# Older releases filed it under so101_follower, so both are searched.
ROBOT_TYPE = "so101_follower"
CALIBRATION_DIRS = ("so_follower", "so101_follower")
MOTORS = (*NAMES, "gripper")
# STS3215 encoders. LeRobot normalizes against resolution - 1 and homes at a half turn.
ENCODER_RESOLUTION = 4096
MAX_RAW = ENCODER_RESOLUTION - 1
HOMING_RAW = MAX_RAW // 2
FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
# Recorded travel this much wider than the URDF limits suggests the sweep was not
# measuring the joint it should have been.
IMPLAUSIBLE_TRAVEL_RATIO = 1.4
# The id LeRobot files a calibration under. It only has to distinguish one arm from
# another, so a single-arm setup never needs to choose one.
DEFAULT_ID = "painter"


def cache_root() -> Path:
    """LeRobot's calibration root, honouring the same environment overrides LeRobot reads."""
    if os.environ.get("HF_LEROBOT_CALIBRATION"):
        root = Path(os.environ["HF_LEROBOT_CALIBRATION"])
    elif os.environ.get("HF_LEROBOT_HOME"):
        root = Path(os.environ["HF_LEROBOT_HOME"]) / "calibration"
    elif os.environ.get("HF_HOME"):
        root = Path(os.environ["HF_HOME"]) / "lerobot" / "calibration"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        root = cache / "huggingface" / "lerobot" / "calibration"
    return root.expanduser() / "robots"


def cache_dir() -> Path:
    """Where a new SO-101 calibration is written by the installed LeRobot."""
    return cache_root() / CALIBRATION_DIRS[0]


def serial_candidates() -> list[str]:
    """Serial devices present now. Listing them never opens or energizes a port."""
    return sorted(
        {
            str(path)
            for pattern in ["cu.usb*", "ttyUSB*", "ttyACM*"]
            for path in Path("/dev").glob(pattern)
        }
    )


def resolve_port(robot) -> tuple[str | None, str]:
    """Which serial port to use, and why. Saved config first, then the obvious device.

    A configured port that is actually present always wins. If it is gone -- macOS
    renames these across reboots -- a single present device is adopted and said so.
    Several present devices are a real ambiguity: an SO-101 pair, or a board that
    exposes two interfaces. Writing to the wrong one is worse than asking, so it asks.
    """
    candidates = serial_candidates()
    if robot.port and robot.port in candidates:
        return robot.port, "configured in the workspace file"
    if len(candidates) == 1:
        if robot.port:
            return candidates[0], (
                f"the only serial device present; the configured {robot.port} is not connected"
            )
        return candidates[0], "the only serial device present"
    if not candidates:
        if robot.port:
            return robot.port, f"configured, but {robot.port} is not connected right now"
        return None, "no serial device is connected"
    return None, (
        f"{len(candidates)} serial devices are present ({', '.join(candidates)}); "
        "run lerobot-find-port or pass --port"
    )


def available_ids() -> dict[str, Path]:
    """Every SO-101 calibration LeRobot already has on this machine, by arm id."""
    found: dict[str, Path] = {}
    for name in CALIBRATION_DIRS:
        directory = cache_root() / name
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                found.setdefault(path.stem, path)
    return found


def resolve_id(robot) -> tuple[str | None, str]:
    """Which arm to use, and why. A single saved calibration needs no naming at all.

    An explicit id wins. Otherwise one existing calibration is adopted; several are a
    real ambiguity that only the user can settle, and none means a fresh arm.
    """
    if robot.id:
        return robot.id, "configured"
    found = available_ids()
    if len(found) == 1:
        return next(iter(found)), "the only LeRobot calibration on this machine"
    if found:
        return None, (
            f"several arms are calibrated ({', '.join(sorted(found))}); "
            "choose one with --id"
        )
    return None, "no calibration on this machine yet"


def calibration_file(robot) -> Path | None:
    """Where this arm's LeRobot calibration lives, or None if no single arm is implied."""
    if robot.calibration_path:
        return Path(robot.calibration_path).expanduser()
    identifier, _ = resolve_id(robot)
    if identifier is None:
        return None
    candidates = [cache_root() / name / f"{identifier}.json" for name in CALIBRATION_DIRS]
    return next((path for path in candidates if path.is_file()), candidates[0])


def load(path) -> dict[str, dict]:
    """Parse a LeRobot calibration file, rejecting anything an SO-101 cannot have written.

    Raises ValueError for implausible values and TypeError for a wrong JSON shape.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != set(MOTORS):
        raise TypeError(f"{path}: expected exactly the SO-101 motors {list(MOTORS)}")
    motors = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or not set(FIELDS) <= set(entry):
            raise TypeError(f"{path}: motor '{name}' is missing {FIELDS}")
        values = {}
        for field in FIELDS:
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{path}: {name}.{field} must be an integer encoder value")
            values[field] = value
        if values["range_max"] - values["range_min"] < 2:
            raise ValueError(f"{path}: {name} has an empty recorded range of motion")
        if not 0 <= values["range_min"] < values["range_max"] <= MAX_RAW:
            raise ValueError(f"{path}: {name} range is outside 0..{MAX_RAW} encoder counts")
        if values["drive_mode"] not in (0, 1):
            raise ValueError(f"{path}: {name}.drive_mode must be 0 or 1")
        motors[name] = values
    ids = [m["id"] for m in motors.values()]
    if sorted(ids) != list(range(1, len(MOTORS) + 1)):
        raise ValueError(f"{path}: motor ids must be 1..{len(MOTORS)}, got {ids}")
    if any(motors[name]["id"] != i + 1 for i, name in enumerate(MOTORS)):
        raise ValueError(f"{path}: motor ids must follow the SO-101 order {list(MOTORS)}")
    return motors


def to_motor_degrees(q_rad, robot) -> dict[str, float]:
    """URDF radians for the five arm joints to LeRobot's calibrated motor degrees."""
    q = np.asarray(q_rad, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all():
        raise ValueError("Expected five finite joint positions in URDF radians")
    signs = np.array(robot.joint_signs, dtype=float)
    offsets = np.array(robot.joint_offsets_deg, dtype=float)
    return dict(zip(NAMES, ((np.rad2deg(q) - offsets) * signs).tolist()))


def from_motor_degrees(degrees, robot) -> np.ndarray:
    """LeRobot's calibrated motor degrees to URDF radians, in `kinematics.NAMES` order."""
    values = np.array([float(degrees[name]) for name in NAMES])
    if not np.isfinite(values).all():
        raise ValueError("Motor feedback contained a non-finite position")
    signs = np.array(robot.joint_signs, dtype=float)
    offsets = np.array(robot.joint_offsets_deg, dtype=float)
    return np.deg2rad(values * signs + offsets)


def _recorded_degrees(motor) -> tuple[float, float]:
    """Range endpoints in LeRobot degrees: it centres degrees on the recorded range."""
    half = (motor["range_max"] - motor["range_min"]) / 2 * 360 / MAX_RAW
    return -half, half


def describe(robot, limits=None) -> dict:
    """Report calibration state and the motor-to-URDF mapping it implies. Never connects.

    `limits` is the URDF (lower, upper) joint limit pair from `kinematics.Arm`. Supplying
    it adds the comparison between each motor's recorded travel and the model's limits.
    """
    path = calibration_file(robot)
    identifier, why = resolve_id(robot)
    port, port_why = resolve_port(robot)
    report = {
        "robot_type": ROBOT_TYPE,
        "robot_id": identifier,
        "robot_id_source": why,
        "available_ids": sorted(available_ids()),
        "port": port,
        "port_source": port_why,
        "serial_candidates": serial_candidates(),
        "calibration_file": str(path) if path else None,
        "calibration_source": "workspace robot.calibration_path"
        if robot.calibration_path
        else "lerobot calibration cache",
        "calibrated": False,
        "mapping": {
            "convention": "calibrated LeRobot motor degrees are URDF joint degrees",
            "joint_signs": list(robot.joint_signs),
            "joint_offsets_deg": list(robot.joint_offsets_deg),
            "verified": "declared in workspace.json; confirm with hover checks before contact",
        },
        "gripper_closed_pct": robot.gripper_closed_pct,
        "warnings": [],
        "notes": [],
        "joints": {},
    }
    if path is None:
        report["error"] = f"No calibration to read: {why}"
        return report
    if not path.is_file():
        report["error"] = f"No LeRobot calibration file at {path}"
        return report
    try:
        motors = load(path)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        report["error"] = str(exc)
        return report
    report["calibrated"] = True
    lower, upper = (None, None) if limits is None else (np.asarray(limits[0]), np.asarray(limits[1]))
    for i, name in enumerate(NAMES):
        motor = motors[name]
        low_deg, high_deg = _recorded_degrees(motor)
        sign = robot.joint_signs[i]
        offset = robot.joint_offsets_deg[i]
        travel = sorted(np.deg2rad([low_deg * sign + offset, high_deg * sign + offset]))
        # Where the arm was parked during homing, relative to the calibrated zero. The
        # SO-101's arm joints have symmetric URDF limits, so LeRobot centring degrees on
        # the recorded range puts zero at the mechanical centre -- provided the sweep
        # reached both stops. A nonzero value here only says the parked pose was not the
        # centre, which is harmless on its own; a short recorded travel is the real fault.
        mid_raw = (motor["range_min"] + motor["range_max"]) / 2
        homing_pose_deg = (HOMING_RAW - mid_raw) * 360 / MAX_RAW * sign + offset
        joint = {
            "motor_id": motor["id"],
            "homing_offset": motor["homing_offset"],
            "drive_mode": motor["drive_mode"],
            "recorded_raw": [motor["range_min"], motor["range_max"]],
            "recorded_travel_rad": travel,
            "homing_pose_offset_deg": homing_pose_deg,
        }
        if lower is not None:
            joint["urdf_limit_rad"] = [float(lower[i]), float(upper[i])]
            joint["commandable_rad"] = [
                max(travel[0], float(lower[i])),
                min(travel[1], float(upper[i])),
            ]
            joint["covers_urdf_limit"] = bool(
                travel[0] <= lower[i] + 1e-6 and travel[1] >= upper[i] - 1e-6
            )
            if joint["commandable_rad"][0] >= joint["commandable_rad"][1]:
                report["warnings"].append(
                    f"{name}: recorded travel does not overlap the URDF limits; "
                    "the mapping signs/offsets or the calibration are wrong"
                )
            elif not joint["covers_urdf_limit"]:
                report["warnings"].append(
                    f"{name}: recorded travel {np.round(travel, 3).tolist()} rad is inside the "
                    "URDF limits. The motors clamp to their calibrated range, so poses beyond "
                    "it are silently limited. Re-record the range or keep the work in reach."
                )
            span = travel[1] - travel[0]
            urdf_span = float(upper[i] - lower[i])
            if span > urdf_span * IMPLAUSIBLE_TRAVEL_RATIO:
                report["warnings"].append(
                    f"{name}: recorded travel is {span / urdf_span:.1f}x the URDF range. "
                    "Check that the sweep moved this joint and not another one."
                )
        report["joints"][name] = joint
    parked = {
        name: round(joint["homing_pose_offset_deg"], 1)
        for name, joint in report["joints"].items()
        if abs(joint["homing_pose_offset_deg"]) > 5
    }
    if parked:
        report["notes"].append(
            f"The pose held during homing was off the calibrated zero on {parked} (degrees). "
            "That is expected unless you parked exactly mid-range, and is harmless when the "
            "sweep reached both mechanical stops -- which the recorded travel above shows. "
            "Confirm the mapping with the measured joints and hover probes anyway."
        )
    gripper = motors["gripper"]
    report["gripper"] = {
        "motor_id": gripper["id"],
        "recorded_raw": [gripper["range_min"], gripper["range_max"]],
        "units": "LeRobot normalizes the gripper to 0..100 over its recorded range",
        "closed_pct": robot.gripper_closed_pct,
        "closed_pct_source": "measured by so-paint calibrate"
        if robot.gripper_closed_pct is not None
        else "not measured yet",
    }
    if robot.gripper_closed_pct is None:
        report["warnings"].append(
            "No measured closed-gripper position. Glue the brush, close the gripper on it, "
            "then run uv run so-paint calibrate again."
        )
    return report


def commands(robot) -> dict[str, str]:
    """The LeRobot commands for this arm. Verify them against the installed version's help."""
    port = resolve_port(robot)[0] or "<PORT>"
    # A fresh arm gets the default id, so the command is runnable rather than a template.
    identifier = resolve_id(robot)[0] or DEFAULT_ID
    target = f"--robot.type={ROBOT_TYPE} --robot.port={port} --robot.id={identifier}"
    # LeRobot is an optional extra, and `uv sync` prunes what the default groups omit,
    # so every command that needs it asks for it.
    run = "uv run --extra hardware"
    return {
        "install": "uv sync --locked --extra hardware",
        "find_port": f"{run} lerobot-find-port",
        "setup_motors": f"{run} lerobot-setup-motors {target}",
        "calibrate": f"{run} lerobot-calibrate {target}",
    }


def instructions(robot, report=None) -> list[str]:
    """The prompt to read to the user. Physical steps only they can perform."""
    report = describe(robot) if report is None else report
    steps = []
    port, port_why = resolve_port(robot)
    if port is None:
        steps.append(
            f"Serial port: {port_why}. Plug in the SO-101 follower and its power supply. "
            f"If more than one device shows up, run `{commands(robot)['find_port']}` and "
            "unplug the USB cable when it asks: it prints the right path."
        )
    elif port_why != "configured in the workspace file":
        steps.append(f"Using the serial port {port}: {port_why}.")
    identifier, why = resolve_id(robot)
    if identifier is None and report.get("available_ids"):
        steps.append(
            f"This machine has more than one calibrated arm ({', '.join(report['available_ids'])}). "
            "Tell me which one is the painter, or pass --id."
        )
    elif identifier is None:
        steps.append(
            f"No arm is calibrated yet, so I will file this one under the id '{DEFAULT_ID}'. "
            "Pass --id only if you want a different label, or already keep several arms."
        )
    elif why != "configured":
        steps.append(f"Using the arm id '{identifier}': {why}.")
    if report.get("calibrated"):
        steps.append(
            f"This arm is already calibrated: {report['calibration_file']}. It is reused as is; "
            "`so-paint calibrate` will not re-record it. Pass --recalibrate only if a motor was "
            "replaced, re-seated or its horn was moved."
        )
    else:
        steps.append(
            "Calibrate the arm. It is interactive -- it needs you to move the arm by hand -- "
            "so run this in your own terminal, not through me:\n"
            "  uv run so-paint calibrate\n"
            f"That runs `{commands(robot)['calibrate']}` and re-checks the result afterwards. "
            "It asks you to (1) move the arm to the middle of its range and press ENTER, then "
            "(2) move every joint except wrist_roll through its full travel and press ENTER. "
            "Move each joint to both of its mechanical ends so the recorded range is complete "
            "and centred, then press ENTER."
        )
        steps.append(
            "If the arm is newly assembled and its motor ids were never set, do that first: "
            f"`{commands(robot)['setup_motors']}` (one motor connected at a time)."
        )
    if robot.gripper_closed_pct is None:
        steps.append(
            "Glue the brush across the closed gripper as described in the README, hold the "
            "gripper closed on the brush, then run `uv run so-paint calibrate` again. It "
            "measures where the gripper actually holds, instead of assuming."
        )
    steps.append(
        "`uv run so-paint calibrate` saves the port, id and backend to the workspace file as "
        "it goes. `uv run so-paint doctor` re-checks everything without touching the motors."
    )
    return steps
