import argparse
import json
import platform
import sys
from pathlib import Path

from .models import CameraConfig, Settings


def read_settings(path):
    return Settings.model_validate_json(Path(path).read_text()) if path else Settings()


def main():
    parser = argparse.ArgumentParser(
        prog="so-paint", description="Local agent-operated painting workbench"
    )
    parser.add_argument("--config", type=Path, help="Workspace/camera/brush settings JSON")
    parser.add_argument(
        "--state", type=Path, default=Path("runs/server.json"), help="Local server connection file"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Keep a local painting session alive")
    serve.add_argument("--output", type=Path)
    for name in ["look-at", "status", "reload", "stop"]:
        commands.add_parser(name)
    commands.add_parser("cancel", help="Stop a running hardware batch in place")
    recover = commands.add_parser(
        "recover",
        help="Nudge the arm back inside the model's joint limits and lift the brush clear",
    )
    recover.add_argument("--preview", action="store_true")
    recover.add_argument("--elbow-lift", action="store_true", help="Preview an upward elbow-only recovery arc")
    move = commands.add_parser("move-to", help="Submit brush-tip waypoints to the running session")
    move.add_argument("--poses", type=Path, required=True)
    move.add_argument("--preview", action="store_true")
    move.add_argument("--orientation-mode", choices=["full", "brush_axis"])
    demo = commands.add_parser(
        "demo", help="Run clean, color load, stroke and visual review examples"
    )
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    commands.add_parser("doctor", help="Read-only setup diagnostics; never connects to motors")
    init = commands.add_parser("init", help="Write an editable simulation config")
    init.add_argument("--output", type=Path, default=Path("workspace.json"))
    arm_calibration = commands.add_parser(
        "calibrate",
        help="Check or guide LeRobot motor calibration for a physical SO-101",
    )
    arm_calibration.add_argument("--port", help="Serial port; see lerobot-find-port")
    arm_calibration.add_argument("--id", help="LeRobot arm id; names its calibration file")
    arm_calibration.add_argument(
        "--check", action="store_true", help="Only report; connect to nothing and save nothing"
    )
    arm_calibration.add_argument(
        "--recalibrate",
        action="store_true",
        help="Re-record a calibration this arm already has (it is reused by default)",
    )
    calibration = commands.add_parser(
        "calibrate-camera", help="Fit camera pose from observed brush-tip or landmark points"
    )
    calibration.add_argument("observations", type=Path)
    reconstruction = commands.add_parser(
        "reconstruct", help="Map model-detected pixels to known work-surface planes"
    )
    reconstruction.add_argument("observations", type=Path)
    recipe = commands.add_parser(
        "recipe", help="Generate an atomic action as move_to-compatible JSON"
    )
    recipe.add_argument("action", choices=["clean", "color", "stroke"])
    recipe.add_argument("--color", default="red")
    recipe.add_argument("--points", help="Robot-base XY point list, e.g. [[0.16,0],[0.20,0]]")
    recipe.add_argument("--rpy", help="Brush roll,pitch,yaw as JSON radians (required on hardware)")
    station = commands.add_parser("station", help="Generate approach/dip/wash poses; no execution")
    station.add_argument("name")
    station.add_argument("action", choices=["approach", "dip", "wash"])
    station.add_argument("--rpy", required=True, help="JSON [roll,pitch,yaw] radians")
    station.add_argument("--cycles", type=int, default=3)
    draw = commands.add_parser("draw", help="Generate a same-color batch from XY polylines")
    draw.add_argument("--strokes", type=Path, required=True, help="JSON list of polylines")
    draw.add_argument("--rpy", required=True, help="JSON [roll,pitch,yaw] radians")
    draw.add_argument("--normalized", action="store_true", help="Map [0,1] XY to saved paper corners")
    review = commands.add_parser("review", help="Find archived dip/rinse/paint camera evidence")
    review.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "review":
        from .review import review_motion

        print(json.dumps(review_motion(args.report), indent=2))
        return
    if args.command in {"look-at", "move-to", "status", "reload", "stop", "cancel", "recover"}:
        from urllib.error import HTTPError, URLError

        from .service import request

        try:
            payload = {}
            if args.command == "recover":
                payload = {"preview": args.preview, "elbow_lift": args.elbow_lift}
            if args.command == "move-to":
                payload = json.loads(args.poses.read_text())
                if isinstance(payload, list):
                    payload = {"poses": payload}
                if args.preview:
                    payload["preview"] = True
                if args.orientation_mode:
                    payload["orientation_mode"] = args.orientation_mode
            print(json.dumps(request(args.state, args.command, payload), indent=2))
        except HTTPError as exc:
            print(exc.read().decode(), file=sys.stderr)
            raise SystemExit(1) from exc
        except (OSError, URLError, ValueError, TypeError) as exc:
            print(
                json.dumps({"error": str(exc), "hint": "Start uv run so-paint serve first."}),
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        return
    config_path = args.config or Path("workspace.json")
    settings = read_settings(config_path if config_path.exists() else args.config)
    if args.command == "init":
        # Do not silently overwrite a calibrated workspace.
        with args.output.open("x") as f:
            f.write(settings.model_dump_json(indent=2) + "\n")
        print(args.output.resolve())
    elif args.command == "doctor":
        from .kinematics import Arm
        from .world import World

        world, arm = World(settings), Arm(settings)
        reachability = {}
        for s in world.stations:
            try:
                arm.ik(s.center)
                arm.ik([*s.center[:2], settings.hover_z])
                reachability[s.name] = "contact and hover reachable"
            except ValueError as e:
                reachability[s.name] = str(e)
        from .calibration import describe, serial_candidates

        hardware = describe(settings.robot, (arm.lower, arm.upper))
        if settings.backend == "lerobot":
            status = (
                "lerobot backend ready; motors are connected on the first look-at"
                if hardware["calibrated"]
                else f"lerobot backend configured but NOT calibrated: {hardware.get('error')}"
            )
            following = "uv run so-paint serve, then uv run so-paint look-at"
        else:
            status = (
                "simulation backend; no motor connection attempted. A calibrated arm was found, "
                "so switching to the lerobot backend is a config change away."
                if hardware["calibrated"]
                else "simulation backend; no motor connection attempted"
            )
            following = (
                "uv run so-paint demo, then uv run so-paint serve. For a physical arm: "
                "uv run so-paint calibrate (see docs/SETUP.md)"
            )
        print(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "platform": platform.system(),
                    "backend": settings.backend,
                    "commands": ["look-at", "move-to"],
                    "robot": settings.robot.model_dump(),
                    "look_at_cameras": [c.name for c in settings.observation_cameras()],
                    "serial_candidates": serial_candidates(),
                    "cameras": [
                        {
                            "name": c.name,
                            "source": c.source,
                            "device": c.device,
                            "registered": c.source == "simulation" or c.calibrated,
                        }
                        for c in settings.cameras
                    ],
                    "workspace_reachability": reachability,
                    "hardware": hardware,
                    "hardware_status": status,
                    "next": following,
                },
                indent=2,
            )
        )
    elif args.command == "serve":
        from .service import serve
        from .workbench import Workbench

        workbench = Workbench(settings, args.output)
        try:
            serve(workbench, args.state, config_path.resolve())
        finally:
            workbench.close()
    elif args.command == "demo":
        if settings.backend != "simulation":
            parser.error(
                f"demo is a simulation example and will not drive the {settings.backend} "
                "backend. Use serve, look-at and move-to for a physical arm."
            )
        run_demo(settings, args.output)
    elif args.command == "calibrate":
        print(json.dumps(run_calibration(args, settings, config_path, parser), indent=2))
    elif args.command == "calibrate-camera":
        from .registration import calibrate, validate_camera

        data = json.loads(args.observations.read_text())
        camera, report = calibrate(
            CameraConfig.model_validate(data["camera"]), data["robot_points"], data["image_points"]
        )
        held_out = None
        if "validation_robot_points" in data or "validation_image_points" in data:
            held_out = validate_camera(camera, data.get("validation_robot_points"),
                                       data.get("validation_image_points"))
            if not held_out["passed"]:
                camera = camera.model_copy(update={"calibrated": False})
        print(json.dumps({"camera": camera.model_dump(), "validation": report,
                          "held_out_validation": held_out,
                          "ready_for_contact": False}, indent=2))
    elif args.command == "reconstruct":
        from .registration import reconstruct

        result = reconstruct(settings.cameras, json.loads(args.observations.read_text()))
        print(json.dumps(result, indent=2))
    elif args.command in {"station", "draw"}:
        from .recipes import drawing, station_action
        from .world import World

        world = World(settings)
        try:
            rpy = json.loads(args.rpy)
            if args.command == "station":
                poses = station_action(world, args.name, args.action, rpy=rpy, cycles=args.cycles)
            else:
                poses = drawing(world, json.loads(args.strokes.read_text()), rpy=rpy,
                                normalized=args.normalized)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps({"poses": [p.model_dump() for p in poses],
                          "orientation_mode": "brush_axis"}, indent=2))
    elif args.command == "recipe":
        from .recipes import brush_stroke, clean_brush, load_color, orient
        from .world import World

        if settings.backend == "lerobot" and args.rpy is None:
            parser.error("Hardware recipes require --rpy; reuse a visually validated orientation")
        if settings.backend == "lerobot" and args.action in {"color", "clean"}:
            parser.error("Use station approach, inspect look-at, then station dip/wash on hardware")
        world = World(settings)
        if args.action == "clean":
            poses = clean_brush(world)
        elif args.action == "color":
            poses = load_color(world, args.color)
        else:
            if args.points is None:
                parser.error("stroke requires --points")
            poses = brush_stroke(world, json.loads(args.points))
        if args.rpy is not None:
            poses = orient(poses, json.loads(args.rpy))
        print(
            json.dumps(
                {"poses": [p.model_dump() for p in poses], "orientation_mode": "brush_axis"},
                indent=2,
            )
        )


def run_calibration(args, settings, config_path, parser):
    """Take the arm's setup as far as it can go, and report what the user must still do.

    Run it, read the `prompt` it returns, do those things, run it again. It calibrates
    when there is no calibration, measures the closed gripper when that is missing, and
    saves what it resolved. `--check` does none of that and only looks. so-paint never
    writes the motor calibration file itself: LeRobot does.
    """
    import shlex
    import subprocess
    from importlib.metadata import PackageNotFoundError, version
    from importlib.util import find_spec

    import numpy as np

    from .calibration import (
        DEFAULT_ID,
        commands,
        describe,
        instructions,
        resolve_id,
        resolve_port,
        serial_candidates,
        to_motor_degrees,
    )
    from .kinematics import NAMES, Arm

    arm = Arm(settings)
    limits = (arm.lower, arm.upper)
    robot = settings.robot.model_copy(
        update={key: value for key, value in [("port", args.port), ("id", args.id)] if value}
    )
    try:
        lerobot = {"installed": find_spec("lerobot") is not None, "version": version("lerobot")}
    except PackageNotFoundError:
        lerobot = {"installed": False, "version": None, "install": "uv sync --extra hardware"}
    result = {"lerobot": lerobot, "serial_candidates": serial_candidates()}
    # The workspace file supplies the port and id; the command line overrides them, and
    # an obvious single device or single saved calibration fills in what is still missing.
    robot = robot.model_copy(
        update={"port": resolve_port(robot)[0], "id": resolve_id(robot)[0] or robot.id}
    )
    state = describe(robot, limits)
    # Calibrating is the default. An arm that is already calibrated keeps what it has.
    if not args.check and (args.recalibrate or not state["calibrated"]):
        # A first calibration gets the default id, without pinning it on this report:
        # once LeRobot has written the file, it is discovered like any other.
        target = robot.model_copy(update={"id": robot.id or DEFAULT_ID})
        command = commands(target)["calibrate"]
        if not lerobot["installed"]:
            result["run"] = {
                "started": False,
                "reason": "LeRobot is not installed.",
                "command": commands(target)["install"],
            }
        elif robot.port is None:
            result["run"] = {
                "started": False,
                "reason": f"No serial port to calibrate on: {resolve_port(robot)[1]}.",
                "command": commands(target)["find_port"],
            }
        elif not sys.stdin.isatty():
            # LeRobot's calibration waits on ENTER while the user moves the arm, so it
            # cannot be driven from an agent or a pipe. Hand the command over instead.
            result["run"] = {
                "started": False,
                "reason": "LeRobot's calibration asks you to move the arm, so it needs an "
                "interactive terminal. Run it yourself:",
                "command": command,
            }
        else:
            print(f"Running {command}\n", file=sys.stderr)
            completed = subprocess.run(shlex.split(command), check=False)
            result["run"] = {"started": True, "command": command, "exit_code": completed.returncode}
    elif state["calibrated"] and not args.check:
        result["run"] = {
            "started": False,
            "reason": f"This arm is already calibrated: {state['calibration_file']}. "
            "Reusing it. Pass --recalibrate only if a motor was replaced or re-seated.",
        }
    # With a calibration in place, the last missing piece is where the gripper holds the
    # brush. Measuring it is a read-only connection, so just take the measurement.
    if (
        not args.check
        and robot.gripper_closed_pct is None
        and robot.port
        and describe(robot)["calibrated"]
    ):
        from .hardware import LeRobotArm

        driver = LeRobotArm(settings.model_copy(update={"robot": robot}))
        try:
            driver.connect(torque=False)
            reading = driver.read()
        except Exception as exc:  # noqa: BLE001 - a diagnostic must report, not crash
            reading = None
            result["measured"] = {"failed": f"{type(exc).__name__}: {exc}"}
        finally:
            driver.disconnect()
    else:
        reading = None
    if reading is not None:
        q = reading.joints_rad
        result["measured"] = {
            "connection": "read-only: this command did not energize the motors",
            "joints_rad": dict(zip(NAMES, q.tolist())),
            "motor_degrees": to_motor_degrees(q, robot),
            "within_urdf_limits": bool(np.all(q >= arm.lower) and np.all(q <= arm.upper)),
            "brush_tip_xyz": arm.fk(q)[0].tolist(),
            "gripper_pct": reading.gripper_pct,
            "check": "Move a joint by hand and re-run: the reported radians must move the "
            "same way. If they do not, that joint's robot.joint_signs is wrong. Never "
            "absorb a mapping error into the brush offset.",
        }
        robot = robot.model_copy(update={"gripper_closed_pct": reading.gripper_pct})
    report = describe(robot, limits)
    if not args.check:
        # Save what this run resolved, so the setup is reproducible.
        resolved = robot.model_copy(update={"id": report["robot_id"], "port": report["port"]})
        report["config_written"] = write_robot_config(config_path, settings, resolved, report)
    result.update(report)
    result["commands"] = commands(robot)
    result["prompt"] = instructions(robot, report)
    return result


def write_robot_config(path, settings, robot, report):
    """Save the robot connection into the config, leaving every other setting untouched.

    The backend only becomes "lerobot" once the arm is calibrated, its port and id are
    known, and the closed-gripper position has actually been measured.
    """
    data = json.loads(path.read_text()) if path.exists() else settings.model_dump(mode="json")
    previous = data.get("robot", {})
    remaining = []
    if not report["calibrated"]:
        remaining.append("LeRobot motor calibration")
    if not robot.port:
        remaining.append("robot.port")
    if not robot.id:
        remaining.append("robot.id")
    if robot.gripper_closed_pct is None:
        remaining.append("measured robot.gripper_closed_pct")
    data["robot"] = robot.model_dump(mode="json")
    data["backend"] = "simulation" if remaining else "lerobot"
    # Never leave an inconsistent setup on disk.
    Settings.model_validate(data)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "path": str(path.resolve()),
        "backend": data["backend"],
        "previous_robot": previous,
        "still_needed": remaining,
    }


def run_demo(settings, output):
    from .recipes import brush_stroke, load_color
    from .workbench import Workbench

    workbench = Workbench(settings, output)
    try:
        reports = []
        workbench.look_at()
        # Three atomic operations, with a camera review between batches.
        for color, line in [
            ("red", [(0.16, 0.012), (0.21, 0.012)]),
            ("blue", [(0.16, -0.012), (0.21, -0.012)]),
        ]:
            for poses in [load_color(workbench.world, color), brush_stroke(workbench.world, line)]:
                reports.append(workbench.move_to(poses))
                workbench.look_at()
        result = {
            "reports": reports,
            "artifacts": str(workbench.output),
            "recording": str(workbench.recording_path),
            "note": "Simulation; no hardware moved.",
        }
        print(json.dumps(result, indent=2))
    finally:
        workbench.close()


if __name__ == "__main__":
    main()
