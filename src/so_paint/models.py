from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vec2 = tuple[Finite, Finite]
Vec3 = tuple[Finite, Finite, Finite]
RGB = tuple[int, int, int]
Joints5 = Annotated[list[Finite], Field(min_length=5, max_length=5)]
Signs5 = Annotated[list[Literal[-1, 1]], Field(min_length=5, max_length=5)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Pose(Model):
    """Brush-tip pose in the robot base frame: metres, radians, XYZ Euler angles."""

    x: Finite
    y: Finite
    z: Finite
    roll: Finite = 3.141592653589793
    pitch: Finite = 0
    yaw: Finite = 0
    hold_s: Finite = Field(default=0, ge=0, le=10)

    def xyz(self):
        return [self.x, self.y, self.z]

    def rpy(self):
        return [self.roll, self.pitch, self.yaw]


class CameraConfig(Model):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,31}$")
    source: Literal["simulation", "opencv"] = "simulation"
    device: int | str = 0
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)
    eye: Vec3 = (0.18, 0.0, 0.55)
    target: Vec3 = (0.18, 0.0, 0.0)
    up: Vec3 = (0, 1, 0)
    focal_px: Finite = Field(default=700, gt=0)
    calibrated: bool = False
    intrinsic_matrix: list[list[Finite]] | None = None
    distortion: list[Finite] = Field(default_factory=lambda: [0, 0, 0, 0, 0])
    # Simulation extrinsics are exact; hardware camera calibration is a separate step.


class RobotConfig(Model):
    """Connection inventory and motor mapping; startup never opens the port or enables torque."""

    model: Literal["so101"] = "so101"
    port: str | None = None
    # LeRobot stores one calibration per arm id under its cache:
    # <HF cache>/lerobot/calibration/robots/so101_follower/<id>.json.
    # calibration_path overrides that lookup; neither is written by so-paint.
    id: str | None = None
    calibration_path: str | None = None
    gripper: Literal["closed"] = "closed"
    # LeRobot's own kinematics treats calibrated motor degrees as URDF joint degrees, so
    # identity is the vendor convention rather than a guess. Change a sign or offset only
    # after hover checks show this arm disagrees; record why in the workspace file.
    joint_signs: Signs5 = Field(default_factory=lambda: [1, 1, 1, 1, 1])
    joint_offsets_deg: Joints5 = Field(default_factory=lambda: [0.0] * 5)
    # Measured holding position of the closed gripper with the brush glued, in LeRobot's
    # 0..100 gripper units. There is no assumption that zero motor units means closed.
    gripper_closed_pct: Finite | None = Field(default=None, ge=0, le=100)
    # The SO-101's geared servos have real slack. On a direction change a joint can sit
    # still, or sag the wrong way, until the teeth re-engage -- that is the arm, not a
    # fault, and small moves are unreliable because of it. Every guard below allows for
    # it, and moves smaller than this are flagged as unlikely to do anything.
    backlash_deg: Finite = Field(default=6, ge=0, le=30)
    # Hardware execution guards. The controller stops in place when one is exceeded.
    max_relative_target_deg: Finite = Field(default=8, gt=0, le=45)
    max_tracking_error_deg: Finite = Field(default=6, gt=0, le=30)
    start_tolerance_deg: Finite = Field(default=4, gt=0, le=30)
    feedback_timeout_s: Finite = Field(default=0.5, gt=0, le=5)
    # Releasing torque drops the arm and the glued brush onto the workspace.
    disable_torque_on_disconnect: bool = False


class Station(Model):
    name: str
    kind: Literal["paint", "washer"]
    center: Vec3
    radius_m: Finite
    rim_z: Finite
    # Bristle insertion below the estimated paint/water contact surface.
    immersion_m: Finite = Field(default=0, ge=0, le=0.02)
    # Explicit destination for intentional color mixing; ordinary wells stay protected.
    mixing_well: bool = False
    color_rgb: RGB


class Workspace(Model):
    paper_corners_xy: list[Vec2] = Field(
        default_factory=lambda: [(0.15, 0.035), (0.225, 0.035), (0.225, -0.035), (0.15, -0.035)],
        min_length=4,
        max_length=4,
    )
    stations: list[Station] = Field(
        default_factory=lambda: [
            Station(name=n, kind=k, center=(x, y, 0.012), radius_m=0.01, rim_z=0.018, color_rgb=c)
            for n, k, x, y, c in [
                ("red", "paint", 0.145, -0.065, (220, 55, 65)),
                ("blue", "paint", 0.18, -0.07, (45, 100, 220)),
                ("yellow", "paint", 0.215, -0.065, (245, 195, 40)),
                ("washer", "washer", 0.18, 0.065, (135, 185, 180)),
            ]
        ]
    )
    source: str = "simulation_fixture"

    @model_validator(mode="after")
    def validate_stations(self):
        if not any(s.kind == "washer" for s in self.stations):
            raise ValueError("Workspace needs a washer")
        if not any(s.kind == "paint" for s in self.stations):
            raise ValueError("Workspace needs at least one paint well")
        if len({s.name for s in self.stations}) != len(self.stations):
            raise ValueError("Station names must be unique")
        for s in self.stations:
            if s.radius_m <= 0.004 or s.center[2] < 0 or s.center[2] >= s.rim_z:
                raise ValueError("Station geometry is invalid")
            if any(not 0 <= c <= 255 for c in s.color_rgb):
                raise ValueError("RGB values must be 0..255")
        return self


MAX_BATCH_DURATION_S = 300


class Settings(Model):
    # "lerobot" drives a physical SO-101 through LeRobot using its motor calibration.
    backend: Literal["simulation", "lerobot"] = "simulation"
    robot: RobotConfig = Field(default_factory=RobotConfig)
    # None selects every configured camera; an explicit list controls composite order.
    look_at_cameras: list[str] | None = None
    rerun_screenshots: bool = True
    record_camera_hz: Finite = Field(default=2, gt=0, le=30)
    # Logging physical camera frames while the arm moves means holding the device open
    # during the batch. A USB camera has one owner, so that competes with look_at for the
    # same device and can leave it busy. Off by default: observation wins. Turn it on only
    # with a camera that look_at does not use.
    record_cameras_during_motion: bool = False
    workspace: Workspace = Field(default_factory=Workspace)
    cameras: list[CameraConfig] = Field(
        default_factory=lambda: [
            CameraConfig(name="overhead"),
            CameraConfig(name="side", eye=(0.38, -0.4, 0.3), target=(0.16, 0, 0.04), up=(0, 0, 1)),
        ],
        min_length=1,
        max_length=16,
    )
    table_z: Finite = Field(default=0, ge=-0.1, le=0.1)
    paper_z: Finite = 0.005
    paper_contact_depth_m: Finite = Field(default=0, ge=0, le=0.01)
    hover_z: Finite = 0.028
    # Offset is in gripper_frame_link; the brush points across its forward Z axis.
    brush_tip_offset: Vec3 = (0.1, 0, -0.1)
    brush_length_m: Finite = Field(default=0.1, gt=0, le=0.5)
    brush_mount_rpy: Vec3 = (0, 1.5707963267948966, 0)
    brush_axis: Vec3 = (0, 0, 1)
    max_tip_speed: Finite = Field(default=0.045, gt=0, le=0.1)
    paint_speed: Finite = Field(default=0.02, gt=0, le=0.05)
    max_joint_speed: Finite = Field(default=0.6, gt=0, le=1)
    max_joint_accel: Finite = Field(default=1.5, gt=0, le=3)
    sample_hz: int = Field(default=50, ge=30, le=100)
    max_batch_duration_s: Finite = Field(default=60, ge=1, le=MAX_BATCH_DURATION_S)
    position_tolerance_m: Finite = Field(default=0.0015, gt=0, le=0.005)
    brush_tilt_tolerance_deg: Finite = Field(default=5, gt=0, le=15)
    edge_margin_m: Finite = Field(default=0.002, ge=0.001, le=0.01)
    max_load_distance_m: Finite = Field(default=0.15, gt=0)
    brush_width_m: Finite = Field(default=0.003, gt=0, le=0.03)
    wash_dwell_s: Finite = Field(default=0.75, gt=0, le=10)
    load_dwell_s: Finite = Field(default=0.3, gt=0, le=10)
    sim_paint_offset_xy: Vec2 = (0.0008, -0.0004)

    @model_validator(mode="after")
    def valid_geometry(self):
        if len({c.name for c in self.cameras}) != len(self.cameras):
            raise ValueError("Camera names must be unique")
        if self.look_at_cameras is not None:
            names = {c.name for c in self.cameras}
            if not self.look_at_cameras or len(set(self.look_at_cameras)) != len(
                self.look_at_cameras
            ):
                raise ValueError("look_at_cameras must be nonempty and unique")
            if not set(self.look_at_cameras) <= names:
                raise ValueError("look_at_cameras must reference configured camera names")
        if not self.table_z <= self.paper_z < self.hover_z:
            raise ValueError("Require table_z <= paper_z < hover_z")
        if sum(v * v for v in self.brush_axis) < 0.5:
            raise ValueError("Brush axis must be nonzero")
        if self.backend == "lerobot":
            missing = [
                name
                for name, value in [
                    ("robot.port", self.robot.port),
                    ("robot.id", self.robot.id),
                    ("robot.gripper_closed_pct", self.robot.gripper_closed_pct),
                ]
                if value is None or value == ""
            ]
            if missing:
                raise ValueError(
                    f"The lerobot backend needs {', '.join(missing)}. "
                    "Run: uv run so-paint calibrate"
                )
        return self

    def observation_cameras(self):
        by_name = {c.name: c for c in self.cameras}
        return (
            [by_name[name] for name in self.look_at_cameras]
            if self.look_at_cameras is not None
            else self.cameras
        )


class Sample(Model):
    t: float
    joints: list[float]
    tip: Vec3
    phase: Literal["travel", "paint", "load", "wash"]
    station: str | None = None


class Trajectory(Model):
    id: str
    start_revision: int
    samples: list[Sample]
    duration_s: float
    max_tip_error_m: float
    max_orientation_error_deg: float
    orientation_mode: Literal["full", "brush_axis"]
    waypoints: list[Pose]
