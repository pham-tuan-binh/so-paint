"""Live execution logging hooks shared by the simulated executor and the LeRobot driver.

Call these while motion is happening, not after move_to returns. All timestamps are
acquisition/send times in seconds from the same session monotonic-clock origin.
A camera frame and joint sample may have different rates/timestamps; do not restamp
both with the later time at which the logger receives them.
"""

import json
import threading
from collections import deque
from pathlib import Path
from uuid import uuid4

import numpy as np
import rerun as rr
from PIL import Image

from .kinematics import NAMES


class ExecutionLog:
    def __init__(self, stream, animate_robot, camera_paths, history_samples=1200, frame_dir=None):
        self.frame_dir = Path(frame_dir).resolve() if frame_dir else None
        self.frame_index = self.frame_dir / "frames.jsonl" if self.frame_dir else None
        if self.frame_dir:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.stream = stream
        self.animate_robot = animate_robot
        self.camera_paths = camera_paths
        self.lock = threading.RLock()
        self.traces = {
            name: deque(maxlen=history_samples) for name in ("commanded", "measured", "simulated")
        }

    @staticmethod
    def _time(time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Telemetry timestamp must be finite session-relative seconds >= 0")

    @staticmethod
    def _joints(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("Telemetry needs five finite joint positions in URDF radians")
        return q

    def _joint_values(self, kind, arm, q, time_s):
        self.stream.set_time("time", duration=time_s)
        tip = arm.fk(q)[0]
        for name, value in zip(NAMES, q):
            self.stream.log(f"telemetry/{kind}/joints/{name}", rr.Scalars(float(value)))
        self.stream.log(
            f"world/execution/{kind}/tip",
            rr.Points3D([tip], radii=0.0018),
            rr.CoordinateFrame("base_link"),
        )
        trace = self.traces[kind]
        # Late arrivals retain their original Rerun timestamp, but never rewind the live trail.
        if not trace or time_s > trace[-1][0]:
            trace.append((time_s, tip))
        elif time_s == trace[-1][0]:
            trace[-1] = (time_s, tip)
        if len(trace) > 1 and time_s >= trace[-1][0]:
            colors = {
                "commanded": [80, 180, 255],
                "measured": [245, 150, 60],
                "simulated": [195, 135, 235],
            }
            self.stream.log(
                f"world/execution/{kind}/path",
                rr.LineStrips3D([[p for _, p in trace]], colors=colors[kind], radii=0.0007),
                rr.CoordinateFrame("base_link"),
            )

    def command(self, arm, joints, sent_at_s):
        """Log the command actually sent to the controller; never animate it as feedback."""
        self._time(sent_at_s)
        q = self._joints(joints)
        with self.lock:
            self._joint_values("commanded", arm, q, sent_at_s)

    def feedback(self, arm, joints, sampled_at_s, *, source, target_joints=None, gripper_rad=None):
        """Log encoder feedback (source='measured') or simulated state, explicitly labeled.

        target_joints, if supplied, is the controller target at this feedback sample's
        acquisition time, not the newest target when the delayed sample arrives.
        The displayed URDF follows this state, never the command stream.
        """
        if source not in ("measured", "simulated"):
            raise ValueError("Feedback source must be measured or simulated")
        self._time(sampled_at_s)
        q = self._joints(joints)
        target = None if target_joints is None else self._joints(target_joints)
        if gripper_rad is not None and not np.isfinite(gripper_rad):
            raise ValueError("Gripper feedback must be finite")
        with self.lock:
            self._joint_values(source, arm, q, sampled_at_s)
            self.animate_robot(arm, q, sampled_at_s, gripper_rad=gripper_rad)
            self.stream.log("telemetry/robot_state_source", rr.TextDocument(source))
            if target is not None:
                for name, error in zip(NAMES, q - target):
                    self.stream.log(
                        f"telemetry/{source}/tracking_error_rad/{name}", rr.Scalars(float(error))
                    )
                error_mm = float(np.linalg.norm(arm.fk(q)[0] - arm.fk(target)[0]) * 1000)
                self.stream.log(f"telemetry/{source}/encoder_tip_error_mm", rr.Scalars(error_mm))

    def gripper(self, percent, sampled_at_s, *, source):
        """Log gripper holding feedback in LeRobot's 0..100 units, not as a joint angle.

        The URDF gripper angle is not recoverable from these units, so this is reported
        as the measured holding position it is, and never animates the model.
        """
        if source not in ("measured", "simulated"):
            raise ValueError("Gripper feedback source must be measured or simulated")
        self._time(sampled_at_s)
        percent = float(percent)
        if not np.isfinite(percent):
            raise ValueError("Gripper feedback must be finite")
        with self.lock:
            self.stream.set_time("time", duration=sampled_at_s)
            self.stream.log(f"telemetry/{source}/gripper_pct", rr.Scalars(percent))

    def camera_frame(self, name, rgb, captured_at_s, *, source):
        """Log an individual RGB frame at capture time, independently of agent look_at calls."""
        self._time(captured_at_s)
        if name not in self.camera_paths:
            raise ValueError(f"Unknown camera {name}")
        if source not in ("physical", "simulation"):
            raise ValueError("Camera source must be physical or simulation")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Camera frames must be uint8 H x W x 3 RGB")
        # JPEG encoding can take longer than a motor command interval. It does not
        # touch the stream timeline, so keep it outside the shared logging lock.
        if self.frame_dir:
            path = self.frame_dir / f"{uuid4().hex}.jpg"
            Image.fromarray(rgb).save(path, quality=90)
            entry = {"camera": name, "captured_at_s": captured_at_s,
                     "source": source, "image": str(path)}
            # Disk and encoding work stay off the motor logger's lock.
            with self.frame_index.open("a") as index:
                index.write(json.dumps(entry) + "\n")
        encoded = rr.Image(rgb).compress(jpeg_quality=90)
        with self.lock:
            self.stream.set_time("time", duration=captured_at_s)
            self.stream.log(self.camera_paths[name], encoded)
            self.stream.log(
                f"telemetry/cameras/{name}",
                rr.AnyValues(captured_at_s=captured_at_s, source=source),
            )
