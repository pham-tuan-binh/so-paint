"""Rerun is the scene viewer/logger; Python owns kinematics and camera capture.

The headless ViewerClient returns an actual rendered 3D view as PNG. It is a
reconstruction image, kept separate from physical camera observations.
"""

import socket
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import ViewerClient
from rerun.urdf import UrdfTree

from .cameras import camera_matrices
from .kinematics import NAMES
from .telemetry import ExecutionLog


class Recorder:
    def __init__(self, path: Path, world, settings):
        self.stream = rr.RecordingStream("so-paint")
        self.world = world
        self.settings = settings
        self.viewer = None
        self.render_error = None
        self.scene_view = rrb.Spatial3DView(
            origin="/",
            contents=["+ /world/**", "+ /robot/**"],
            name="SO-101 · reconstructed workspace",
            spatial_information=rrb.SpatialInformation(target_frame="base_link"),
            eye_controls=rrb.EyeControls3D(
                position=(0.52, -0.45, 0.38), look_target=(0.14, 0, 0.10), eye_up=(0, 0, 1)
            ),
            background=[29, 34, 41],
            line_grid=False,
        )
        self.camera_paths = {c.name: f"world/cameras/{c.name}/image" for c in settings.cameras}
        if settings.rerun_screenshots:
            try:
                # Avoid sharing an unrelated viewer or another painting session's port.
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                self.viewer = ViewerClient.spawn(
                    headless=True,
                    port=port,
                    hide_welcome_screen=True,
                    detach_process=False,
                    memory_limit="512MiB",
                )
                self.stream.set_sinks(rr.FileSink(path), rr.GrpcSink(self.viewer.url))
            except Exception as e:  # noqa: BLE001 — optional renderer must not break camera capture
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                self.render_error = f"Rerun renderer unavailable: {e}"
                self.stream.save(str(path))
        else:
            self.stream.save(str(path))
        self.stream.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    self.scene_view,
                    rrb.Vertical(
                        *[
                            rrb.Spatial2DView(origin=self.camera_paths[c.name], name=c.name)
                            for c in settings.cameras
                        ]
                    ),
                    column_shares=[2, 1],
                ),
                rrb.TimePanel(timeline="time", play_state="Following"),
                collapse_panels=True,
            )
        )
        self.urdf = UrdfTree.from_file_path(
            Path(__file__).parent / "assets/so101.urdf",
            entity_path_prefix="robot",
            static_transform_entity_path="tf_static",
        )
        self.urdf.log_urdf_to_recording(self.stream)
        self.stream.log(
            "world",
            rr.CoordinateFrame("base_link"),
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )
        # Coarse workspace surfaces help compare the model with camera observations.
        self._world_log(
            "table",
            rr.Mesh3D(
                vertex_positions=[
                    [0.02, -0.15, 0],
                    [0.36, -0.15, 0],
                    [0.36, 0.15, 0],
                    [0.02, 0.15, 0],
                ],
                triangle_indices=[[0, 1, 2], [0, 2, 3]],
                albedo_factor=[95, 104, 110],
            ),
            static=True,
        )
        self._world_log(
            "paper/boundary",
            rr.LineStrips3D(
                [np.vstack([world.corners, world.corners[0]])], colors=[70, 230, 195], radii=0.0005
            ),
            static=True,
        )
        for s in world.stations:
            theta = np.linspace(0, 2 * np.pi, 50)
            ring = np.c_[
                s.center[0] + s.radius_m * np.cos(theta),
                s.center[1] + s.radius_m * np.sin(theta),
                np.full(50, s.rim_z),
            ]
            self._world_log(
                f"stations/{s.name}/rim",
                rr.LineStrips3D([ring], colors=s.color_rgb, radii=0.001),
                static=True,
            )
            self._world_log(
                f"stations/{s.name}/contact",
                rr.Points3D([s.center], labels=[s.name], radii=0.002, colors=s.color_rgb),
                static=True,
            )
        for c in settings.cameras:
            if c.source == "simulation" or c.calibrated:
                k, w = camera_matrices(c)
                inv = np.linalg.inv(w)
                self.stream.log(
                    f"world/cameras/{c.name}",
                    rr.Transform3D(
                        translation=inv[:3, 3],
                        mat3x3=inv[:3, :3],
                        parent_frame="base_link",
                        child_frame=f"{c.name}/camera",
                    ),
                    static=True,
                )
                self.stream.log(
                    self.camera_paths[c.name],
                    rr.Pinhole(
                        image_from_camera=k,
                        resolution=[c.width, c.height],
                        image_plane_distance=0.025,
                        parent_frame=f"{c.name}/camera",
                        child_frame=f"{c.name}/pixels",
                    ),
                    rr.CoordinateFrame(f"{c.name}/pixels"),
                    static=True,
                )
        self.execution = ExecutionLog(
            self.stream, self.robot, self.camera_paths, history_samples=settings.sample_hz * 60,
            frame_dir=path.parent / "motion-frames"
        )
        self.stream.flush()

    def _world_log(self, path, data, **kwargs):
        self.stream.log(f"world/{path}", data, rr.CoordinateFrame("base_link"), **kwargs)

    def robot(self, arm, q, time_s, *, gripper_rad=None):
        self.stream.set_time("time", duration=time_s)
        for name, value in zip(NAMES, q):
            joint = self.urdf.get_joint_by_name(name)
            self.stream.log(
                f"world/joint_transforms/{name}", joint.compute_transform(float(value), clamp=False)
            )
        # Do not manufacture physical gripper feedback. Simulation holds the closed reference at zero.
        if gripper_rad is not None:
            self.stream.log(
                "world/joint_transforms/gripper",
                self.urdf.get_joint_by_name("gripper").compute_transform(gripper_rad, clamp=False),
            )
        tip, axis, _ = arm.fk(q)
        self._world_log(
            "brush",
            rr.LineStrips3D(
                [[tip - axis * self.settings.brush_length_m, tip]],
                radii=0.0015,
                colors=[220, 190, 145],
            ),
        )
        self._world_log("brush_tip", rr.Points3D([tip], radii=0.002, colors=[250, 245, 220]))

    def observation(self, frames, time_s, arm, q, *, source="simulated"):
        """Log a review-time snapshot. `source` says whether the state is measured.

        Simulation holds the closed gripper reference at zero; a physical gripper angle
        is not manufactured. In measured mode the textured paper is the *predicted*
        deposition, logged under its own entity path so it cannot be read as observed.
        """
        self.execution.feedback(
            arm, q, time_s, source=source, gripper_rad=0 if source == "simulated" else None
        )
        painting = self.world.canvas if source == "simulated" else self.world.expected
        self._world_log(
            "paper/surface" if source == "simulated" else "paper/predicted_surface",
            rr.Mesh3D(
                vertex_positions=self.world.corners,
                triangle_indices=[[0, 2, 1], [0, 3, 2]],
                vertex_texcoords=[[0, 0], [1, 0], [1, 1], [0, 1]],
                albedo_texture=painting,
            ),
        )
        sources = {
            c.name: "simulation" if c.source == "simulation" else "physical"
            for c in self.settings.cameras
        }
        for name, frame in frames.items():
            self.execution.camera_frame(name, frame, time_s, source=sources[name])
        self.stream.flush()

    def plan(self, trajectory, time_s):
        self.stream.set_time("time", duration=time_s)
        self._world_log(
            "plan",
            rr.LineStrips3D(
                [[s.tip for s in trajectory.samples]], colors=[60, 220, 190], radii=0.0006
            ),
        )
        self._world_log(
            "waypoints",
            rr.Points3D(
                [p.xyz() for p in trajectory.waypoints],
                labels=[str(i) for i in range(len(trajectory.waypoints))],
                radii=0.0015,
                colors=[90, 190, 255],
            ),
        )

    def screenshot(self, path: Path):
        if self.viewer is None:
            return None
        try:
            self.stream.flush()
            # View ID gives just the rendered 3D scene, without timeline or UI chrome.
            # Blueprint ingestion is asynchronous even after the data stream flushes.
            # Retry only the transient missing-view response; all other failures propagate.
            for attempt in range(6):
                try:
                    self.viewer.save_screenshot(str(path.resolve()), view_id=self.scene_view.id)
                    break
                except RuntimeError as error:
                    if "NotFound" not in str(error) or attempt == 5:
                        raise
                    time.sleep(min(0.05 * 2**attempt, 0.5))
            if not path.is_file():
                raise RuntimeError("Viewer did not produce a screenshot")
            self.render_error = None
            return str(path.resolve())
        except Exception as e:  # noqa: BLE001 — optional renderer must not break camera capture
            self.render_error = f"Rerun screenshot failed: {e}"
            return None

    def close(self):
        self.stream.flush()
        self.stream.disconnect()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
