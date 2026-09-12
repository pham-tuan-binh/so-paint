# Record what the robot does and sees

Rerun is the execution record for the real robot as well as the simulated workbench.
Recording happens continuously while the controller and cameras run; it must not depend
on the agent calling `look_at`. That tool returns the most recent observation for reasoning.

The logging interface lives in `src/so_paint/telemetry.py`, available as
`recorder.execution`. The LeRobot driver in `src/so_paint/hardware.py` calls exactly
these hooks while it executes:

```python
from time import monotonic

session_origin = monotonic()  # one origin shared by controller and camera workers

# After sending a setpoint to the physical controller:
recorder.execution.command(arm, command_q_rad, sent_at - session_origin)

# On an encoder packet (use its acquisition time in the shared clock domain):
recorder.execution.feedback(
    arm,
    encoder_q_rad,
    sampled_at - session_origin,
    source="measured",
    target_joints=target_at_sample_time,
    gripper_rad=encoder_gripper_rad,
)

# On every captured RGB frame, independently for each camera:
recorder.execution.camera_frame(
    camera_name,
    rgb_uint8,
    captured_at - session_origin,
    source="physical",
)
```

These are internal Python hooks, not additional CLI commands. Do not call a physical
frame a simulated frame or present a command as encoder feedback. `arm` supplies the
calibrated tool-tip forward kinematics; joint arrays contain the five arm axes in
URDF radians, after the verified motor-to-URDF mapping.

Use actual send/sample/capture timestamps, not the time a queued packet reaches the
logger. Translate device timestamps into the same host clock domain first. Cameras
need not have the same frame rate. A delayed image retains its original capture time.
If an aligned target isn't available for an encoder sample, omit `target_joints` rather
than compare against the wrong command. The encoder-derived tip is a kinematic estimate,
not an independent optical measurement of the brush.

The recording contains:

- The planned batch and its waypoints.
- Commanded joint positions and tip trajectory.
- Measured joint positions and encoder-derived tip trajectory, distinctly labeled.
- The URDF pose driven by feedback, never by commands presented as measurements.
- Joint tracking errors and tip error when an aligned target is supplied.
- Individual camera images at capture times, with source and timestamp metadata.
- Measured gripper holding position in LeRobot's 0..100 units. The URDF gripper angle is
  not recoverable from those units, so hardware feedback never animates that joint.

The live path display retains a bounded sample history (60 seconds at the configured
control rate). The `.rrd` keeps the full joint/image stream for replay. Camera images
are JPEG-compressed at quality 90 for logging; original frames remain available to the
capture pipeline and `look_at` image generation. Logging does not establish a controller
stop policy: the driver handles stale feedback and excessive error itself.

**Physical camera frames are not logged during motion by default.** A USB camera has one
owner: holding it open through a batch competes with `look_at` for the same device, and a
capture stuck in a read leaves it busy afterwards. Seeing the brush after a move matters
more than frames during it, so `record_cameras_during_motion` is false and hardware
batches are recorded from encoders alone. Turn it on only for a camera that `look_at`
does not use; when it is on, a camera that drops out is reopened and released rather than
held broken.

The simulation exercises these hooks at every motion sample and produces synthetic
camera frames during execution at `record_camera_hz` (default 2 Hz). Its feedback is
explicitly tagged `simulated`. Real camera devices attached to a simulation session are
view-only snapshots on the simulation review timeline; they are not used as evidence of
physical execution. A real backend must use the independently timestamped capture hooks
above, which is what it does: setpoints through `command`, every encoder sample through
`feedback(source="measured")` with the setpoint that was active when it was acquired, the
gripper through `gripper`, and physical frames from a separate capture thread on their own
acquisition times. `Recorder.observation(..., source="measured")` is the review-time
snapshot only; the continuous record does not depend on `look_at`.
