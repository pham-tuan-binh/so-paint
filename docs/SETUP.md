# Agent-led setup

Use the shortest path that applies. The user should not need to learn robot internals
before describing a painting. Ask one concrete missing question at a time.

## Already calibrated and positioned

Read the saved workspace and motor calibration, inspect cameras, and verify that the
active backend matches the intended robot. Do not repeat motor calibration. Recheck
visual alignment only when the brush, camera, arm base, paper or stations have moved.
A ready setup goes directly from the painting description to look / move / look.

A physical arm runs on `backend: "lerobot"`, which uses the arm's own LeRobot motor
calibration. **That driver has not yet been exercised against a real SO-101 from this
repository**: its execution path is covered by tests against a stubbed follower only. Say
so, bring it up with hover probes under supervision, and never silently substitute
simulated motion for a real request.

## New machine

1. Run `uv sync --locked`, then `uv run so-paint doctor`. The diagnostic lists candidate
   serial devices without opening or energizing them and checks demo station reachability.
2. If there is no robot, demonstrate simulation with `uv run so-paint demo`.
3. For a physical arm, run `uv run so-paint calibrate`. It opens no port, and reports the
   LeRobot calibration state plus a `prompt` list of the physical steps only the user can
   perform. Work through that list with them. Do not overwrite an existing calibration.
4. Identify available cameras. Start with the user's known devices; do not scan every
   index or use an unrelated private feed. A wide view of the table and a second view
   resolving brush height are helpful, but no particular placement is required.
5. `calibrate` saves the connection under `robot` as it goes. Add camera devices under
   `cameras` and the composite feeds under `look_at_cameras`. Start `uv run so-paint serve`
   in a persistent shell, then `uv run so-paint look-at`: that is when the port opens and
   the first encoder measurement arrives.

## Motor calibration

Install the hardware extra: `uv sync --locked --extra hardware` (LeRobot with its
Feetech driver). Keep the extra on every later sync -- `uv sync` prunes packages the
requested groups do not name, so a plain `uv sync --locked` uninstalls LeRobot again. Then `uv run so-paint calibrate` is the entry point for everything below. Run it, do what it
says, run it again. From a terminal with an uncalibrated arm it launches LeRobot's
calibration; from an agent or a pipe it hands the command over instead, because that
calibration needs a human to move the arm. It then measures the closed gripper and saves
what it resolved. `--check` does none of that and only looks; an already-calibrated arm is
reused as is, and re-recording one takes `--recalibrate`. Every invocation reports:

- `port` / `port_source` and `robot_id` / `robot_id_source`: both resolve themselves.
  The workspace file wins when its port is actually connected; otherwise a single present
  serial device is adopted and said so. Several devices (an SO-101 pair, or a board with
  two interfaces) are a real ambiguity, so it stops and points at `lerobot-find-port`
  rather than guessing which one to write to. `--port` and `--id` override either.
- `calibration_file` / `calibrated`: whether this arm already has a LeRobot calibration.
  LeRobot files one per arm id under
  `<HF cache>/lerobot/calibration/robots/so_follower/<id>.json`. A single saved
  calibration is picked up without naming it, several need `--id`, and a first
  calibration is filed under `painter` unless you choose otherwise. An id you have used
  before reuses that calibration; **do not re-record it.**
- `joints`: each motor's recorded travel converted to URDF radians and compared with the
  model's joint limits, plus `homing_pose_offset_deg` -- where the arm was parked during
  homing, relative to the calibrated zero.
- `warnings`: recorded travel that does not reach the model's limits (the motors clamp to
  their calibrated range, so poses beyond it are silently limited), travel far wider than
  the model's range, or a missing closed-gripper measurement.
- `notes`: things worth knowing that are not faults, such as a homing pose that was not
  mid-range.
- `prompt`: the steps to read to the user, and `commands`: the exact LeRobot commands.

`so-paint calibrate` runs the middle command below for them. The equivalent by hand, in
their own terminal:

```sh
uv run --extra hardware lerobot-find-port     # unplug the arm when asked; prints the port
uv run --extra hardware lerobot-setup-motors --robot.type=so101_follower --robot.port=... --robot.id=...
uv run --extra hardware lerobot-calibrate  --robot.type=so101_follower --robot.port=... --robot.id=...
```

`setup-motors` is only for a newly assembled arm whose motor ids were never set.
`lerobot-calibrate` asks the user to park the arm mid-range, then to move every joint
except `wrist_roll` through its whole travel. **Both mechanical ends of every joint
matter**, and they matter far more than where the arm was parked first: LeRobot centres
its degrees on the *recorded* range, and the SO-101's arm joints have symmetric URDF
limits, so a sweep that reaches both stops puts zero at the mechanical centre wherever
the arm started. A sweep that stops short does not, and that is what the recorded travel
in `joints` shows. so-paint never writes the calibration file itself; LeRobot does.

Verify the installed LeRobot's own CLI help before quoting commands, and see the
[official SO-101 instructions](https://huggingface.co/docs/lerobot/so101). A connected
USB port alone does not prove correct motor ids, zero offsets or units.

Finally, glue the brush across the closed gripper, hold the gripper closed on it, and run
`uv run so-paint calibrate` once more. With a calibration in place it makes a read-only
connection -- energizing nothing -- and reports the measured joints in URDF radians, the
same values in LeRobot motor degrees, the brush-tip position they imply, and where the
gripper actually holds. That last number is saved as `robot.gripper_closed_pct`, so
nothing assumes zero motor units means closed, and `backend` flips to `lerobot`.

Read those measured joints against the arm in front of you. Move a joint by hand, re-run,
and check the reported radians moved the same way: if they did not, that joint's sign is
wrong. Fix it in `robot.joint_signs` rather than absorbing it into the brush offset.

### The motor-to-URDF mapping

LeRobot's own kinematics treats calibrated motor degrees as URDF joint degrees, so
`robot.joint_signs` and `robot.joint_offsets_deg` default to identity -- the vendor
convention, not a guess. Change one only when hover checks prove this arm disagrees, and
record why in the workspace file. A measured pose outside the URDF limits is refused
outright: that means the mapping or the calibration is wrong, not that the pose needs
clamping.

## Workspace and arbitrary brushes

Let the model inspect the feeds and propose the paper polygon, washer, paint wells,
color identities and brush tip. A new brush needs an estimated tip offset, local brush
axis, usable width and initial contact height. Those are estimates to refine, not fixed
properties of the robot. Ask the user for a ruler/known scale only if no metric reference
exists. Camera placement alone does not register camera coordinates to the arm base.

Use measured fiducial points or a known board rigidly related to the robot to establish
scale and camera extrinsics. The model reads the pixel positions; users need not click.
See `REAL2SIM.md`. If geometry changes, update `workspace.json` and run `uv run so-paint reload`, then `look-at`.
Keep prior settings and observations for comparison.

Then use short hover probes above paper edges and each station. Compare predicted tip
pixels against the real tip in all views, correct transform/offset errors, and repeat.
When those match, test one short stroke on a spare margin. Estimate footprint, useful
contact height, speed and load distance from the result. Refine wash/load dwell times
with the actual brush and materials. Keep the setup compact and comfortably reachable;
move a well or paper when a vertical approach saturates a joint.

## What the hardware backend does

`src/so_paint/hardware.py` is the driver; `src/so_paint/calibration.py` owns the
calibration files and the mapping. The look/move interface does not change.

- Nothing connects on import, in `doctor`, in `calibrate --check`, or at `serve` startup.
  Plain `calibrate` connects read-only to measure the gripper; the port opens for real on
  the first `look-at`.
- Connecting releases torque, adopts the saved calibration, makes the *present measured*
  pose the goal, then configures and energizes -- so it holds the arm where it already
  is instead of snapping to a stale goal, as LeRobot's own `connect()` would. It assumes
  the arm is resting: support it if it is holding a pose.
- `look-at` reports measured encoder positions, labeled `joints_source: "measured"`, and
  the tip pose is forward kinematics of those joints, never an optical measurement.
- Cameras are opened fresh for each `look-at` and retried three times. Nothing else opens
  them: `record_cameras_during_motion` is false by default, because a USB camera has one
  owner and logging frames through a batch competes with observation for the same device.
  A single camera should stay that way. Big frames make every open slower, so prefer
  1280x720 over 1920x1080 -- the composite is scaled to 640x480 regardless.
- `move-to` preflights the whole batch exactly as simulation does, then replays the
  trajectory at `sample_hz`, holding `gripper_closed_pct` in every setpoint, logging each
  command and each encoder sample through the [telemetry hooks](TELEMETRY.md), and
  capturing the physical cameras in a separate thread on their own timestamps.
- It stops in place -- commanding the measured position -- on cancellation, tracking
  error beyond `max_tracking_error_deg`, feedback latency beyond `feedback_timeout_s`, or
  any device fault, and it refuses to start from a pose that disagrees with the plan by
  more than `start_tolerance_deg` or with a gripper that has moved off its reference.
- Execution runs outside the session lock, so `status` and `cancel` stay answerable while
  the arm moves. A partial batch is still recorded and reported with `completed`,
  `stop_reason`, `commanded_samples` and the measured final pose. Observe before deciding
  what to do next; never resend a partially executed batch.
- Torque is left engaged on disconnect (`disable_torque_on_disconnect: false`), because
  releasing it drops the arm and the glued brush onto the workspace.
- Neither `cancel` nor `stop` is an emergency stop: both talk to the arm over the same
  serial link. Keep the power switch reachable.

### Backlash, and why probes should be big

The SO-101's geared servos have several degrees of slack, and it dominates small moves.
A joint asked to reverse will sit still, or sag the wrong way under gravity, until the
teeth re-engage. `robot.backlash_deg` (6) is the allowance for it, and it is the first
thing to raise for an arm with more play.

The tracking limit is therefore `max_tracking_error_deg` (6) plus `backlash_deg` plus
0.2 seconds of commanded travel -- the servo's following lag. What is left over
is a standing disagreement between command and reality, which is a fault. A probe that
stops on `tracking error` while the arm was still moving wants a bigger allowance, not a
smaller move: shrinking the move puts it further inside the slack, where nothing the arm
does is informative.

`move-to` and `recover` report `backlash_warning` when no joint in the batch turns
further than `backlash_deg`, because such a move tells you nothing: the arm may not move
at all, or may take up slack the wrong way first.

Direction is checked before that limit, but only against travel the slack cannot explain:
a joint stops the batch as `inverted_joints` when it keeps moving away from a command
larger than `backlash_deg` and is still keeping pace with it. Slack plateaus at the size
of the gap; a wrong `robot.joint_signs` tracks the command all the way. Fix the sign,
confirm with `so-paint calibrate`, and only then retry.

The controller defaults to 50 Hz (`sample_hz`, minimum 30). Straight, same-orientation
stroke waypoints with no dwell are traversed continuously; corners and holds still stop.
Execution reports `measured_command_hz` and `max_command_gap_s` from actual send times.
The saved workspace speed and acceleration limits determine how fast the arm travels.

If each control cycle costs more than `1 / sample_hz`, the replay simply runs slower than
the timed plan -- no sample is skipped, and `elapsed_s` is the time it actually took.
`start_tolerance_deg` (4) is how far the measured pose may sit from the planned start
before a batch is refused.

## Getting unstuck: `recover`

A measured pose outside the URDF limits is normal -- a real arm's travel is wider than
the model's -- and it blocks planning, because the planner needs a start pose the model
can hold. `uv run so-paint recover` is the way out, and it belongs to the agent, not the
user: it nudges only the offending joints just inside their limits, lifts the brush to
hover height if that would otherwise leave the tip under the table, and does both slowly,
in joint space, with no IK reaching and no more than 20 degrees of joint motion per call.
Preview it first and read `tip_travel_mm` and `min_tip_z_m`; a brush resting against
something still drags. It is repeatable: each call moves a nudge, `tip_clear` says whether
it is done, and it reports when it cannot improve the pose on its own. An excursion far
beyond the limits is refused instead -- that is a wrong mapping, and a person should look.

What it does not do: force or contact sensing, mesh collision checking, obstacle routing,
and any verification that paint reached the paper. The painting raster on hardware is
`predicted_painting.png`, a prediction from the commanded strokes. Only the camera views
are evidence. Validate with supervised hover motions before any contact.

A station may specify `immersion_m` (default zero, maximum 20 mm) for bristle
insertion below its estimated contact surface. It applies only within that opening;
rim, paper, and outside-table guards remain active. Establish it from observed contact.

`paper_contact_depth_m` (default zero, maximum 10 mm) allows calibrated bristle
compression inside the paper boundary. Targets below that allowance or outside the
paper remain rejected. Tune it from camera-observed bristle contact and paint transfer.

Recovery searches lift heights at 5 mm spacing. If a fixed brush angle blocks a lift,
it may relax wrist angle with pan/roll fixed, validating an upward path within 3 mm
of the initial horizontal position. `recover --elbow-lift --preview` is an explicit
alternative for an elbow-only upward arc; inspect its horizontal travel and camera
clearance before execution. It retains the 20 degree cap and rejects downward arcs.

`table_z` locates the tabletop in robot-base coordinates (default 0). It may
be negative when the robot base origin is above the tabletop. Keep
`table_z <= paper_z < hover_z`; the tip and moving-joint table guards use
this same reference. Estimate the surface from observed contact, separately
from the brush attachment, and retain an inset paper boundary.

A paint station may explicitly set `mixing_well: true` to accept a brush already loaded with another color. Other wells still require washing before a color change. Its `color_rgb` describes the expected mixture for the predicted raster; inspect the real paint to judge the actual mixture. Rim, depth, dwell and motion checks still apply.
