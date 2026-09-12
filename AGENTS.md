# Start here

This is an agent-operated SO-101 painting workbench built with **uv + Python**.
The user should be able to paste this repository into chat and say what to paint.
The normal interface is a local CLI: `serve` owns the session; `look-at` and `move-to`
are the painting operations. Atomic actions are ordinary pose sequences.

## First response / setup

1. Read `README.md`. Run `uv sync --locked` (add `--extra hardware` for a real arm) and `uv run so-paint doctor`.
2. If the user wants simulation, run `uv run so-paint --config examples/workspace.json demo`; start the persistent service
   with `uv run so-paint serve` in a background shell session. Call the CLI from another shell.
3. If a saved setup exists, use `uv run so-paint --config workspace.json doctor`.
   Do not replace it or rerun calibration merely because this is a new conversation.
4. If the user asks for a physical arm, read `docs/SETUP.md`. Run
   `uv run so-paint calibrate`: it reports the LeRobot calibration state without opening
   the port and returns a `prompt` list of the physical steps only the user can do.
   Read those steps to them, wait, then re-run it. Never run LeRobot's calibration for
   them non-interactively, and never re-record an existing calibration. The driver has been exercised on one physical setup; for an unverified setup start
   with hover probes, and do
   not pretend simulated state is a hardware observation.
5. Ask only for information that is actually missing. Never ask the user to manually
   click paper corners or label paint wells: use the model's visual capabilities.

## Once ready: no ceremony

The user says “paint a sunset”. Call `look-at`; inspect the raw combined camera image.
Find the paper, available colors and washer. The third image, if present, is the
Rerun-rendered URDF/workspace reconstruction, not a physical camera observation. Check the alignment overlay against the
raw views. Then plan and issue `move_to` batches. Do not ask the user to convert their
idea to strokes or approve each ordinary batch. Inspect with `look-at` after each
batch and improve the next one. The host model supplies vision, artistic decisions
and iterative reasoning; the local service does not need an API key or nested LLM.

## Keep the representation small

- Where to paint: paper boundary and surface in robot-base coordinates.
- Where to clean: washer opening, rim and usable contact depth.
- Where to load: color identity, well opening, rim and paint surface.
- Brush: estimated tip transform, direction, width, contact height, load capacity.
- Actions: lift / clean / load / brush stroke. See `src/so_paint/recipes.py` for examples.

Arbitrary brushes, cameras and environments will differ from the demo. Reconstruct the
scene zero-shot from whatever cameras exist -- one mono view is enough to start -- and
never ask the user to measure, print a target or click a corner. Then let the arm supply
the scale: the brush tip at a known robot pose is a fiducial you can move and re-observe,
so hover probes both register the camera and check the result. Treat geometric and brush
estimates as hypotheses until predicted and observed tip pixels agree. Do not turn
uncertain visual guesses into confident metric calibration, and do not absorb a
registration error into the brush offset. Then test a short stroke and tune
contact/width/speed/load. Ready setups skip this discovery work. See `docs/REAL2SIM.md`.

## Motion contract

`move_to(poses=[{x,y,z,roll,pitch,yaw,hold_s}], preview=false,
orientation_mode="brush_axis")`: robot-base metres, XYZ Euler radians, **brush-tip**
frame. The SO-101 has five positioning joints. `full` enforces all six pose components;
`brush_axis` frees rotation about a round brush. Use full orientation for a flat brush
when its edge orientation matters; reposition the work if that pose is unreachable.

The planner checks the whole batch before simulation execution. It times the path to
joint speed/acceleration limits and a configurable maximum (`max_batch_duration_s`, default 60 s, maximum 300 s); it never silently truncates.
End lifted. Explicitly insert vertical station approaches and lift before transfers.
After each executed batch, `look_at` is required before the next. A failed preflight
moves nothing. Split a long batch at a lifted pose and inspect before continuing.

## The arm has backlash

The SO-101's geared servos have several degrees of slack. On any direction change a joint
sits still, or sags the wrong way under load, until the teeth re-engage. Consequences:

- Small probe moves are unreliable and prove nothing. Probe with decisive moves and read
  the result once the arm has settled, rather than inferring from a millimetre nudge.
- A joint travelling the wrong way at the start of a move is slack, not a sign error. The
  executor allows for `robot.backlash_deg` (6 by default) in both its tracking limit and
  its direction check, and only calls a sign fault when wrong-way travel keeps pace with a
  command larger than that. Slack runs out; a wrong sign does not.
- A stop on `tracking error` while the arm was still moving means the allowance is too
  tight for this arm. Raise `robot.backlash_deg` or `max_tracking_error_deg`. Do not make
  the move smaller: that makes it worse.
- Expect the first contact of any stroke to lag the command, and prefer one decisive
  stroke over several timid ones.

## Judgement while iterating

Iterate on what the cameras show. Tracking error, feedback latency and reprojection
residuals exist to catch faults; they are not scores to improve, and a batch that
completed with a few degrees of following error tells you nothing about your painting.
When a probe moves and the image shows something, act on the image. Cameras are opened
fresh for every `look_at` and retried, and the motion-time logger reopens a camera that
drops out, so a camera error is something to look again at, not to stop for.

An arm outside the model's joint limits or with the brush low is a normal outcome of a
probe, not an incident: `recover` nudges it back and lifts the brush, a little at a time,
and is safe to repeat. Use it. Never ask the user to move the arm by hand -- the arm is
the one thing here you can actually control.

Stop and involve the user only for a fault they can act on: a joint that moves opposite
its command (a wrong `robot.joint_signs`, which the executor names for you), feedback
that is impossible for this arm, or hardware that will not respond. Say which joint and
what you measured. Diagnosing motor internals is not your job; getting a picture,
adjusting an estimate and trying a smaller move is.

## Physical execution

`backend: "lerobot"` drives a real SO-101. `serve` still opens nothing: the port opens on
the first `look-at`, which reports measured encoder positions. `move-to` replays the
prevalidated trajectory and stops in place on tracking error, stale feedback, a device
fault or `cancel`; `status` and `cancel` stay answerable while the arm moves. A batch may
stop partway: read `completed`, `stop_reason` and `commanded_samples`, and always
`look-at` before deciding what to do next -- never resend a partially executed batch.
On hardware the painting raster is a prediction (`predicted_painting.png`); only the
camera views show whether paint reached the paper.

## Logging

For hardware, record commands, encoder feedback and camera frames continuously during
motion with `recorder.execution`; see `docs/TELEMETRY.md`. Use acquisition timestamps
from one session clock. Never label commanded or simulated positions as measurements.
Keep the URDF driven by feedback and log observations independently of `look_at` calls.

## Development

- Use `uv`; `uv run pytest`, `uv run ruff check .`, `uv run so-paint --config examples/workspace.json demo`.
- Commands return JSON; open the absolute image paths returned by `look-at`.
- Save robot port/calibration path and camera inventory in `workspace.json`; select feeds
  with `look_at_cameras`. Edit and `reload` the running setup, then inspect `look-at`.
- Keep simulation/estimated geometry and physical observations distinguishable.
- Do not introduce a web UI, another AI orchestrator, or MuJoCo without a concrete need.
- Preserve existing workspace calibration. No hardware motor motion during ordinary
  tests; no power, torque, or serial connection on server import/startup.

## Brush attachment

The user glues the brush to the closed gripper, with its shaft perpendicular to the
length of the gripper. Keep the gripper closed throughout painting; never use it to
pick up, release or change brushes. `robot.gripper` accepts only `"closed"`; there is
no gripper-opening action. The hardware driver holds the measured
`robot.gripper_closed_pct` and reports the encoder value it reads back, rather than
assuming zero motor units means closed.

The simulation fixture uses a 90-degree mount rotation (`brush_mount_rpy`), a 100 mm
shaft (`brush_length_m`), and `brush_tip_offset` in the URDF `gripper_frame_link` frame.
These dimensions describe the demo only. Estimate the actual glued attachment and tip
from images and refine with hover checks. Pose orientations describe the brush frame;
IK applies the mounting rotation to the gripper. Reuse saved mount calibration.

The attachment and bristles can shift during a stroke or wash. For every paint load,
inspect the actual bristle tip at full insertion in the physical camera frames before
using that load. Check that it enters the center of the well and reaches the paint;
an aligned shaft, hover pose, or projected marker does not establish either. Correct
the horizontal position first, then adjust insertion depth from observed contact.
Use the motion recording for the dip and `look-at` after the lifted batch. Keep paint
loading depth separate from paper contact pressure, and leave room for bristle spread
inside the white paper boundary, away from masking tape.

Use the washer between colors and when dried paint or clumps impede loading. Inspect
submersion and the next paint transfer; stained bristles alone do not prove a usable
load. Where visible, a separated tip and shadow help identify hovering; bristle contact
and actual paint transfer are stronger evidence than the shadow alone.

## Efficient agent workflow

Read [docs/AGENT_TOOLS.md](docs/AGENT_TOOLS.md) for runnable recipes. Prefer `station`
and `draw` to hand-generating repetitive pose JSON. These tools generate poses only;
`move-to` still performs full preflight, and `look-at` is required after execution.
Use an explicit, visually validated brush `--rpy`; demo orientations are not a hardware
calibration. Reuse each station's calibrated geometry and immersion independently.

Before EVERY paint load, execute a separate `station NAME approach`, pause and inspect
`look-at` to verify the actual tip is above the center of the intended opening. Correct
horizontal alignment before pushing in. Then execute `station NAME dip`, inspect
`look-at` AND full insertion frames via `review`, and only then paint. A pause inside
a combined hover/dip batch does not let the agent check the center before insertion.
If the global hover view is insufficient, use recorded closer hover probes that still
end lifted; do not bypass the end-lift guard or guess from a projected marker.

For washing, approach and check the washer center first. Reach its verified usable
bottom/contact depth and swipe back and forth several times; soaking alone was inadequate
in the physical session. Inspect pigment release and the subsequent paint transfer.
Some setups have washer and paint-slot bottoms at the same level; save that only when
established for the actual setup, never assume it globally. Keep depth separate from
paper contact pressure. Honor a user's report that they have manually cleaned the brush.

Once motion and deposition work, group useful same-color strokes into longer batches.
Use preview duration to size them to the configured limit; split at lifted stroke
boundaries. Do not continually make tiny probes or repeat already successful checks.
The duration cap may be raised for a reliable setup (maximum 300 s); do not disable
tracking, rim, contact or review checks to gain speed.

Open raw full-resolution `cameras.*.image`, not just the combined thumbnail. Observation
paths are immutable. `motion_frames` in reports points to acquisition-timestamped images;
`review` selects candidates, not proof of contact. Check time gaps and use the full index
when a selected frame misses insertion. Camera failures require another observation.

Before painting, define visible completion criteria for the requested composition. For
a flower meadow: distributed flowers at varied scales, visible stems/grass, intentional
coverage and readable flower shapes. Inspect these in the real paper at the end. A
successful trajectory, predicted raster or a few sparse marks is not a finished meadow.

Keep setup-specific values and concise handoff notes in ignored `workspace.json` and
`LOCAL_NOTES.md`: last verified poses/orientations, unresolved registration hypotheses,
last observed brush state, completed composition layers, next action and session state
path. Never put a private port, calibration path or chronological session log in this
public prompt. Check `status` on the recorded state path before starting another service;
never delete a state file or open a competing hardware session without checking ownership.

Camera calibration output distinguishes fitting residuals from held-out validation.
Supply independent `validation_robot_points` and `validation_image_points`; a failed
check marks the returned camera uncalibrated. Even a passed check does not certify brush
geometry across wrist angles or establish contact depth. Preserve existing calibration
until a replacement has better independent visual evidence.
