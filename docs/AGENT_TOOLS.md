# Agent tools: generate, preview, execute, inspect

All commands return JSON. `station` and `draw` only generate pose files; they do not
connect to hardware. Use the same `--config` that the running service uses. Angles below
are simulation examples, not suitable defaults for an arbitrary glued brush.

## Start or reuse a session

```sh
uv run so-paint --config examples/workspace.json --state runs/demo-server.json serve
# From another shell:
uv run so-paint --state runs/demo-server.json status
uv run so-paint --state runs/demo-server.json look-at
```

The rest of these examples use the default state `runs/server.json`; pass `--state`
consistently if you started a named session. Physical settings belong in `workspace.json`.
A stale connection file is not proof that the arm is idle. Check the owning process;
do not start two services for one arm.

## Center first, then load

```sh
uv run so-paint station red approach --rpy '[3.141592653589793,0,0]' > runs/approach.json
uv run so-paint move-to --poses runs/approach.json
uv run so-paint look-at
# OPEN the physical camera image and check actual bristles over the well center.
uv run so-paint station red dip --rpy '[3.141592653589793,0,0]' > runs/dip.json
uv run so-paint move-to --poses runs/dip.json --preview
uv run so-paint move-to --poses runs/dip.json > runs/dip-report.json
uv run so-paint look-at
uv run so-paint review runs/dip-report.json
# OPEN insertion frames before using this load.
```

An approach ends lifted at `hover_z`; a dip descends vertically to
`station.center[2] - station.immersion_m`, holds and lifts. A camera check is an agent
responsibility: these generators cannot determine whether you inspected the image.
Ordinary `move-to` still enforces an observation after every executed batch.

Wash with `station washer approach`, inspect, then `station washer wash --cycles 3`.
The wash uses repeated horizontal swipes inside the configured opening, holds at the
configured depth, then lifts. Verify the washer's actual bottom, brush spread and pigment
release. The planner can reject an unreachable wrist angle or deep swipe; change the
validated geometry/orientation based on evidence, not by bypassing checks.

## Batch same-color strokes

Save a JSON list of polylines, for example `runs/strokes.json`:

```json
[[[0.2,0.7],[0.3,0.3]], [[0.45,0.8],[0.5,0.4]], [[0.7,0.75],[0.65,0.3]]]
```

```sh
uv run so-paint draw --strokes runs/strokes.json --normalized --rpy '[3.141592653589793,0,0]' > runs/drawing.json
uv run so-paint move-to --poses runs/drawing.json --preview
uv run so-paint move-to --poses runs/drawing.json
uv run so-paint look-at
```

Normalized coordinates map the square corners `(0,0),(1,0),(1,1),(0,1)` to the saved paper
corners in their listed order, not necessarily the camera's left/right. Without
`--normalized`, coordinates are robot-base metres. Leave space for bristle spread and
masking tape. The planner checks paper margins, reachability, paint capacity and timing;
a generated file is not yet a validated trajectory. Each polyline is continuous; the
brush lifts between polylines. Split batches exceeding 128 poses or the duration cap.

## Evidence and calibration

Every `look-at` returns stable image paths and `observation_file`. Raw images retain camera
resolution. `image` is a combined overview; `alignment_image` and `scene_image` are model
projections/reconstruction, not independent evidence.

With recording enabled, motion camera JPEGs and `frames.jsonl` live under the run's
`motion-frames/`. `review REPORT.json` finds the nearest captured frame per camera to the
lowest commanded point of each phase within the executed portion. It reports time gaps;
inspect the whole index for repeated dips/strokes or when the candidate misses contact.
No frames means no evidence, not a successful load. Archives consume disk space; delete
completed run directories when their evidence is no longer needed. Hardware recording
also requires `record_cameras_during_motion: true`.

For `calibrate-camera`, add `validation_robot_points` and `validation_image_points` to the
existing fit input. These independent XYZ/pixel pairs (at least three) are checked without
refitting. Failed validation returns `calibrated: false`. Use a spread of heights and wrist
angles and assess raw image alignment before contact; low training residuals are insufficient.
