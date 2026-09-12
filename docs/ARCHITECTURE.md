# Implementation map

`cli.py` sends look/move requests to the persistent loopback `service.py`.
Setup commands report status, guide motor calibration, reload the config, cancel a
running physical batch, and close the session. The host model translates arbitrary
painting descriptions into action sequences and interprets feedback; there is no nested
LLM, prompt-to-image dependency or semantic planner service.

`workbench.py` owns one serialized session, on the simulation or on a physical arm. It preflights a complete batch,
checks cleaning/loading state on a copy, then commits execution: synthetic deposition in
simulation, measured encoder state on hardware. `look_at`
returns combined cameras and projected scene/waypoint alignment. All selected look-at cameras must
succeed to release the next execution boundary. Sessions persist artifacts, not resumable
motor state; a restart re-measures the arm rather than assuming where it was left. Config reload preserves the current
joints, brush, paper image and time while starting a new recording segment.

`kinematics.py` parses the vendor URDF chain and uses bounded least squares for tip IK.
`planner.py` uses quintic progress per Cartesian segment, seeded IK and clamped joint
splines, time-scaled against velocity/acceleration bounds. The planner stops at each
waypoint. It optimizes segment duration, not global stroke ordering or coverage; those
choices belong to the model. Full orientation or position+brush-axis constraints are
available, accounting for the five arm joints.

`world.py` represents configurable paper and stations and checks tip contact geometry.
`brush.py` models rinse/load dwell and finite paint capacity. `recipes.py` composes the
three atomic actions as ordinary poses. `cameras.py` handles arbitrary configured views
and projects scene points. `registration.py` fits camera extrinsics and intersects
model-supplied pixel rays with known planes. `recording.py` loads the bundled URDF and meshes via `rerun.urdf.UrdfTree`, animates
named joint transforms, and exports Rerun recordings. When enabled, an owned headless
viewer receives the same stream and `ViewerClient.save_screenshot(view_id=...)` produces
a PNG for `look_at`. The viewer screenshot is a reconstruction; camera observations
are captured/generated independently. Renderer failures are reported without blocking
camera capture, and the viewer process is closed when the session ends.

`telemetry.py` supplies thread-serialized command, feedback and camera callbacks on a
shared acquisition-time timeline. Commands and measured/simulated states are separate;
only feedback animates the robot. The simulated executor logs every joint sample and
camera frames during motion, and the LeRobot driver calls the same hooks with encoder
feedback. See `TELEMETRY.md`.

`calibration.py` finds and parses the arm's LeRobot motor calibration, converts its
recorded travel and units into URDF radians, reports what disagrees with the model, and
renders the calibration steps the user has to perform. It opens no port and never writes
a calibration file. `hardware.py` is the `lerobot` backend: an explicit connect that holds
the arm's present pose rather than a stale goal, encoder reads, setpoints that always hold
the measured closed-gripper position, and a trajectory replay that stops in place on
cancellation, tracking error, stale feedback or a device fault. Physical execution runs
outside the session lock so `status` and `cancel` stay answerable, and it commits the
measured final pose rather than the last setpoint. See `SETUP.md`.

No closed-loop servo, force sensing, mesh collision checker, fluid simulation, automatic
brush parameter fitting or automatic semantic perception service is implemented, and the
LeRobot backend has been exercised on one physical SO-101 setup; its
execution path is tested against a stubbed follower. Visual interpretation/refinement is
done by the host agent, with editable settings and registration utilities. Hardware
onboarding is documented separately.
