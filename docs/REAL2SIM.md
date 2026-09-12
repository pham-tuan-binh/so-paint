# Match the physical workspace to the model

The useful scene is small: paper boundary/surface, paint openings/surfaces, washer
opening/depth, arm base and brush tip. Geometry is configurable in `workspace.json`.

**The motors are the only thing a human calibrates.** Everything past that -- where the
paper is, which well holds which colour, how deep the washer is, where the brush tip
sits -- is the agent's guess from the camera, corrected by iteration. Do not ask the user
to measure the table, print a fiducial board, click on corners or label the paints. There
is no automatic semantic detector hidden in the server and no manual corner-click UI:
the host model looks at the picture, and the arm supplies the scale.

## Reconstruct it zero-shot, then let the arm supply the scale

Start by writing down what you see. From one `look-at` image -- a single mono webcam is
enough -- estimate the paper polygon, each well's centre and opening radius, the washer,
the rim and surface heights, and the brush tip, in robot-base metres. That estimate is a
hypothesis, and a monocular view has no inherent scale, so expect it to be wrong by
centimetres. Write it into `workspace.json` and `reload` anyway: it replaces the demo
fixture, and the loop below is what makes it metric.

The scale reference you need is already in the picture. **The brush tip at a known robot
pose is a fiducial you own, can move, and can re-observe.** `look-at` reports
`robot.tip_xyz` in robot-base metres, from forward kinematics of the measured joints, and
the same frame shows where that tip actually is in pixels. Six or more of those pairs
register the camera to the robot base with no external target at all:

1. `move_to` a hover pose well above everything -- z of 0.10 m or more clears the wells
   and the paper, so a wrong geometry estimate cannot drive the brush into anything.
2. `look-at`, find the brush tip in the raw frame, and record that pixel together with
   the reported `robot.tip_xyz`.
3. Repeat across the working area **and at several different heights**. Coplanar points
   leave the fit under-determined, and an unknown focal length hides in a flat spread.

Then fit the camera. `calibrate-camera` accepts a JSON file containing:

```json
{
  "camera": {"name":"front","source":"opencv","device":0,"width":640,"height":480,"focal_px":700},
  "robot_points": [[0.10,-0.06,0],[0.20,-0.06,0],[0.30,-0.06,0],[0.10,0.06,0],[0.20,0.06,0],[0.30,0.06,0]],
  "image_points": [[218,316],[345,316],[473,316],[218,163],[345,163],[473,163]]
}
```

The numbers illustrate the schema; do not use them as calibration. `robot_points` are the
tip positions the arm actually visited and `image_points` are where you saw the tip.

```sh
uv run so-paint calibrate-camera tip-observations.json
```

It prints a camera configuration and reprojection residuals, and refuses collinear
landmarks, points behind the camera or excessive pixel error. `focal_px` is a guess for
an uncalibrated webcam, and a wrong one is absorbed into the fitted pose: run the fit
across a few candidate focal lengths and keep the lowest residual, or supply a measured
`intrinsic_matrix` and `distortion` if the lens is characterised. Copy the fitted camera
object into the workspace config. OpenCV images from calibrated cameras are undistorted;
later detections use those undistorted original-size pixels.

A good fit residual is not validation -- it only says the fit explains the points it was
given. Hover to two or three poses you did not fit to and check that the predicted tip
pixel lands on the real one. Then re-detect the paper and stations through the registered
camera and update the geometry; that is where the centimetres come out.

## Turn detections into coordinates

With a registered camera, the host vision model's pixel detections become robot metres.
For each paper corner / station center, provide its estimated plane height and pixels:

```json
{
  "paper_top_left": {"z_m":0.005,"pixels":{"overhead":[280,195],"side":[302,244]}},
  "red_center": {"z_m":0.012,"pixels":{"overhead":[260,330]}}
}
```

Again these are schema examples, not measurements. Heights are still your estimate: the
utility intersects a ray with the plane you name, so a wrong height moves the point along
the ray. Run:

```sh
uv run so-paint --config workspace.json reconstruct detections.json
```

The utility intersects camera rays with the specified surface plane, combines views,
and rejects more than 3 mm disagreement from their median. It reports view count;
a single view is not independent cross-validation. It does not recover unknown surface
heights, distinguish similarly colored liquids, or estimate unseen vessel depth.
Use the output to update paper/station coordinates, estimate opening radii and heights,
then verify their projections and brush approaches. The model performs semantic
recognition; the server supplies deterministic geometry.

## Iterate the three primitives

1. **Clean:** hover over the washer → vertical descent inside its rim → small rinse
   movement and dwell → vertical withdrawal. Inspect residual color and clearance.
2. **Load:** hover over the chosen color → vertical descent → brief contact/dwell →
   withdrawal. Inspect loading and the first test stroke. Wash before switching colors.
3. **Stroke:** approach above its start → lower → follow the stroke → lift. Compare
   actual footprint to intended position, width and coverage; update estimates and repeat.

`look_at` gives raw views and a separate projected overlay. If both were generated from
the same incorrect transform, their predicted lines agreeing with each other proves
nothing: compare them with actual image evidence. Raw real frames must never be replaced
with rendered frames for validation. The simulated fixture is labeled as synthetic.

Use a few short probes instead of a minute-long uncertain first move. A minute is the
maximum planning horizon, not a minimum. After the primitives are reliable, compose
longer batches and keep observing between them. Never fit all geometric errors into
brush offset: shared error across the whole table suggests registration; error that
changes with wrist orientation suggests tool geometry; contact-only error suggests
brush bending, height or paint behavior.
