"""Three atomic actions. They only produce poses; all execution goes through move_to.

The host model may build these sequences itself, or the onboarding agent can use
these helpers to generate JSON. No extra robot tools or hidden motion commands.
"""

from .models import Pose


def at(xyz, hold_s=0):
    return Pose(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]), hold_s=hold_s)


def clean_brush(world):
    station = next(s for s in world.stations if s.kind == "washer")
    x, y, z = station.center
    h = world.settings.hover_z
    # Lift/translate, descend vertically, small rinse motion, dwell, withdraw.
    r = min(0.002, station.radius_m / 5)
    return [
        at((x, y, h)),
        at((x, y, z)),
        at((x + r, y, z)),
        at((x, y + r, z)),
        at((x - r, y, z)),
        at((x, y - r, z)),
        at((x, y, z), world.settings.wash_dwell_s + 0.1),
        at((x, y, h)),
    ]


def load_color(world, color, *, clean=True):
    station = next((s for s in world.stations if s.name == color and s.kind == "paint"), None)
    if station is None:
        raise ValueError(f"Unknown paint color {color}")
    x, y, z = station.center
    return (clean_brush(world) if clean else []) + [
        at((x, y, world.settings.hover_z)),
        at((x, y, z), world.settings.load_dwell_s + 0.1),
        at((x, y, world.settings.hover_z)),
    ]


def brush_stroke(world, points_xy):
    if len(points_xy) < 2:
        raise ValueError("A stroke needs at least two points")
    z, h = world.settings.paper_z, world.settings.hover_z
    return [at((*points_xy[0], h))] + [at((*p, z)) for p in points_xy] + [at((*points_xy[-1], h))]


def station_action(world, name, action, *, rpy, cycles=3):
    """Generate one reviewable station action; never execute or combine load + paint.

    Approach ends at the globally safe hover for a live center check. Dip/wash use
    the separately calibrated contact surface and immersion, never paper pressure.
    """
    station = next((s for s in world.stations if s.name == name), None)
    if station is None:
        raise ValueError(f"Unknown station {name}")
    if action not in {"approach", "dip", "wash"}:
        raise ValueError("Station action must be approach, dip or wash")
    if action == "wash" and station.kind != "washer":
        raise ValueError("Wash requires a washer")
    if action == "dip" and station.kind != "paint":
        raise ValueError("Dip requires a paint well")
    if not isinstance(cycles, int) or not 1 <= cycles <= 10:
        raise ValueError("Wash cycles must be between 1 and 10")
    x, y, surface = station.center
    z, h = surface - station.immersion_m, world.settings.hover_z
    points = [at((x, y, h), 1)]
    if action != "approach":
        points.append(at((x, y, z), 1))
        if action == "wash":
            # Decisive swipes, leaving room for bristles and the planner's rim guard.
            radius = min(station.radius_m * .5,
                         station.radius_m - max(.004, world.settings.brush_width_m))
            if radius <= 0:
                raise ValueError("Washer too narrow for this brush")
            for _ in range(cycles):
                points.extend([at((x - radius, y, z)), at((x + radius, y, z))])
        dwell = world.settings.wash_dwell_s if action == "wash" else world.settings.load_dwell_s
        points.extend([at((x, y, z), dwell + .1), at((x, y, h))])
    return orient(points, rpy)


def orient(poses, rpy):
    """Validate orientations rather than silently inheriting the demo's orientation."""
    import numpy as np

    angles = np.asarray(rpy, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("rpy needs three finite radians")
    return [Pose(**(p.model_dump() | dict(zip(("roll", "pitch", "yaw"), angles)))) for p in poses]


def drawing(world, strokes, *, rpy, normalized=False):
    """Bundle same-color polylines; lift between strokes and at the end.

    Normalized coordinates map [0,1]^2 to the four saved paper corners in their
    listed order. All bounds/contact/reachability remain the planner's responsibility.
    """
    import cv2
    import numpy as np

    if not strokes:
        raise ValueError("A drawing needs at least one stroke")
    poses = []
    for stroke in strokes:
        xy = np.asarray(stroke, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError("Each stroke needs at least two finite XY points")
        if normalized:
            if np.any(xy < 0) or np.any(xy > 1):
                raise ValueError("Normalized coordinates must be in [0,1]")
            xy = cv2.perspectiveTransform(
                (xy * 511).astype(np.float64)[None], np.linalg.inv(world.to_canvas)
            )[0]
        poses.extend(brush_stroke(world, xy))
    if len(poses) > 128:
        raise ValueError("Drawing exceeds 128 poses; split at a lifted stroke boundary")
    return orient(poses, rpy)
