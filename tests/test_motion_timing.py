import numpy as np
import pytest
from pydantic import ValidationError

from so_paint.models import Settings
from so_paint.planner import continuous_waypoints
from so_paint.recipes import at
from so_paint.workbench import Workbench


def test_straight_stroke_has_no_internal_stop(tmp_path):
    with_bench = Workbench(output_dir=tmp_path, record=False)
    try:
        w = with_bench
        poses = [at((x, 0.01, w.settings.hover_z)) for x in (.16, .18, .20)]
        continuous = w.planner.plan(poses, w.q, 0)
        stopped_poses = [p.model_copy() for p in poses]
        stopped_poses[1].hold_s = .1
        stopped = w.planner.plan(stopped_poses, w.q, 0)
        assert continuous.duration_s < stopped.duration_s
        samples = continuous.samples
        index = min(range(len(samples)), key=lambda i: np.linalg.norm(np.array(samples[i].tip) - poses[1].xyz()))
        speed = np.linalg.norm(np.array(samples[index+1].tip) - samples[index-1].tip) / (samples[index+1].t - samples[index-1].t)
        assert speed > .005
        assert np.max(np.diff([s.t for s in samples])) <= 1 / 30 + 1e-9
    finally:
        with_bench.close()


def test_preserves_corner_hold_reversal_and_orientation():
    a, b, c = [at((x, .01, .028)) for x in (.16, .18, .20)]
    assert continuous_waypoints([a, b, c]) == [a, c]
    for end in [a, c.model_copy(update={'y': .02}), c.model_copy(update={'yaw': .1})]:
        assert len(continuous_waypoints([a, b, end])) == 3
    assert len(continuous_waypoints([a, b.model_copy(update={'hold_s': .1}), c])) == 3


def test_control_rate_minimum():
    assert Settings().sample_hz == 50
    with pytest.raises(ValidationError):
        Settings(sample_hz=20)


def test_immersion_only_inside_configured_well():
    from so_paint.world import World
    settings = Settings()
    station = settings.workspace.stations[0]
    station.center = (*station.center[:2], 0)
    station.immersion_m = .005
    world = World(settings)
    x, y, _ = station.center
    assert world.classify((x, y, -.005)) == ('load', station.name)
    with pytest.raises(ValueError, match='contact surface'):
        world.classify((x, y, -.007))
    with pytest.raises(ValueError, match='rim'):
        world.classify((x + station.radius_m - .001, y, -.005))
    with pytest.raises(ValueError, match='below the table'):
        world.classify((.18, 0, -.005))


def test_paper_contact_depth_preserves_edges_and_depth_limit():
    from so_paint.world import World
    world = World(Settings(paper_z=0, paper_contact_depth_m=.006))
    assert world.classify((.18, 0, -.006)) == ('paint', None)
    with pytest.raises(ValueError, match='penetrates'):
        world.classify((.18, 0, -.008))
    with pytest.raises(ValueError, match='below the table'):
        world.classify((.1, 0, -.006))


def test_table_height_in_robot_base_coordinates():
    from so_paint.world import World
    world = World(Settings(table_z=-.012, paper_z=-.012, paper_contact_depth_m=.008))
    assert world.classify((.18, 0, -.02)) == ('paint', None)
    with pytest.raises(ValueError, match='penetrates'):
        world.classify((.18, 0, -.023))
    with pytest.raises(ValueError, match='below the table'):
        world.classify((.1, 0, -.02))
    with pytest.raises(ValidationError, match='table_z'):
        Settings(table_z=.01, paper_z=0)


def test_recovery_finds_small_lift_with_high_hover():
    from types import SimpleNamespace

    from so_paint.kinematics import Arm

    settings = Settings(hover_z=.13, brush_tip_offset=(.08397, .01303, .01451))
    context = SimpleNamespace(settings=settings, arm=Arm(settings))
    q = np.array([.02915, -.49483, 1.59036, -.82702, .04219])
    target, _ = Workbench._recovery_target(context, q, q.copy())
    assert np.rad2deg(abs(target-q)).max() <= 20
    assert context.arm.fk(target)[0][2] > context.arm.fk(q)[0][2] + .005
