import copy
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from so_paint.models import Pose, Settings
from so_paint.recipes import at, brush_stroke, load_color
from so_paint.workbench import Workbench


@pytest.fixture
def bench(tmp_path):
    w = Workbench(output_dir=tmp_path, record=False)
    yield w
    w.close()


def test_clean_load_stroke_and_review(bench):
    observation = bench.look_at()
    assert set(observation["cameras"]) == {"overhead", "side"}
    before = bench.world.canvas.copy()
    loaded = bench.move_to(load_color(bench.world, "red"))
    assert 0 < loaded["duration_s"] <= 60
    assert bench.brush.color == "red"
    assert np.array_equal(before, bench.world.canvas)
    poses = brush_stroke(bench.world, [(0.16, 0.01), (0.21, 0.01)])
    with pytest.raises(ValueError, match="look_at"):
        bench.move_to(poses)
    bench.look_at()
    report = bench.move_to(poses)
    assert report["changed_canvas_pixels"] > 100
    assert report["planned_vs_painted_mean_abs_rgb"] > 0
    samples = json.loads(Path(report["trajectory"]).read_text())["samples"]
    q = np.array([s["joints"] for s in samples])
    t = np.array([s["t"] for s in samples])
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    assert np.max(np.abs(v)) <= bench.settings.max_joint_speed * 1.01
    assert t[-1] <= 60
    assert samples[-1]["tip"][2] >= bench.settings.hover_z - 0.0015


def test_preview_and_rejected_batch_are_nonmutating(bench):
    bench.look_at()
    q, brush, canvas = bench.q.copy(), copy.deepcopy(bench.brush), bench.world.canvas.copy()
    result = bench.move_to([at((0.18, 0.01, 0.028))], preview=True)
    assert result["preview"]
    np.testing.assert_array_equal(bench.q, q)
    with pytest.raises(ValueError, match="Unreachable"):
        bench.move_to([at((0.18, 0.01, 0.028)), at((5, 0, 0.028))])
    np.testing.assert_array_equal(bench.q, q)
    np.testing.assert_array_equal(bench.world.canvas, canvas)
    assert bench.brush == brush
    assert bench.revision == 0


def test_dry_brush_failure_is_atomic(bench):
    bench.look_at()
    with pytest.raises(ValueError, match="load a color"):
        bench.move_to(brush_stroke(bench.world, [(0.17, 0), (0.20, 0)]))
    assert bench.revision == 0
    assert np.all(bench.world.canvas == 250)


def test_color_change_requires_washing(bench):
    bench.look_at()
    bench.move_to(load_color(bench.world, "red", clean=False))
    bench.look_at()
    with pytest.raises(ValueError, match="Clean the brush"):
        bench.move_to(load_color(bench.world, "blue", clean=False))
    assert bench.brush.color == "red"
    bench.move_to(load_color(bench.world, "blue"))
    assert bench.brush.color == "blue"


def test_explicit_mixing_well_does_not_unlock_other_wells(bench):
    blue = next(s for s in bench.world.stations if s.name == "blue")
    blue.mixing_well = True
    bench.look_at()
    bench.move_to(load_color(bench.world, "red", clean=False))
    bench.look_at()
    bench.move_to(load_color(bench.world, "blue", clean=False))
    assert bench.brush.color == "blue"
    assert bench.brush.remaining_m == bench.settings.max_load_distance_m
    bench.look_at()
    with pytest.raises(ValueError, match="Clean the brush"):
        bench.move_to(load_color(bench.world, "red", clean=False))
    assert bench.brush.color == "blue"


def test_long_batch_and_unlifted_end_rejected(bench):
    bench.look_at()
    with pytest.raises(ValueError, match="60"):
        bench.move_to([at((0.18, 0, 0.028), 10)] * 7)
    with pytest.raises(ValueError, match="lifted"):
        bench.move_to([at((0.18, 0, 0.005))], preview=True)
    assert bench.revision == 0


def test_rim_table_edge_guards(bench):
    w = bench.world
    for point in [(0.145 + 0.01, -0.065, 0.015), (0.18, 0, -0.01), (0.15, 0, 0.005)]:
        with pytest.raises(ValueError):
            w.classify(point)
    assert w.classify((0.18, 0, 0.005))[0] == "paint"
    assert w.classify((0.145, -0.065, 0.012)) == ("load", "red")


def test_configurable_batch_limit_keeps_atomic_rejection_and_lift_guard(bench):
    settings = bench.settings.model_copy(update={"max_batch_duration_s": 180})
    bench.reconfigure(settings)
    bench.look_at()
    q = bench.q.copy()
    poses = [at((0.18, 0, settings.hover_z), 10)] * 7
    report = bench.move_to(poses, preview=True)
    assert 60 < report["duration_s"] <= 180
    with pytest.raises(ValueError, match="180"):
        bench.move_to(poses * 3)
    with pytest.raises(ValueError, match="lifted"):
        bench.move_to([at((0.18, 0, settings.paper_z))], preview=True)
    np.testing.assert_array_equal(bench.q, q)


@pytest.mark.parametrize("limit", [0, 301, float("inf")])
def test_invalid_batch_limit(limit):
    with pytest.raises(ValidationError):
        Settings(max_batch_duration_s=limit)


def test_full_pose_ik_round_trip_and_impossible_pose(bench):
    arm = bench.arm
    tip = arm.fk(bench.q)[0]
    orientation = arm.rotation(bench.q)
    q = arm.ik(tip, bench.q, orientation, "full")
    np.testing.assert_allclose(arm.fk(q)[0], tip, atol=0.0015)
    assert (orientation.inv() * arm.rotation(q)).magnitude() < np.deg2rad(5)
    with pytest.raises(ValueError, match="Unreachable"):
        arm.ik(tip, bench.q, Rotation.identity(), "full")


def test_failed_camera_blocks_next_motion(bench, monkeypatch):
    import so_paint.workbench as module

    original = module.capture

    def capture(c, *args):
        if c.name == "side":
            raise RuntimeError("camera unplugged")
        return original(c, *args)

    monkeypatch.setattr(module, "capture", capture)
    state = bench.look_at()
    assert "side" in state["camera_errors"]
    with pytest.raises(ValueError, match="look_at"):
        bench.move_to([at((0.18, 0, 0.028))])


def test_invalid_numeric_inputs_and_duplicate_cameras():
    with pytest.raises(ValidationError):
        Pose(x=float("nan"), y=0, z=0)
    settings = Settings().model_dump()
    settings["cameras"][1]["name"] = "overhead"
    with pytest.raises(ValidationError):
        Settings.model_validate(settings)


def test_workspace_geometry_is_configurable(tmp_path):
    s = Settings()
    s.workspace.paper_corners_xy = [(0.15, 0.03), (0.22, 0.02), (0.22, -0.03), (0.15, -0.02)]
    w = Workbench(s, tmp_path, record=False)
    try:
        np.testing.assert_allclose(w.world.paper_xy, s.workspace.paper_corners_xy)
        assert w.world.classify((0.18, 0, 0.005))[0] == "paint"
    finally:
        w.close()


def test_perpendicular_fixed_brush_mount():
    from so_paint.kinematics import Arm
    from so_paint.models import RobotConfig

    with pytest.raises(ValueError):
        RobotConfig(gripper="open")
    arm = Arm(Settings())
    q = np.zeros(5)
    gripper_rotation = arm.rotation(q) * arm.mount.inv()
    tip, direction, _ = arm.fk(q)
    assert abs(np.dot(direction, gripper_rotation.apply([0, 0, 1]))) < 1e-10
    np.testing.assert_allclose(direction, arm.rotation(q).apply([0, 0, 1]), atol=1e-10)
    solved = arm.ik(tip, q, arm.rotation(q), "full")
    np.testing.assert_allclose(arm.fk(solved)[0], tip, atol=0.0015)
