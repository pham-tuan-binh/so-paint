import json
from pathlib import Path

import pytest

from so_paint.cameras import project
from so_paint.models import Settings
from so_paint.recipes import drawing, station_action
from so_paint.registration import validate_camera
from so_paint.review import review_motion
from so_paint.workbench import Workbench
from so_paint.world import World


def test_observation_evidence_survives_next_look(tmp_path):
    w = Workbench(output_dir=tmp_path, record=False)
    try:
        first = w.look_at()
        original = Path(first['image']).read_bytes()
        second = w.look_at()
        assert first['image'] != second['image']
        assert Path(first['image']).read_bytes() == original
        assert Path(first['observation_file']).exists()
        for c in first['cameras'].values():
            from PIL import Image
            assert list(Image.open(c['image']).size) == c['pixel_size']
    finally:
        w.close()


def test_station_recipes_keep_approach_separate_and_use_calibrated_depth():
    s = Settings()
    station = s.workspace.stations[-1]
    station.immersion_m = .01
    world = World(s)
    approach = station_action(world, station.name, 'approach', rpy=[0, -.3, 0])
    assert len(approach) == 1 and approach[0].z == s.hover_z
    poses = station_action(world, station.name, 'wash', rpy=[0, -.3, 0], cycles=3)
    assert poses[-1].z == s.hover_z
    assert min(p.z for p in poses) == pytest.approx(station.center[2] - .01)
    assert all(p.pitch == -.3 for p in poses)
    assert sum(p.x != station.center[0] for p in poses) == 6
    for p in poses:
        world.classify([p.x, p.y, p.z])
    with pytest.raises(ValueError, match='washer'):
        station_action(world, 'red', 'wash', rpy=[0, 0, 0])


def test_normalized_drawing_maps_paper_and_lifts_between_strokes():
    world = World(Settings())
    poses = drawing(world, [[[.2, .2], [.8, .8]], [[.3, .4], [.7, .4]]],
                    normalized=True, rpy=[0, -.2, 0])
    assert len(poses) == 8
    assert poses[3].z == poses[4].z == poses[-1].z == world.settings.hover_z
    assert world.canvas_px([poses[1].x, poses[1].y]) == (102, 102)
    assert all(p.pitch == -.2 for p in poses)
    with pytest.raises(ValueError, match='finite'):
        drawing(world, [[[0, 0], [float('nan'), 1]]], rpy=[0, 0, 0])


def test_held_out_validation_detects_bad_alignment():
    camera = Settings().cameras[0]
    xyz = [[.16, 0, .03], [.20, .01, .05], [.18, -.01, .04]]
    uv, _ = project(xyz, camera)
    assert validate_camera(camera, xyz, uv)['passed']
    assert not validate_camera(camera, xyz, uv + 10)['passed']


def test_review_uses_acquisition_time_and_excludes_unexecuted_dip(tmp_path):
    trajectory = tmp_path / 'trajectory.json'
    trajectory.write_text(json.dumps({'samples': [
        {'t': 0, 'phase': 'travel', 'tip': [0, 0, .1]},
        {'t': 1, 'phase': 'load', 'tip': [0, 0, 0]}]}))
    index = tmp_path / 'frames.jsonl'
    index.write_text('\n'.join(json.dumps({'camera': 'front', 'captured_at_s': t,
                                         'image': str(t), 'source': 'physical'})
                               for t in [2, 10.1, 11, 20]))
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'trajectory': str(trajectory), 'motion_frames': str(index),
                                 'session_start_s': 10, 'duration_s': 1, 'elapsed_s': .2,
                                 'commanded_samples': 0}))
    result = review_motion(report)
    assert len(result['frames']) == 1
    assert result['frames'][0]['image'] == '10.1'
    assert result['frames'][0]['planned_phase'] == 'travel'


def test_motion_archive_is_readable_and_preserves_timestamp(tmp_path):
    from PIL import Image

    from so_paint.telemetry import ExecutionLog

    class Stream:
        def set_time(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

    log = ExecutionLog(Stream(), lambda *a: None, {'front': 'front'}, frame_dir=tmp_path)
    import numpy as np
    log.camera_frame('front', np.zeros((24, 32, 3), dtype=np.uint8), 7.125, source='physical')
    entry = json.loads(log.frame_index.read_text())
    assert entry['captured_at_s'] == 7.125
    assert entry['source'] == 'physical'
    assert Image.open(entry['image']).size == (32, 24)
