import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rerun.urdf import UrdfTree
from scipy.spatial.transform import Rotation

from so_paint.kinematics import NAMES, Arm
from so_paint.models import Settings
from so_paint.recording import Recorder
from so_paint.workbench import Workbench
from so_paint.world import World

ASSETS = Path(__file__).parents[1] / "src/so_paint/assets"


def test_all_urdf_meshes_bundled_and_verified():
    meshes = {m.attrib["filename"] for m in ET.parse(ASSETS / "so101.urdf").getroot().iter("mesh")}
    manifest = json.loads((ASSETS / "mesh-manifest.json").read_text())
    assert {f["path"] for f in manifest["files"]} == meshes
    for f in manifest["files"]:
        assert hashlib.sha256((ASSETS / f["path"]).read_bytes()).hexdigest() == f["sha256"]


def test_rerun_joint_transforms_match_motion_solver():
    s = Settings()
    arm = Arm(s)
    tree = UrdfTree.from_file_path(ASSETS / "so101.urdf", entity_path_prefix="robot")
    assert all(p.startswith("/robot/") for p in tree.get_visual_geometry_paths("base_link"))
    for q in [[0] * 5, arm.ik([0.18, 0, 0.028]), arm.ik([0.145, -0.065, 0.012])]:
        t = np.eye(4)
        for name, value in zip(NAMES + ["gripper_frame_joint"], list(q) + [0]):
            tf = tree.get_joint_by_name(name).compute_transform(float(value), clamp=False)
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(
                tf.quaternion.as_arrow_array().to_pylist()[0]
            ).as_matrix()
            local[:3, 3] = tf.translation.as_arrow_array().to_pylist()[0]
            t = t @ local
        rendered_tip = (t @ np.r_[s.brush_tip_offset, 1])[:3]
        np.testing.assert_allclose(rendered_tip, arm.fk(q)[0], atol=1e-6)


def test_renderer_failure_retains_recording_and_no_stale_image(tmp_path, monkeypatch):
    class BrokenViewer:
        @classmethod
        def spawn(cls, **kwargs):
            raise RuntimeError("no graphics adapter")

    monkeypatch.setattr("so_paint.recording.ViewerClient", BrokenViewer)
    s = Settings()
    rec = Recorder(tmp_path / "session.rrd", World(s), s)
    try:
        (tmp_path / "scene.png").write_bytes(b"stale")
        assert rec.screenshot(tmp_path / "scene.png") is None
        assert "no graphics adapter" in rec.render_error
    finally:
        rec.close()
    assert (tmp_path / "session.rrd").stat().st_size > 1000


@pytest.mark.skipif(
    os.environ.get("SO_PAINT_TEST_RENDER") != "1", reason="requires GPU/headless Rerun"
)
def test_real_headless_urdf_screenshot(tmp_path):
    w = Workbench(output_dir=tmp_path)
    try:
        state = w.look_at()
        assert state["renderer_error"] is None
        assert state["scene_image_source"] == "Rerun headless 3D render"
        image = np.array(Image.open(state["scene_image"]).convert("RGB"))
        # Nonempty scene, including the yellow vendor robot meshes.
        assert image.shape[0] > 200 and image.shape[1] > 200
        yellow = (image[:, :, 0] > 140) & (image[:, :, 1] > 110) & (image[:, :, 2] < 120)
        assert np.count_nonzero(yellow) > 1000
    finally:
        w.close()
