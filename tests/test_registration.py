import numpy as np
import pytest

from so_paint.cameras import camera_matrices, project
from so_paint.models import CameraConfig, Settings
from so_paint.registration import calibrate, pixel_to_plane, reconstruct


def test_two_camera_reconstruction_recovers_physical_points():
    cameras = Settings().cameras
    xyz = np.array([0.18, 0.01, 0.005])
    observations = {
        "corner": {
            "z_m": xyz[2],
            "pixels": {c.name: project([xyz], c)[0][0].tolist() for c in cameras},
        }
    }
    result = reconstruct(cameras, observations)["corner"]
    np.testing.assert_allclose(result["xyz"], xyz, atol=1e-8)
    assert result["views"] == 2
    observations["corner"]["pixels"]["side"][0] += 100
    with pytest.raises(ValueError, match="disagree"):
        reconstruct(cameras, observations)


def test_fiducial_registration_and_independent_point():
    camera = Settings().cameras[1]
    xyz = np.array([[x, y, 0] for x in [0.1, 0.2, 0.3] for y in [-0.06, 0.06]])
    pixels = project(xyz, camera)[0]
    fitted, report = calibrate(camera, xyz, pixels)
    assert report["max_px"] < 0.01
    _, original = camera_matrices(camera)
    _, recovered = camera_matrices(fitted)
    np.testing.assert_allclose(recovered, original, atol=1e-4)
    heldout = [0.17, 0.012, 0.025]
    pixel = project([heldout], camera)[0][0]
    np.testing.assert_allclose(pixel_to_plane(fitted, pixel, heldout[2]), heldout, atol=1e-5)


def test_uncalibrated_real_camera_and_bad_correspondences_rejected():
    camera = CameraConfig(name="real", source="opencv")
    with pytest.raises(ValueError, match="not registered"):
        pixel_to_plane(camera, [320, 240], 0)
    with pytest.raises(ValueError, match="span"):
        calibrate(camera, [[i * 0.01, 0, 0] for i in range(6)], [[i, 0] for i in range(6)])
    xyz = np.array([[x, y, 0] for x in [0.1, 0.2, 0.3] for y in [-0.06, 0.06]])
    pixels = project(xyz, camera)[0]
    pixels[0] += [50, 20]
    with pytest.raises(ValueError, match="Reprojection"):
        calibrate(camera, xyz, pixels)
