import json
import threading
import time
from urllib.error import HTTPError

import numpy as np
import pytest

from so_paint.models import Settings
from so_paint.service import request, serve
from so_paint.workbench import Workbench


def test_persistent_service_and_setup_reload(tmp_path):
    config = tmp_path / "workspace.json"
    settings = Settings(rerun_screenshots=False)
    config.write_text(settings.model_dump_json())
    state = tmp_path / "server.json"
    wb = Workbench(settings, tmp_path / "run", record=False)
    thread = threading.Thread(target=serve, args=(wb, state, config))
    thread.start()
    try:
        for _ in range(100):
            if state.exists() and state.stat().st_size:
                break
            time.sleep(0.02)
        status = request(state, "status")
        assert status["hardware_connected"] is False
        assert status["backend"] == "simulation" and status["executing"] is None
        # Cancel is answerable even with nothing running, and never blocks on the session.
        assert request(state, "cancel")["cancelling"] is None
        # recover is routed, and is a hardware operation only.
        with pytest.raises(HTTPError):
            request(state, "recover", {"preview": True})
        obs = request(state, "look-at")
        tip = obs["robot"]["tip_xyz"]
        result = request(state, "move-to", {"poses": [{"x": tip[0], "y": tip[1], "z": tip[2]}]})
        assert result["revision"] == 1
        wb.world.canvas[20, 30] = [100, 20, 10]
        old_q = wb.q.copy()
        updated = settings.model_dump()
        updated["robot"] = {
            "model": "so101",
            "port": "/dev/cu.usbmodem-example",
            "calibration_path": "arm.json",
        }
        updated["look_at_cameras"] = ["side"]
        config.write_text(json.dumps(updated))
        assert request(state, "reload")["revision"] == 2
        assert np.array_equal(wb.q, old_q)
        assert wb.world.canvas[20, 30].tolist() == [100, 20, 10]
        assert wb.observed_revision == -1
        assert list(request(state, "look-at")["cameras"]) == ["side"]
        assert request(state, "status")["robot"]["port"] == "/dev/cu.usbmodem-example"
        config.write_text('{"look_at_cameras": ["missing"]}')
        with pytest.raises(HTTPError):
            request(state, "reload")
        assert wb.revision == 2
        assert wb.settings.look_at_cameras == ["side"]
        bad_state = tmp_path / "bad.json"
        connection = json.loads(state.read_text())
        connection["token"] = "incorrect"
        bad_state.write_text(json.dumps(connection))
        with pytest.raises(HTTPError) as error:
            request(bad_state, "status")
        assert error.value.code == 401
    finally:
        request(state, "stop")
        thread.join(timeout=10)
        wb.close()
    assert not thread.is_alive()
    assert not state.exists()


def test_camera_selection_validation():
    for selection in [[], ["missing"], ["side", "side"]]:
        with pytest.raises(ValueError):
            Settings(look_at_cameras=selection)
    assert [
        c.name for c in Settings(look_at_cameras=["side", "overhead"]).observation_cameras()
    ] == ["side", "overhead"]
