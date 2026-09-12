from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from so_paint.kinematics import Arm
from so_paint.models import Settings
from so_paint.recipes import at
from so_paint.telemetry import ExecutionLog
from so_paint.workbench import Workbench


class Stream:
    def __init__(self):
        self.events = []
        self.time = None

    def set_time(self, name, *, duration):
        assert name == "time"
        self.time = duration

    def log(self, path, *data):
        self.events.append((self.time, path, data))


def test_slow_camera_encoding_does_not_block_command_logging(monkeypatch):
    stream = Stream()
    recorder = ExecutionLog(stream, lambda *a, **k: None, {"front": "image"})
    arm = Arm(Settings())
    started, release = Event(), Event()

    class SlowImage:
        def __init__(self, rgb):
            pass

        def compress(self, *, jpeg_quality):
            started.set()
            assert release.wait(3)
            return "encoded camera frame"

    monkeypatch.setattr("so_paint.telemetry.rr.Image", SlowImage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        camera = pool.submit(
            recorder.camera_frame, "front", np.zeros((4, 4, 3), np.uint8),
            3.01, source="physical",
        )
        try:
            assert started.wait(1)
            # The camera is still encoding when this command must be logged.
            pool.submit(recorder.command, arm, [0] * 5, 3.02).result(timeout=1)
            assert any(path.startswith("telemetry/commanded/")
                       for _, path, _ in stream.events)
        finally:
            release.set()
        camera.result(timeout=1)
    assert next(t for t, path, _ in stream.events if path == "image") == 3.01
    assert all(t == 3.02 for t, path, _ in stream.events
               if path.startswith("telemetry/commanded/"))


def test_commands_do_not_masquerade_as_feedback_and_timestamps_are_preserved():
    stream = Stream()
    animation = []
    recorder = ExecutionLog(
        stream,
        lambda *args, **kwargs: animation.append(args),
        {"front": "world/cameras/front/image"},
    )
    arm = Arm(Settings())
    target = arm.ik([0.18, 0, 0.028])
    actual = target.copy()
    actual[0] += 0.02
    recorder.command(arm, target, 3.0)
    assert not animation
    recorder.feedback(arm, actual, 3.02, source="measured", target_joints=target)
    assert len(animation) == 1
    np.testing.assert_array_equal(animation[0][1], actual)
    # Frame arrived after feedback, but was acquired earlier. Keep capture time.
    recorder.camera_frame("front", np.zeros((12, 16, 3), np.uint8), 3.01, source="physical")
    assert next(t for t, p, _ in stream.events if p == "world/cameras/front/image") == 3.01
    assert all(t == 3.0 for t, p, _ in stream.events if p.startswith("telemetry/commanded/"))
    errors = [(t, d) for t, p, d in stream.events if p.endswith("encoder_tip_error_mm")]
    assert errors[0][0] == 3.02
    assert errors[0][1][0].scalars.as_arrow_array().to_pylist()[0] > 0
    assert not any("simulated" in path for _, path, _ in stream.events)


def test_invalid_packets_are_rejected_before_logging():
    stream = Stream()
    recorder = ExecutionLog(stream, lambda *a, **k: None, {"front": "image"})
    arm = Arm(Settings())
    with pytest.raises(ValueError, match="timestamp"):
        recorder.command(arm, [0] * 5, float("nan"))
    with pytest.raises(ValueError, match="five finite"):
        recorder.feedback(arm, [0] * 4, 0, source="measured")
    with pytest.raises(ValueError, match="Unknown camera"):
        recorder.camera_frame("absent", np.zeros((4, 4, 3), np.uint8), 0, source="physical")
    with pytest.raises(ValueError, match="uint8"):
        recorder.camera_frame("front", np.zeros((4, 4, 3)), 0, source="physical")
    assert not stream.events


def test_simulation_records_camera_frames_during_motion(tmp_path, monkeypatch):
    w = Workbench(Settings(rerun_screenshots=False), tmp_path)
    try:
        w.look_at()
        frames, feedback, commands = [], [], []
        execution = w.recorder.execution
        original_frame, original_feedback, original_command = (
            execution.camera_frame,
            execution.feedback,
            execution.command,
        )

        def camera(name, rgb, t, **kwargs):
            frames.append((name, t, w.revision))
            return original_frame(name, rgb, t, **kwargs)

        def state(arm, joints, t, **kwargs):
            feedback.append((t, kwargs["source"]))
            return original_feedback(arm, joints, t, **kwargs)

        def command(arm, joints, t):
            commands.append(t)
            return original_command(arm, joints, t)

        monkeypatch.setattr(execution, "camera_frame", camera)
        monkeypatch.setattr(execution, "feedback", state)
        monkeypatch.setattr(execution, "command", command)
        report = w.move_to([at((0.18, 0.01, 0.028))])
        assert len(commands) == report["sample_count"] - 1
        assert [t for t, _ in feedback] == commands
        assert {source for _, source in feedback} == {"simulated"}
        assert {name for name, _, _ in frames} == {"overhead", "side"}
        assert all(revision == 0 for _, _, revision in frames)  # streamed before completion
        assert any(0 < t < report["duration_s"] for _, t, _ in frames)
        assert max(t for _, t, _ in frames) == pytest.approx(report["duration_s"])
    finally:
        w.close()
