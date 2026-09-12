"""Keep macOS camera ownership on the main thread of a disposable process."""

import json
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class CameraProcess:
    def __init__(self, device, width, height):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "so_paint.camera_process", json.dumps([device, width, height])],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            self.opened = self._read(1) == b"1"
        except Exception:
            self.release()
            raise

    def _read(self, size):
        result = bytearray()
        deadline = time.monotonic() + 10
        fd = self.process.stdout.fileno()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise RuntimeError("Camera process timed out")
            chunk = os.read(fd, min(size - len(result), 1024 * 1024))
            if not chunk:
                raise RuntimeError("Camera process closed unexpectedly")
            result.extend(chunk)
        return bytes(result)

    def isOpened(self):
        return self.opened and self.process.poll() is None

    def read(self):
        self.process.stdin.write(b"R")
        h, w, channels = struct.unpack("!III", self._read(12))
        if not h:
            return False, None
        return True, np.frombuffer(self._read(h * w * channels), np.uint8).reshape(h, w, channels)

    def release(self):
        self.opened = False
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdout.close()


def main():
    import cv2

    device, width, height = json.loads(sys.argv[1])
    camera = cv2.VideoCapture(device)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    output = sys.stdout.buffer
    archive = os.environ.get("SO_PAINT_FRAME_ARCHIVE")
    if archive:
        Path(archive).mkdir(parents=True, exist_ok=True)
    output.write(b"1" if camera.isOpened() else b"0")
    output.flush()
    try:
        while sys.stdin.buffer.read(1):
            ok, frame = camera.read()
            if ok:
                acquired_ns = time.monotonic_ns()
                if archive:
                    cv2.imwrite(str(Path(archive) / f"frame-{acquired_ns}.jpg"), frame)
                output.write(struct.pack("!III", *frame.shape))
                output.write(frame.tobytes())
            else:
                output.write(struct.pack("!III", 0, 0, 0))
            output.flush()
    finally:
        camera.release()


if __name__ == "__main__":
    main()
