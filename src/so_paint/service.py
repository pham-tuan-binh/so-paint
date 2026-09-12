"""Loopback JSON service. One persistent workbench; the CLI is its client."""

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .models import MAX_BATCH_DURATION_S, Pose, Settings


def request(state_path, action, payload=None):
    state = json.loads(Path(state_path).read_text())
    # Never follow a config file to an external host or send its token through a proxy.
    port = int(state["port"])
    req = Request(
        f"http://127.0.0.1:{port}/{action}",
        data=json.dumps(payload or {}).encode(),
        headers={"Authorization": f"Bearer {state['token']}", "Content-Type": "application/json"},
        method="POST",
    )
    # A validated physical batch can run for the configured maximum, plus planning
    # and settling time. Do not time out the CLI while the arm is still executing.
    with build_opener(ProxyHandler({})).open(req, timeout=MAX_BATCH_DURATION_S + 60) as response:
        return json.load(response)


def serve(workbench, state_path, config_path):
    state_path = Path(state_path).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            status = 200
            try:
                if not secrets.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {token}"
                ):
                    self.respond(401, {"error": "Invalid local service token"})
                    return
                if self.headers.get("Origin"):
                    raise ValueError("Browser requests are not supported")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Request must contain at most 1 MB of JSON")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise TypeError("Expected a JSON object")
                if self.path == "/look-at":
                    result = workbench.look_at()
                elif self.path == "/move-to":
                    if set(data) - {"poses", "preview", "orientation_mode"}:
                        raise ValueError("Unknown move-to fields")
                    if not isinstance(data.get("preview", False), bool):
                        raise ValueError("preview must be a boolean")
                    result = workbench.move_to(
                        [Pose.model_validate(p) for p in data["poses"]],
                        preview=data.get("preview", False),
                        orientation_mode=data.get("orientation_mode", "brush_axis"),
                    )
                elif self.path == "/recover":
                    if set(data) - {"preview", "elbow_lift"}:
                        raise ValueError("Unknown recover fields")
                    if not isinstance(data.get("preview", False), bool):
                        raise ValueError("preview must be a boolean")
                    if not isinstance(data.get("elbow_lift", False), bool):
                        raise ValueError("elbow_lift must be a boolean")
                    result = workbench.recover(preview=data.get("preview", False),
                                               elbow_lift=data.get("elbow_lift", False))
                elif self.path == "/reload":
                    settings = Settings.model_validate_json(Path(config_path).read_text())
                    result = workbench.reconfigure(settings)
                elif self.path == "/status":
                    # Answerable while a physical batch is running: no session lock here.
                    result = {**workbench.status(), "config": str(config_path)}
                elif self.path == "/cancel":
                    result = workbench.request_cancel()
                elif self.path == "/stop":
                    # Stop any physical motion in place before the server goes away.
                    workbench.request_cancel()
                    result = {"status": "stopping"}
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    status, result = 404, {"error": "Unknown command"}
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                status, result = 400, {"error": str(exc)}
            self.respond(status, result)

        def respond(self, status, result):
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    # Exclusive creation prevents two servers from taking ownership of one state file.
    # A stale file after a crash is removed explicitly, never by guessing that a busy server died.
    owned = False
    try:
        fd = os.open(state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owned = True
        with os.fdopen(fd, "w") as f:
            json.dump({"port": server.server_port, "token": token}, f)
        print(json.dumps({"status": "ready", "state": str(state_path)}), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        if owned:
            state_path.unlink(missing_ok=True)
