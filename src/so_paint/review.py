"""Offline camera evidence lookup using acquisition times, never file modification times."""

import json
from pathlib import Path


def review_motion(report_path):
    report = json.loads(Path(report_path).read_text())
    if report.get("preview"):
        raise ValueError("A preview has no executed camera evidence")
    index = report.get("motion_frames")
    if not index or not Path(index).exists():
        return {"frames": [], "note": "No archived motion frames. Inspect look-at; do not infer contact."}
    start = report["session_start_s"]
    end = start + report.get("elapsed_s", report["duration_s"])
    frames = [json.loads(line) for line in Path(index).read_text().splitlines() if line.strip()]
    frames = [f for f in frames if start <= f["captured_at_s"] <= end]
    trajectory = json.loads(Path(report["trajectory"]).read_text())
    samples = trajectory["samples"]
    # Do not select unexecuted contact points from a partially stopped trajectory.
    samples = samples[:report.get("commanded_samples", len(samples) - 1) + 1]
    targets = []
    for phase in ("load", "wash", "paint", "travel"):
        candidates = [s for s in samples if s["phase"] == phase]
        if candidates:
            sample = min(candidates, key=lambda s: s["tip"][2])
            targets.append((phase, start + sample["t"]))
    selected = []
    for name in sorted({f["camera"] for f in frames}):
        camera = [f for f in frames if f["camera"] == name]
        for phase, target in targets:
            frame = min(camera, key=lambda f: abs(f["captured_at_s"] - target))
            selected.append(frame | {"planned_phase": phase, "target_session_s": target,
                                     "time_gap_s": abs(frame["captured_at_s"] - target)})
    return {"frames": selected, "frame_index": index,
            "note": "Nearest acquired frames to the lowest commanded point per phase. "
                    "Timing/phase do not prove bristle contact; inspect images and time gaps. "
                    "Use the index for other strokes or repeated dips."}
