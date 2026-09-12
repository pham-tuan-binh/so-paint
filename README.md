# SO Paint

An agent-operated SO-101 painting workbench. A coding agent reads cameras, locates
paper and paints, plans brush strokes, and inspects the actual result. Python owns
motion planning, hardware checks and recording. No nested AI service or API key.

## Give this to your agent

> Check out this repo: https://github.com/pham-tuan-binh/so-paint. Read AGENTS.md,
> then help me paint a flower meadow. Reuse my saved calibration if it exists.

Tell the agent whether you have a physical arm or want simulation. Motor calibration
requires your hands; the agent handles visual workspace registration and painting.

## Quick start

Requires Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run so-paint --config examples/workspace.json doctor
uv run so-paint --config examples/workspace.json demo
```

For a physical arm, install with `uv sync --locked --extra hardware` and follow
[AGENTS.md](AGENTS.md) and [hardware setup](docs/SETUP.md). Keep your hardware profile
in ignored `workspace.json`. Never overwrite an existing calibration with demo settings.

## Interface

- `serve`: persistent local session; hardware connects only on first observation.
- `look-at`: immutable raw camera images, alignment overlay, measured/estimated state.
- `station`: generate a separate approach, dip or repeated washer swipe sequence.
- `draw`: generate a batch of same-color polylines, optionally in paper coordinates.
- `move-to`: preview or execute generated poses through the same guarded planner.
- `review`: locate motion camera evidence for loading, washing and painting.
- `calibrate-camera`, `reconstruct`: fit and check visual workspace geometry.
- `status`, `cancel`, `recover`, `reload`, `stop`: session controls.

See [agent tool examples](docs/AGENT_TOOLS.md), [registration](docs/REAL2SIM.md),
[telemetry](docs/TELEMETRY.md), and [architecture](docs/ARCHITECTURE.md).

The physical backend has been exercised on one SO-101 setup. It is experimental:
kinematic/rim/table checks do not model full mesh collisions, brush force or fluid
physics. Only physical camera images establish whether paint reached the paper.

## Development

```sh
uv run pytest
uv run ruff check .
```

Tests use simulation and hardware stubs; they must never power or move a real arm.
Generated runs, recordings and local hardware calibration are excluded from Git.
SO-101 asset provenance and licensing are in [assets/ATTRIBUTION.md](src/so_paint/assets/ATTRIBUTION.md).
