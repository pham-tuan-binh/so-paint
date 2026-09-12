# SO-101 model

`so101.urdf` is an unmodified copy of `Simulation/SO101/so101_new_calib.urdf` from
TheRobotStudio/SO-ARM100, retrieved 2026-09-09:
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf

The 13 STL files in `assets/` are its referenced visual/collision meshes, retrieved
from commit `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`. `mesh-manifest.json` records their
paths and SHA-256 digests. Assets are bundled so visualization works offline.

Distributed under Apache-2.0; see `SO-ARM100-LICENSE` in this directory.
Rerun loads the full URDF and animates its visual meshes using joint transforms. Mesh
visualization does not imply that the planner performs mesh collision detection.
