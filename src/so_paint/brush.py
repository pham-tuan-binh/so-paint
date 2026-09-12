"""Deterministic brush state. Validate on a copy before committing any motion."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Brush:
    color: str | None = None
    remaining_m: float = 0
    dwell_station: str | None = None
    dwell_s: float = 0

    def advance(self, previous, current, world):
        dt = current.t - previous.t
        if current.phase in ("load", "wash"):
            if current.station != self.dwell_station or previous.station != current.station:
                self.dwell_station, self.dwell_s = current.station, 0
            else:
                self.dwell_s += dt
            station = next(s for s in world.stations if s.name == current.station)
            if current.phase == "wash" and self.dwell_s >= world.settings.wash_dwell_s:
                self.color, self.remaining_m = None, 0
            if current.phase == "load":
                if (
                    self.color is not None
                    and self.color != station.name
                    and not station.mixing_well
                ):
                    raise ValueError("Clean the brush before entering a different color well")
                if self.dwell_s >= world.settings.load_dwell_s:
                    self.color = station.name
                    self.remaining_m = world.settings.max_load_distance_m
        else:
            self.dwell_station, self.dwell_s = None, 0
        if current.phase == "paint":
            if self.color is None:
                raise ValueError("Brush is clean/dry: load a color before painting")
            distance = float(np.linalg.norm(np.array(current.tip[:2]) - previous.tip[:2]))
            self.remaining_m -= distance
            if self.remaining_m < 0:
                raise ValueError("Stroke exceeds brush paint capacity; split it and reload")
