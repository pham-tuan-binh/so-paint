"""A planar paint workbench and approximate raster deposition, not fluid physics."""

import cv2
import numpy as np

from .models import Settings


class World:
    # Rectangle is deliberately small enough for a vertical brush on the SO-101.
    paper_xy = np.array([[0.15, 0.035], [0.225, 0.035], [0.225, -0.035], [0.15, -0.035]])
    raster_size = 512

    def __init__(self, settings: Settings):
        self.settings = settings
        self.canvas = np.full((512, 512, 3), 250, np.uint8)
        self.expected = self.canvas.copy()
        self.paper_xy = np.array(settings.workspace.paper_corners_xy)
        polygon = self.paper_xy.astype(np.float32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 0.0001:
            raise ValueError("Paper corners must form a nondegenerate convex quadrilateral")
        self.to_canvas = cv2.getPerspectiveTransform(
            polygon, np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
        )
        self.stations = settings.workspace.stations
        for station in self.stations:
            if station.rim_z + 0.004 > settings.hover_z:
                raise ValueError("Hover height must clear all station rims by at least 4 mm")
            if cv2.pointPolygonTest(polygon, station.center[:2], True) >= -station.radius_m - 0.004:
                raise ValueError("A station overlaps the paper")
        for i, a in enumerate(self.stations):
            for b in self.stations[i + 1 :]:
                if (
                    np.linalg.norm(np.array(a.center[:2]) - b.center[:2])
                    < a.radius_m + b.radius_m + 0.006
                ):
                    raise ValueError("Station rims overlap")
        self.loaded_color = None
        self.paint_remaining_m = 0.0

    @property
    def corners(self):
        return np.c_[self.paper_xy, np.full(4, self.settings.paper_z)]

    def canvas_px(self, xy):
        p = self.to_canvas @ np.r_[xy[:2], 1]
        return tuple(np.rint(p[:2] / p[2]).astype(int))

    def classify(self, tip):
        x, y, z = tip
        s = self.settings
        # Each station is a solid rim with an open centre.
        for station in self.stations:
            d = np.linalg.norm(np.array([x, y]) - station.center[:2])
            if d < station.radius_m + 0.003 and z < station.rim_z + 0.002:
                if d > station.radius_m - 0.003:
                    raise ValueError(f"Brush intersects {station.name} rim; lift before travelling")
                if z < station.center[2] - station.immersion_m - 0.001:
                    raise ValueError(f"Brush target is below {station.name} contact surface")
                if z <= station.center[2] + 0.0015:
                    return ("wash" if station.kind == "washer" else "load"), station.name
        inside_paper = cv2.pointPolygonTest(
            self.paper_xy.astype(np.float32), (float(x), float(y)), True
        ) >= s.edge_margin_m
        if z < s.table_z - 0.0001 and not (inside_paper and s.paper_contact_depth_m > 0):
            raise ValueError("Brush target is below the table")
        if z < s.paper_z - s.paper_contact_depth_m - 0.001:
            raise ValueError("Brush target penetrates the paper/table")
        if z <= s.paper_z + 0.0015:
            margin = s.edge_margin_m
            if (
                cv2.pointPolygonTest(self.paper_xy.astype(np.float32), (float(x), float(y)), True)
                < margin
            ):
                raise ValueError("Paint contact is outside the paper or too close to its edge")
            return "paint", None
        return "travel", None

    def deposit(self, previous, current, expected=False, color=None):
        target = self.expected if expected else self.canvas
        rgb = color or self.loaded_color
        if rgb is None:
            return
        offset = np.zeros(2) if expected else np.array(self.settings.sim_paint_offset_xy)
        a = self.canvas_px(np.array(previous[:2]) + offset)
        b = self.canvas_px(np.array(current[:2]) + offset)
        width = max(
            1,
            round(
                self.settings.brush_width_m
                * 511
                / np.linalg.norm(self.paper_xy[1] - self.paper_xy[0])
            ),
        )
        cv2.line(target, a, b, tuple(int(v) for v in rgb), width, cv2.LINE_AA)

    def metadata(self):
        return {
            "frame": "robot_base",
            "units": "metres",
            "paper_corners_xyz": self.corners.tolist(),
            "stations": [s.model_dump() for s in self.stations],
            "hover_z": self.settings.hover_z,
            "geometry_source": self.settings.workspace.source,
        }
