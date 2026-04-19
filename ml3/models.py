from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """Simple north-up affine transform for raster-to-map conversion."""

    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float

    @property
    def pixel_area_sq_m(self) -> float:
        return abs(self.pixel_width * self.pixel_height)

    def pixel_center_to_geo(self, row: float, col: float) -> tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.pixel_width
        y = self.origin_y - (row + 0.5) * self.pixel_height
        return (x, y)

    def geo_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        col = (x - self.origin_x) / self.pixel_width
        row = (self.origin_y - y) / self.pixel_height
        return (row, col)


@dataclass
class RasterScene:
    acquired_on: date
    blue: np.ndarray
    green: np.ndarray
    red: np.ndarray
    nir: np.ndarray
    swir: np.ndarray | None
    transform: AffineTransform
    crs: str
    source: str
    valid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.red.shape


@dataclass
class PremisesBoundary:
    site_id: str
    site_name: str
    mask: np.ndarray
    transform: AffineTransform
    crs: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def area_sq_m(self) -> float:
        return self.total_pixels * self.transform.pixel_area_sq_m

    @property
    def bbox_pixels(self) -> tuple[int, int, int, int]:
        rows, cols = np.where(self.mask)
        return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


@dataclass
class ConnectedRegion:
    pixel_count: int
    bbox_pixels: tuple[int, int, int, int]
    centroid_pixel: tuple[float, float]
    centroid_geo: tuple[float, float]
    area_sq_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pixel_count": self.pixel_count,
            "bbox_pixels": list(self.bbox_pixels),
            "centroid_pixel": [round(self.centroid_pixel[0], 3), round(self.centroid_pixel[1], 3)],
            "centroid_geo": [round(self.centroid_geo[0], 3), round(self.centroid_geo[1], 3)],
            "area_sq_m": round(self.area_sq_m, 2),
        }


@dataclass
class VegetationMetrics:
    green_pixels: int
    total_pixels: int
    green_cover_pct: float
    green_cover_area_sq_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "green_pixels": self.green_pixels,
            "total_pixels": self.total_pixels,
            "green_cover_pct": round(self.green_cover_pct, 2),
            "green_cover_area_sq_m": round(self.green_cover_area_sq_m, 2),
        }


@dataclass
class ViolationAlert:
    alert_type: str
    severity: str
    message: str
    geo_coordinate: tuple[float, float] | None
    area_sq_m: float | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
        }
        if self.geo_coordinate is not None:
            payload["geo_coordinate"] = [
                round(self.geo_coordinate[0], 3),
                round(self.geo_coordinate[1], 3),
            ]
        if self.area_sq_m is not None:
            payload["area_sq_m"] = round(self.area_sq_m, 2)
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 2)
        return payload


@dataclass
class ComplianceReport:
    site_id: str
    site_name: str
    rule_name: str
    required_green_cover_pct: float
    before_date: str
    after_date: str
    before_metrics: VegetationMetrics
    after_metrics: VegetationMetrics
    green_cover_delta_pct_points: float
    green_loss_area_sq_m: float
    new_construction_area_sq_m: float
    new_construction_regions: list[ConnectedRegion]
    alerts: list[ViolationAlert]
    evidence_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "rule_name": self.rule_name,
            "required_green_cover_pct": round(self.required_green_cover_pct, 2),
            "before_date": self.before_date,
            "after_date": self.after_date,
            "before_metrics": self.before_metrics.as_dict(),
            "after_metrics": self.after_metrics.as_dict(),
            "green_cover_delta_pct_points": round(self.green_cover_delta_pct_points, 2),
            "green_loss_area_sq_m": round(self.green_loss_area_sq_m, 2),
            "new_construction_area_sq_m": round(self.new_construction_area_sq_m, 2),
            "new_construction_regions": [region.as_dict() for region in self.new_construction_regions],
            "alerts": [alert.as_dict() for alert in self.alerts],
            "evidence_paths": self.evidence_paths,
            "metadata": self.metadata,
        }


@dataclass
class KGISInspectionResult:
    source_path: Path
    extracted_layer_path: Path
    geometry_type: str
    record_count: int
    bbox: tuple[float, float, float, float]
    fields: list[str]
    sample_record: dict[str, Any]
    crs_wkt: str | None
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "extracted_layer_path": str(self.extracted_layer_path),
            "geometry_type": self.geometry_type,
            "record_count": self.record_count,
            "bbox": [round(value, 3) for value in self.bbox],
            "fields": self.fields,
            "sample_record": self.sample_record,
            "crs_wkt": self.crs_wkt,
            "notes": self.notes,
        }
