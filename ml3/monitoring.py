from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml3.geometry import extract_connected_regions
from ml3.ml_models import MonitoringModelBundle, classify_with_model
from ml3.models import ComplianceReport, PremisesBoundary, RasterScene, VegetationMetrics, ViolationAlert


@dataclass(frozen=True)
class MonitoringRule:
    rule_name: str = "Industrial green-belt compliance baseline"
    required_green_cover_pct: float = 15.0
    ndvi_threshold: float = 0.35
    ndbi_threshold: float = 0.12
    built_red_threshold: float = 0.24
    built_swir_threshold: float = 0.38
    max_allowed_green_cover_drop_pct_points: float = 5.0
    min_new_construction_area_sq_m: float = 120.0
    min_construction_pixels: int = 20


@dataclass
class MonitoringRunResult:
    report: ComplianceReport
    before_vegetation_mask: np.ndarray
    after_vegetation_mask: np.ndarray
    green_loss_mask: np.ndarray
    new_construction_mask: np.ndarray


class MonitoringPipeline:
    def __init__(self, rule: MonitoringRule, model_bundle: MonitoringModelBundle | None = None) -> None:
        self.rule = rule
        self.model_bundle = model_bundle

    def run(
        self,
        premises: PremisesBoundary,
        before_scene: RasterScene,
        after_scene: RasterScene,
    ) -> MonitoringRunResult:
        if self.model_bundle is not None:
            before_vegetation_mask = classify_with_model(before_scene, self.model_bundle.vegetation_model) & premises.mask
            after_vegetation_mask = classify_with_model(after_scene, self.model_bundle.vegetation_model) & premises.mask
            before_built_mask = classify_with_model(before_scene, self.model_bundle.built_up_model) & premises.mask
            after_built_mask = classify_with_model(after_scene, self.model_bundle.built_up_model) & premises.mask
        else:
            before_vegetation_mask = classify_vegetation(before_scene, self.rule.ndvi_threshold) & premises.mask
            after_vegetation_mask = classify_vegetation(after_scene, self.rule.ndvi_threshold) & premises.mask
            before_built_mask = classify_built_up(scene=before_scene, rule=self.rule) & premises.mask
            after_built_mask = classify_built_up(scene=after_scene, rule=self.rule) & premises.mask

        green_loss_mask = before_vegetation_mask & ~after_vegetation_mask
        new_construction_mask = after_built_mask & ~before_built_mask

        before_metrics = vegetation_metrics(before_vegetation_mask, premises)
        after_metrics = vegetation_metrics(after_vegetation_mask, premises)

        min_pixels = max(
            self.rule.min_construction_pixels,
            int(np.ceil(self.rule.min_new_construction_area_sq_m / premises.transform.pixel_area_sq_m)),
        )
        new_construction_regions = extract_connected_regions(
            new_construction_mask,
            transform=premises.transform,
            min_pixels=min_pixels,
        )
        new_construction_area_sq_m = sum(region.area_sq_m for region in new_construction_regions)
        green_loss_area_sq_m = float(np.count_nonzero(green_loss_mask)) * premises.transform.pixel_area_sq_m
        green_delta = after_metrics.green_cover_pct - before_metrics.green_cover_pct

        alerts: list[ViolationAlert] = []
        if after_metrics.green_cover_pct < self.rule.required_green_cover_pct:
            alerts.append(
                ViolationAlert(
                    alert_type="green_cover_non_compliance",
                    severity="high",
                    message=(
                        f"Green cover is {after_metrics.green_cover_pct:.2f}% "
                        f"against a required {self.rule.required_green_cover_pct:.2f}%."
                    ),
                    geo_coordinate=self._premises_centroid(premises),
                    area_sq_m=after_metrics.green_cover_area_sq_m,
                    confidence=0.91,
                )
            )

        if abs(green_delta) > self.rule.max_allowed_green_cover_drop_pct_points and green_delta < 0:
            alerts.append(
                ViolationAlert(
                    alert_type="green_cover_reduction",
                    severity="high",
                    message=f"Green cover reduced by {abs(green_delta):.2f} percentage points.",
                    geo_coordinate=self._premises_centroid(premises),
                    area_sq_m=green_loss_area_sq_m,
                    confidence=0.88,
                )
            )

        for index, region in enumerate(new_construction_regions, start=1):
            alerts.append(
                ViolationAlert(
                    alert_type="new_construction_candidate",
                    severity="medium",
                    message=f"New built-up candidate #{index} detected inside the monitored premises.",
                    geo_coordinate=region.centroid_geo,
                    area_sq_m=region.area_sq_m,
                    confidence=min(0.97, 0.62 + (region.area_sq_m / 1800.0)),
                )
            )

        report = ComplianceReport(
            site_id=premises.site_id,
            site_name=premises.site_name,
            rule_name=self.rule.rule_name,
            required_green_cover_pct=self.rule.required_green_cover_pct,
            before_date=before_scene.acquired_on.isoformat(),
            after_date=after_scene.acquired_on.isoformat(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            green_cover_delta_pct_points=green_delta,
            green_loss_area_sq_m=green_loss_area_sq_m,
            new_construction_area_sq_m=new_construction_area_sq_m,
            new_construction_regions=new_construction_regions,
            alerts=alerts,
            metadata={
                "premises_area_sq_m": round(premises.area_sq_m, 2),
                "source_before": before_scene.source,
                "source_after": after_scene.source,
                "source_before_metadata": before_scene.metadata,
                "source_after_metadata": after_scene.metadata,
                "crs": premises.crs,
                "env_category": premises.metadata.get("env_category"),
                "classification_mode": "ml_model" if self.model_bundle is not None else "spectral_rules",
            },
        )
        return MonitoringRunResult(
            report=report,
            before_vegetation_mask=before_vegetation_mask,
            after_vegetation_mask=after_vegetation_mask,
            green_loss_mask=green_loss_mask,
            new_construction_mask=new_construction_mask,
        )

    @staticmethod
    def _premises_centroid(premises: PremisesBoundary) -> tuple[float, float]:
        rows, cols = np.where(premises.mask)
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        return premises.transform.pixel_center_to_geo(centroid_row, centroid_col)


def vegetation_metrics(mask: np.ndarray, premises: PremisesBoundary) -> VegetationMetrics:
    green_pixels = int(np.count_nonzero(mask))
    total_pixels = premises.total_pixels
    green_cover_pct = (green_pixels / total_pixels) * 100 if total_pixels else 0.0
    green_cover_area_sq_m = green_pixels * premises.transform.pixel_area_sq_m
    return VegetationMetrics(
        green_pixels=green_pixels,
        total_pixels=total_pixels,
        green_cover_pct=green_cover_pct,
        green_cover_area_sq_m=green_cover_area_sq_m,
    )


def normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return (numerator - denominator) / (numerator + denominator + 1e-6)


def classify_vegetation(scene: RasterScene, threshold: float) -> np.ndarray:
    ndvi = normalized_difference(scene.nir, scene.red)
    return ndvi >= threshold


def classify_built_up(scene: RasterScene, rule: MonitoringRule) -> np.ndarray:
    ndvi = normalized_difference(scene.nir, scene.red)
    if scene.swir is not None:
        ndbi = normalized_difference(scene.swir, scene.nir)
        return (
            (ndbi >= rule.ndbi_threshold)
            & (ndvi < rule.ndvi_threshold * 0.55)
            & (scene.red >= rule.built_red_threshold)
            & (scene.swir >= rule.built_swir_threshold)
        )

    brightness = (scene.red + scene.green + scene.blue) / 3.0
    ndbi = normalized_difference(brightness, scene.nir)
    brightness = (scene.red + scene.green + scene.blue + scene.nir) / 4.0
    return (ndbi >= rule.ndbi_threshold) & (ndvi < rule.ndvi_threshold * 0.55) & (brightness > 0.18)
