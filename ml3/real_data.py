from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
from PIL import Image, ImageDraw

from ml3.models import AffineTransform, PremisesBoundary, RasterScene
from ml3.monitoring import MonitoringRule

DEFAULT_SENSOR_WEIGHTS = {
    "sentinel-2": 1.0,
    "sentinel2": 1.0,
    "sentinel": 1.0,
    "landsat": 0.75,
    "landsat-8": 0.75,
    "landsat-9": 0.75,
}


@dataclass
class RealRunInputs:
    premises: PremisesBoundary
    before_scene: RasterScene
    after_scene: RasterScene
    rule: MonitoringRule
    output_dir: Path
    ml_models: dict[str, Any] = field(default_factory=dict)


def load_real_run_inputs(config_path: str | Path) -> RealRunInputs:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = config_file.parent

    before_scene = _load_scene_group(base_dir, config, prefix="before")
    after_scene = _load_scene_group(base_dir, config, prefix="after")
    _validate_scene_alignment(before_scene, after_scene)

    site = config["site"]
    boundary = config["boundary"]
    premises = _load_premises_boundary(
        base_dir=base_dir,
        site_id=site["site_id"],
        site_name=site["site_name"],
        site_metadata=site.get("metadata", {}),
        boundary_spec=boundary,
        transform=after_scene.transform,
        raster_shape=after_scene.shape,
        raster_crs=after_scene.crs,
    )
    rule = _load_rule(config.get("rule", {}))
    output_dir = Path(_resolve_path(base_dir, config.get("output_dir", "outputs/real_run")))
    ml_models = dict(config.get("ml_models", {}))

    return RealRunInputs(
        premises=premises,
        before_scene=before_scene,
        after_scene=after_scene,
        rule=rule,
        output_dir=output_dir,
        ml_models=ml_models,
    )


def _load_scene_group(base_dir: Path, config: dict[str, Any], prefix: str) -> RasterScene:
    composite_key = f"{prefix}_composite"
    scene_key = f"{prefix}_scene"
    if composite_key in config:
        return _load_composite_scene(base_dir, config[composite_key], prefix=prefix)
    if scene_key in config:
        return _load_scene(base_dir, config[scene_key])
    raise KeyError(f"Expected either {composite_key!r} or {scene_key!r} in the run config.")


def _load_scene(base_dir: Path, scene_spec: dict[str, Any]) -> RasterScene:
    npz_path = Path(_resolve_path(base_dir, scene_spec["npz_path"]))
    scene_data = np.load(npz_path)

    blue = np.asarray(scene_data["blue"], dtype=float)
    green = np.asarray(scene_data["green"], dtype=float)
    red = np.asarray(scene_data["red"], dtype=float)
    nir = np.asarray(scene_data["nir"], dtype=float)
    swir = np.asarray(scene_data["swir"], dtype=float) if "swir" in scene_data.files else None

    _validate_band_shapes(npz_path, [blue, green, red, nir] + ([swir] if swir is not None else []))
    valid_mask = _derive_valid_mask(scene_data, [blue, green, red, nir], swir)
    transform = AffineTransform(
        origin_x=float(scene_spec["origin_x"]),
        origin_y=float(scene_spec["origin_y"]),
        pixel_width=float(scene_spec["pixel_width"]),
        pixel_height=float(scene_spec["pixel_height"]),
    )

    return RasterScene(
        acquired_on=date.fromisoformat(scene_spec["acquired_on"]),
        blue=blue,
        green=green,
        red=red,
        nir=nir,
        swir=swir,
        transform=transform,
        crs=str(scene_spec["crs"]),
        source=str(scene_spec["source"]),
        valid_mask=valid_mask,
        metadata={
            "sensor": str(scene_spec.get("sensor", "unknown")),
            "npz_path": str(npz_path),
            "valid_pixel_ratio": round(float(np.count_nonzero(valid_mask)) / blue.size, 4),
        },
    )


def _load_composite_scene(base_dir: Path, composite_spec: dict[str, Any], prefix: str) -> RasterScene:
    scene_specs = composite_spec.get("scenes", [])
    if not scene_specs:
        raise ValueError(f"{prefix}_composite must contain at least one scene entry.")

    scenes = [_load_scene(base_dir, scene_spec) for scene_spec in scene_specs]
    for scene in scenes[1:]:
        _validate_scene_alignment(scenes[0], scene)

    composite_date = date.fromisoformat(composite_spec["acquired_on"])
    composite_source = str(
        composite_spec.get(
            "source",
            " + ".join(sorted({str(scene.metadata.get("sensor", "unknown")) for scene in scenes})),
        )
    )
    fused_bands, composite_valid_mask = _weighted_composite_bands(
        scenes,
        sensor_weights=composite_spec.get("sensor_weights", {}),
    )

    return RasterScene(
        acquired_on=composite_date,
        blue=fused_bands["blue"],
        green=fused_bands["green"],
        red=fused_bands["red"],
        nir=fused_bands["nir"],
        swir=fused_bands["swir"],
        transform=scenes[0].transform,
        crs=scenes[0].crs,
        source=composite_source,
        valid_mask=composite_valid_mask,
        metadata={
            "composite": True,
            "composite_prefix": prefix,
            "component_sources": [scene.source for scene in scenes],
            "component_sensors": [scene.metadata.get("sensor", "unknown") for scene in scenes],
            "component_dates": [scene.acquired_on.isoformat() for scene in scenes],
            "component_count": len(scenes),
            "sensor_weights": _resolve_sensor_weights(scenes, composite_spec.get("sensor_weights", {})),
            "valid_pixel_ratio": round(float(np.count_nonzero(composite_valid_mask)) / composite_valid_mask.size, 4),
        },
    )


def _weighted_composite_bands(
    scenes: list[RasterScene],
    sensor_weights: dict[str, float],
) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
    band_names = ("blue", "green", "red", "nir", "swir")
    valid_masks = [_build_scene_valid_mask(scene) for scene in scenes]
    resolved_weights = _resolve_sensor_weights(scenes, sensor_weights)
    composite_valid_mask = np.zeros(scenes[0].shape, dtype=bool)

    composite: dict[str, np.ndarray | None] = {}
    for band_name in band_names:
        band_arrays = [getattr(scene, band_name) for scene in scenes]
        if all(array is None for array in band_arrays):
            composite[band_name] = None
            continue

        reference_shape = next(array.shape for array in band_arrays if array is not None)
        numerator = np.zeros(reference_shape, dtype=float)
        denominator = np.zeros(reference_shape, dtype=float)

        for scene, valid_mask, array in zip(scenes, valid_masks, band_arrays):
            if array is None:
                continue

            band_valid = valid_mask & np.isfinite(array)
            weight = resolved_weights.get(str(scene.metadata.get("sensor", "unknown")).lower(), 1.0)
            numerator += np.where(band_valid, array * weight, 0.0)
            denominator += np.where(band_valid, weight, 0.0)

        if not np.any(denominator > 0):
            raise ValueError(f"No valid pixels available while compositing band {band_name!r}.")
        composite_valid_mask |= denominator > 0
        composite[band_name] = numerator / np.maximum(denominator, 1e-6)

    return (composite, composite_valid_mask)


def _resolve_sensor_weights(
    scenes: list[RasterScene],
    explicit_weights: dict[str, float],
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    lower_explicit = {str(key).lower(): float(value) for key, value in explicit_weights.items()}
    for scene in scenes:
        sensor = str(scene.metadata.get("sensor", "unknown")).lower()
        resolved[sensor] = lower_explicit.get(sensor, DEFAULT_SENSOR_WEIGHTS.get(sensor, 1.0))
    return resolved


def _load_premises_boundary(
    base_dir: Path,
    site_id: str,
    site_name: str,
    site_metadata: dict[str, Any],
    boundary_spec: dict[str, Any],
    transform: AffineTransform,
    raster_shape: tuple[int, int],
    raster_crs: str,
) -> PremisesBoundary:
    boundary_path = Path(_resolve_path(base_dir, boundary_spec["path"]))
    if boundary_path.suffix.lower() in {".geojson", ".json"}:
        geometry, properties = _load_geojson_boundary(boundary_path, boundary_spec)
        boundary_crs = str(boundary_spec.get("crs", "EPSG:4326"))
        if boundary_crs != raster_crs:
            raise ValueError(
                f"GeoJSON boundary CRS {boundary_crs} does not match raster CRS {raster_crs}. "
                "Reproject the imagery or boundary first."
            )
        mask = rasterize_geojson_geometry(geometry, transform, raster_shape)
        if not np.any(mask):
            raise ValueError(
                "GeoJSON boundary did not overlap the supplied raster extent. "
                "Check CRS, transform values, and feature selection."
            )
        boundary_metadata = site_metadata | {
            "boundary_source_path": str(boundary_path),
            "boundary_feature_index": int(boundary_spec.get("feature_index", 0)),
            "boundary_record": properties,
        }
        return PremisesBoundary(
            site_id=site_id,
            site_name=site_name,
            mask=mask,
            transform=transform,
            crs=raster_crs,
            metadata=boundary_metadata,
        )

    with _open_boundary_reader(boundary_path) as reader:
        fields = [field[0] for field in reader.fields[1:]]
        feature_index = _resolve_feature_index(reader, fields, boundary_spec)
        shape_record = reader.shapeRecord(feature_index)
        geometry_type = reader.shapeTypeName.upper()
        if "POLYGON" not in geometry_type:
            raise ValueError(
                f"Boundary layer {boundary_path} has geometry type {reader.shapeTypeName}. "
                "Real monitoring requires polygon premises boundaries."
            )

        mask = rasterize_polygon_shape(shape_record.shape, transform, raster_shape)
        if not np.any(mask):
            raise ValueError(
                "Boundary did not overlap the supplied raster extent. "
                "Check CRS, transform values, and feature selection."
            )

        boundary_metadata = site_metadata | {
            "boundary_source_path": str(boundary_path),
            "boundary_feature_index": feature_index,
            "boundary_record": _normalize_record(shape_record.record.as_dict()),
        }
        return PremisesBoundary(
            site_id=site_id,
            site_name=site_name,
            mask=mask,
            transform=transform,
            crs=raster_crs,
            metadata=boundary_metadata,
        )


def rasterize_polygon_shape(
    shape: shapefile.Shape,
    transform: AffineTransform,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    polygon_rings: list[list[tuple[float, float]]] = []
    points = shape.points
    part_indexes = list(shape.parts) + [len(points)]

    for start, end in zip(part_indexes[:-1], part_indexes[1:]):
        ring = [(float(x), float(y)) for x, y in points[start:end]]
        if len(ring) >= 3:
            polygon_rings.append(ring)

    return _rasterize_polygon_rings(
        polygon_rings=polygon_rings,
        transform=transform,
        raster_shape=raster_shape,
    )


def rasterize_geojson_geometry(
    geometry: dict[str, Any],
    transform: AffineTransform,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        polygons = [geometry.get("coordinates", [])]
    elif geometry_type == "MultiPolygon":
        polygons = geometry.get("coordinates", [])
    else:
        raise ValueError(f"Unsupported GeoJSON geometry type: {geometry_type}")

    combined_mask = np.zeros(raster_shape, dtype=bool)
    for polygon in polygons:
        converted_rings = []
        for ring in polygon:
            converted_ring = [(float(point[0]), float(point[1])) for point in ring]
            if len(converted_ring) >= 3:
                converted_rings.append(converted_ring)
        if not converted_rings:
            continue
        combined_mask |= _rasterize_geojson_polygon_rings(
            polygon_rings=converted_rings,
            transform=transform,
            raster_shape=raster_shape,
        )

    return combined_mask


def _rasterize_polygon_rings(
    *,
    polygon_rings: list[list[tuple[float, float]]],
    transform: AffineTransform,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    height, width = raster_shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for ring in polygon_rings:
        pixel_ring = [_geo_to_image_point(transform, x, y) for x, y in ring]
        fill_value = 0 if _is_hole_ring(ring) else 1
        draw.polygon(pixel_ring, fill=fill_value)

    return np.array(canvas, dtype=bool)


def _rasterize_geojson_polygon_rings(
    *,
    polygon_rings: list[list[tuple[float, float]]],
    transform: AffineTransform,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    height, width = raster_shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for ring_index, ring in enumerate(polygon_rings):
        pixel_ring = [_geo_to_image_point(transform, x, y) for x, y in ring]
        fill_value = 1 if ring_index == 0 else 0
        draw.polygon(pixel_ring, fill=fill_value)

    return np.array(canvas, dtype=bool)


def _geo_to_image_point(transform: AffineTransform, x: float, y: float) -> tuple[float, float]:
    row, col = transform.geo_to_pixel(x, y)
    return (col, row)


def _is_hole_ring(ring: list[tuple[float, float]]) -> bool:
    return _signed_area(ring) > 0


def _signed_area(ring: list[tuple[float, float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _resolve_feature_index(
    reader: shapefile.Reader,
    fields: list[str],
    boundary_spec: dict[str, Any],
) -> int:
    if "feature_index" in boundary_spec:
        return int(boundary_spec["feature_index"])

    filter_field = boundary_spec.get("filter_field")
    filter_value = boundary_spec.get("filter_value")
    if filter_field and filter_value is not None:
        if filter_field not in fields:
            raise KeyError(f"Field {filter_field!r} not found in boundary layer.")
        field_index = fields.index(filter_field)
        for index, record in enumerate(reader.iterRecords()):
            if str(record[field_index]) == str(filter_value):
                return index
        raise ValueError(
            f"No boundary feature matched {filter_field}={filter_value!r}."
        )

    return 0


def _open_boundary_reader(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".shp":
        return _ReaderContext(shapefile.Reader(str(path)))
    if suffix == ".zip":
        return _TemporaryZipReader(path)
    raise ValueError("Boundary path must be a .shp, zipped shapefile (.zip), or GeoJSON file.")


def _load_geojson_boundary(path: Path, boundary_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
        feature_index = int(boundary_spec.get("feature_index", 0))
        filter_field = boundary_spec.get("filter_field")
        filter_value = boundary_spec.get("filter_value")
        if filter_field and filter_value is not None:
            for feature in features:
                if str(feature.get("properties", {}).get(filter_field)) == str(filter_value):
                    geometry = feature.get("geometry")
                    properties = feature.get("properties", {})
                    return (geometry, properties)
            raise ValueError(f"No GeoJSON feature matched {filter_field}={filter_value!r}.")
        if not features:
            raise ValueError(f"GeoJSON feature collection at {path} is empty.")
        feature = features[feature_index]
        return (feature.get("geometry"), feature.get("properties", {}))

    if payload.get("type") == "Feature":
        return (payload.get("geometry"), payload.get("properties", {}))

    return (payload, {})


class _ReaderContext:
    def __init__(self, reader: shapefile.Reader) -> None:
        self.reader = reader

    def __enter__(self) -> shapefile.Reader:
        return self.reader

    def __exit__(self, exc_type, exc, tb) -> None:
        self.reader.close()


class _TemporaryZipReader:
    def __init__(self, zip_path: Path) -> None:
        self.zip_path = zip_path
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.reader: shapefile.Reader | None = None

    def __enter__(self) -> shapefile.Reader:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="ml3_boundary_")
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(self.temp_dir.name)

        shp_paths = sorted(Path(self.temp_dir.name).rglob("*.shp"))
        if not shp_paths:
            raise FileNotFoundError(f"No .shp file found inside {self.zip_path}")

        self.reader = shapefile.Reader(str(shp_paths[0]))
        return self.reader

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.reader is not None:
            self.reader.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def _validate_band_shapes(npz_path: Path, bands: list[np.ndarray]) -> None:
    if not bands:
        raise ValueError(f"No raster bands found in {npz_path}")
    first_shape = bands[0].shape
    if any(band.shape != first_shape for band in bands):
        raise ValueError(f"Band shapes do not match in {npz_path}")


def _validate_scene_alignment(before_scene: RasterScene, after_scene: RasterScene) -> None:
    if before_scene.shape != after_scene.shape:
        raise ValueError("Before and after scenes must have the same raster shape.")
    if before_scene.crs != after_scene.crs:
        raise ValueError("Before and after scenes must use the same CRS.")
    if before_scene.transform != after_scene.transform:
        raise ValueError("Before and after scenes must use the same affine transform.")


def _build_scene_valid_mask(scene: RasterScene) -> np.ndarray:
    band_list = [scene.blue, scene.green, scene.red, scene.nir]
    if scene.swir is not None:
        band_list.append(scene.swir)
    valid_mask = np.ones(scene.shape, dtype=bool)
    for band in band_list:
        valid_mask &= np.isfinite(band)
    if scene.valid_mask is not None:
        valid_mask &= np.asarray(scene.valid_mask, dtype=bool)
    return valid_mask


def _derive_valid_mask(
    scene_data: np.lib.npyio.NpzFile,
    required_bands: list[np.ndarray],
    swir: np.ndarray | None,
) -> np.ndarray:
    valid_mask = np.ones(required_bands[0].shape, dtype=bool)
    for band in required_bands:
        valid_mask &= np.isfinite(band)
    if swir is not None:
        valid_mask &= np.isfinite(swir)

    if "valid_mask" in scene_data.files:
        valid_mask &= np.asarray(scene_data["valid_mask"], dtype=bool)
    if "cloud_mask" in scene_data.files:
        valid_mask &= ~np.asarray(scene_data["cloud_mask"], dtype=bool)
    return valid_mask


def _load_rule(rule_spec: dict[str, Any]) -> MonitoringRule:
    return MonitoringRule(
        rule_name=str(rule_spec.get("rule_name", MonitoringRule.rule_name)),
        required_green_cover_pct=float(
            rule_spec.get("required_green_cover_pct", MonitoringRule.required_green_cover_pct)
        ),
        ndvi_threshold=float(rule_spec.get("ndvi_threshold", MonitoringRule.ndvi_threshold)),
        ndbi_threshold=float(rule_spec.get("ndbi_threshold", MonitoringRule.ndbi_threshold)),
        built_red_threshold=float(
            rule_spec.get("built_red_threshold", MonitoringRule.built_red_threshold)
        ),
        built_swir_threshold=float(
            rule_spec.get("built_swir_threshold", MonitoringRule.built_swir_threshold)
        ),
        max_allowed_green_cover_drop_pct_points=float(
            rule_spec.get(
                "max_allowed_green_cover_drop_pct_points",
                MonitoringRule.max_allowed_green_cover_drop_pct_points,
            )
        ),
        min_new_construction_area_sq_m=float(
            rule_spec.get(
                "min_new_construction_area_sq_m",
                MonitoringRule.min_new_construction_area_sq_m,
            )
        ),
        min_construction_pixels=int(
            rule_spec.get("min_construction_pixels", MonitoringRule.min_construction_pixels)
        ),
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, date):
            normalized[key] = value.isoformat()
        else:
            normalized[key] = value
    return normalized


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())
