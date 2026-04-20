from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import shapefile

from ml3.models import KGISInspectionResult


def inspect_boundary_source(path: str | Path) -> KGISInspectionResult:
    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.lower()

    if suffix == ".zip":
        with tempfile.TemporaryDirectory(prefix="ml3_kgis_") as temp_dir:
            with zipfile.ZipFile(source_path) as archive:
                archive.extractall(temp_dir)

            shp_paths = sorted(Path(temp_dir).rglob("*.shp"))
            if not shp_paths:
                raise FileNotFoundError(f"No .shp file found inside {source_path}")
            return _inspect_shapefile(source_path, shp_paths[0])

    if suffix == ".shp":
        return _inspect_shapefile(source_path, source_path)

    if suffix in {".geojson", ".json"}:
        return _inspect_geojson(source_path)

    raise ValueError("Supported boundary formats: .zip shapefile, .shp, .geojson, and .json (GeoJSON).")


def _inspect_geojson(source_path: Path) -> KGISInspectionResult:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload_type = str(payload.get("type", "")).lower()

    if payload_type == "featurecollection":
        features = payload.get("features") or []
    elif payload_type == "feature":
        features = [payload]
    elif payload_type:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]
    else:
        raise ValueError("Invalid GeoJSON: missing geometry type.")

    if not isinstance(features, list) or not features:
        raise ValueError("GeoJSON has no features to inspect.")

    geometry_types: set[str] = set()
    fields: set[str] = set()
    sample_record: dict[str, Any] = {}
    all_coords: list[tuple[float, float]] = []

    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        geometry_type = str(geometry.get("type", "Unknown"))
        geometry_types.add(geometry_type)

        coordinates = geometry.get("coordinates")
        all_coords.extend(_extract_xy_pairs(coordinates))

        props = feature.get("properties") or {}
        if isinstance(props, dict):
            fields.update(str(key) for key in props.keys())
            if not sample_record:
                normalized = _normalize_value_dict(props)
                non_empty = {key: value for key, value in normalized.items() if value not in ("", 0, None)}
                if non_empty:
                    sample_record = non_empty

    if all_coords:
        xs = [xy[0] for xy in all_coords]
        ys = [xy[1] for xy in all_coords]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    geometry_type_summary = ", ".join(sorted(geometry_types)) if geometry_types else "Unknown"
    geometry_upper = geometry_type_summary.upper()
    notes: list[str] = []
    if "POLYGON" not in geometry_upper:
        notes.append("Area-based green-cover compliance needs polygon premises boundaries in a later iteration.")

    crs_wkt = None
    crs_payload = payload.get("crs")
    if crs_payload is not None:
        crs_wkt = json.dumps(crs_payload)

    return KGISInspectionResult(
        source_path=source_path,
        extracted_layer_path=source_path,
        geometry_type=geometry_type_summary,
        record_count=len(features),
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        fields=sorted(fields),
        sample_record=sample_record,
        crs_wkt=crs_wkt,
        notes=notes,
    )


def _extract_xy_pairs(node: Any) -> list[tuple[float, float]]:
    if not isinstance(node, list):
        return []
    if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
        return [(float(node[0]), float(node[1]))]
    coords: list[tuple[float, float]] = []
    for child in node:
        coords.extend(_extract_xy_pairs(child))
    return coords


def _inspect_shapefile(source_path: Path, shapefile_path: Path) -> KGISInspectionResult:
    reader = shapefile.Reader(str(shapefile_path))
    fields = [field[0] for field in reader.fields[1:]]
    sample_record = _sample_record(reader)
    prj_path = shapefile_path.with_suffix(".prj")
    crs_wkt = prj_path.read_text(encoding="utf-8", errors="ignore").strip() if prj_path.exists() else None
    notes: list[str] = []

    geometry_type = reader.shapeTypeName
    if "POLYLINE" in geometry_type.upper():
        notes.append("Line geometry only: useful as a reference layer, not as a premises polygon layer.")
    if "POLYGON" not in geometry_type.upper():
        notes.append("Area-based green-cover compliance needs polygon premises boundaries in a later iteration.")

    if any(field_name in fields for field_name in ("KGISVill_2", "UniqueVill", "Bhucode", "LGD_Villag")):
        notes.append("Village-level identifiers are present and can support manual factory-boundary creation.")

    return KGISInspectionResult(
        source_path=source_path,
        extracted_layer_path=shapefile_path,
        geometry_type=geometry_type,
        record_count=len(reader),
        bbox=tuple(float(value) for value in reader.bbox),
        fields=fields,
        sample_record=sample_record,
        crs_wkt=crs_wkt,
        notes=notes,
    )


def _sample_record(reader: shapefile.Reader) -> dict[str, Any]:
    for record in reader.iterRecords():
        payload = _normalize_value_dict(record.as_dict())
        non_empty = {key: value for key, value in payload.items() if value not in ("", 0, None)}
        if non_empty:
            return non_empty
    return {}


def _normalize_value_dict(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (datetime, date)):
            normalized[key] = value.isoformat()
        else:
            normalized[key] = value
    return normalized
