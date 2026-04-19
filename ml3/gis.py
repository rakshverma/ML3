from __future__ import annotations

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

    raise ValueError("Only zipped shapefiles (.zip) and .shp layers are supported right now.")


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
