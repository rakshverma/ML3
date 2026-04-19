from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml3.gis import inspect_boundary_source
from ml3.ml_models import (
    load_monitoring_model_bundle,
    save_monitoring_model_bundle,
    train_monitoring_model_bundle,
)
from ml3.monitoring import MonitoringPipeline, MonitoringRule
from ml3.open_industrial import discover_open_industrial_sites, write_discovery_outputs
from ml3.real_data import load_real_run_inputs
from ml3.reporting import render_report_bundle, write_report_documents


def inspect_kgis_workflow(path: str, json_out: str | None = None) -> dict[str, Any]:
    result = inspect_boundary_source(path)
    payload = result.as_dict()
    if json_out:
        output_path = Path(json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["json_summary_path"] = str(output_path.resolve())
    return payload


def run_real_workflow(config_path: str) -> dict[str, Any]:
    real_inputs = load_real_run_inputs(config_path)
    ml_config = real_inputs.ml_models
    model_bundle = None
    trained_model_path: str | None = None

    bundle_path_raw = ml_config.get("bundle_path")
    bundle_path = _resolve_optional_path(config_path, bundle_path_raw)
    train_from_current_run = bool(ml_config.get("train_from_current_run", False))

    if train_from_current_run:
        model_bundle = train_monitoring_model_bundle(
            before_scene=real_inputs.before_scene,
            after_scene=real_inputs.after_scene,
            ndvi_threshold=real_inputs.rule.ndvi_threshold,
            ndbi_threshold=real_inputs.rule.ndbi_threshold,
            built_red_threshold=real_inputs.rule.built_red_threshold,
            built_swir_threshold=real_inputs.rule.built_swir_threshold,
            max_samples_per_scene=int(ml_config.get("max_samples_per_scene", 30000)),
        )
        if bundle_path is None:
            bundle_path = real_inputs.output_dir / "trained_monitoring_models.json"
        trained_model_path = str(save_monitoring_model_bundle(model_bundle, bundle_path).resolve())
    elif bundle_path is not None:
        model_bundle = load_monitoring_model_bundle(bundle_path)

    pipeline = MonitoringPipeline(real_inputs.rule, model_bundle=model_bundle)
    run_result = pipeline.run(
        premises=real_inputs.premises,
        before_scene=real_inputs.before_scene,
        after_scene=real_inputs.after_scene,
    )

    if model_bundle is not None:
        run_result.report.metadata["ml_models"] = {
            "enabled": True,
            "bundle_path": str(bundle_path.resolve()) if bundle_path is not None else None,
            "trained_in_run": train_from_current_run,
        }
    else:
        run_result.report.metadata["ml_models"] = {
            "enabled": False,
        }

    return _finalize_monitoring_run(
        output_dir=str(real_inputs.output_dir),
        premises=real_inputs.premises,
        before_scene=real_inputs.before_scene,
        after_scene=real_inputs.after_scene,
        run_result=run_result,
        mode="real",
        extra={
            "config_path": str(Path(config_path).resolve()),
            "trained_model_path": trained_model_path,
        },
    )


def discover_open_industrial_workflow(
    *,
    query: str | None,
    bbox: tuple[float, float, float, float] | None,
    output_dir: str,
    limit: int,
    include_buildings: bool,
) -> dict[str, Any]:
    result = discover_open_industrial_sites(
        query=query,
        bbox=bbox,
        limit=limit,
        include_buildings=include_buildings,
    )
    written = write_discovery_outputs(result, output_dir=output_dir)
    return {
        "region": {
            "query": result.region.query,
            "display_name": result.region.display_name,
            "lat": result.region.lat,
            "lon": result.region.lon,
            "bbox": list(result.region.bbox),
        },
        "candidate_count": len(result.summary_rows),
        "candidates": result.summary_rows,
        "artifacts": written,
    }


def workspace_summary(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_files = sorted(str(path.relative_to(root_path)) for path in (root_path / "configs").glob("*.json")) if (root_path / "configs").exists() else []
    output_jsons = sorted(
        str(path.relative_to(root_path))
        for path in root_path.rglob("compliance_report.json")
        if ".venv" not in path.parts
    )
    boundary_candidates = sorted(
        str(path.relative_to(root_path))
        for path in root_path.glob("*")
        if path.suffix.lower() in {".zip", ".shp", ".geojson", ".json", ".kmz", ".kml"}
    )
    return {
        "root": str(root_path),
        "configs": config_files,
        "report_jsons": output_jsons,
        "boundary_candidates": boundary_candidates,
    }


def _finalize_monitoring_run(
    *,
    output_dir: str,
    premises,
    before_scene,
    after_scene,
    run_result,
    mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_paths = render_report_bundle(
        output_dir=output_dir,
        premises=premises,
        before_scene=before_scene,
        after_scene=after_scene,
        run_result=run_result,
    )
    run_result.report.evidence_paths.update(evidence_paths)
    document_paths = write_report_documents(run_result.report, output_dir=output_dir)
    run_result.report.evidence_paths.update(document_paths)
    payload = {
        "mode": mode,
        "output_dir": str(Path(output_dir).resolve()),
        "report": run_result.report.as_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_optional_path(config_path: str, maybe_path: str | None) -> Path | None:
    if maybe_path is None or str(maybe_path).strip() == "":
        return None
    candidate = Path(maybe_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(config_path).expanduser().resolve().parent / candidate
