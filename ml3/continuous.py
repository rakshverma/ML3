from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml3.workflows import run_real_workflow


@dataclass
class ContinuousRunResult:
    history_path: str
    run_count: int
    runs: list[dict[str, Any]]


def run_continuous_monitoring(
    *,
    config_path: str,
    iterations: int,
    interval_seconds: int,
    history_path: str | None = None,
) -> ContinuousRunResult:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be >= 0")

    config_file = Path(config_path).expanduser().resolve()
    history_file = (
        Path(history_path).expanduser().resolve()
        if history_path
        else config_file.parent / "continuous_history.json"
    )
    history_file.parent.mkdir(parents=True, exist_ok=True)

    history_payload = _load_history(history_file)
    runs: list[dict[str, Any]] = []

    for index in range(iterations):
        started_at = datetime.now(timezone.utc).isoformat()
        run_data = run_real_workflow(str(config_file))
        finished_at = datetime.now(timezone.utc).isoformat()

        run_record = {
            "sequence": len(history_payload["runs"]) + 1,
            "started_at": started_at,
            "finished_at": finished_at,
            "config_path": str(config_file),
            "output_dir": run_data["output_dir"],
            "mode": run_data.get("mode", "real"),
            "report": {
                "site_id": run_data["report"].get("site_id"),
                "site_name": run_data["report"].get("site_name"),
                "before_date": run_data["report"].get("before_date"),
                "after_date": run_data["report"].get("after_date"),
                "green_cover_before_pct": run_data["report"]["before_metrics"].get("green_cover_pct"),
                "green_cover_after_pct": run_data["report"]["after_metrics"].get("green_cover_pct"),
                "green_cover_delta_pct_points": run_data["report"].get("green_cover_delta_pct_points"),
                "new_construction_area_sq_m": run_data["report"].get("new_construction_area_sq_m"),
                "alert_count": len(run_data["report"].get("alerts", [])),
            },
        }
        history_payload["runs"].append(run_record)
        runs.append(run_record)
        _write_history(history_file, history_payload)

        if index < iterations - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)

    return ContinuousRunResult(
        history_path=str(history_file),
        run_count=len(runs),
        runs=runs,
    )


def _load_history(history_file: Path) -> dict[str, Any]:
    if history_file.exists():
        return json.loads(history_file.read_text(encoding="utf-8"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }


def _write_history(history_file: Path, payload: dict[str, Any]) -> None:
    history_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
