from __future__ import annotations

import argparse
import json

import uvicorn

from ml3.continuous import run_continuous_monitoring
from ml3.tk_ui import launch_desktop_ui
from ml3.workflows import (
    discover_open_industrial_workflow,
    inspect_kgis_workflow,
    run_real_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml3",
        description="Industrial compliance monitoring pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-kgis",
        help="Inspect a KGIS boundary shapefile or zipped shapefile",
    )
    inspect_parser.add_argument("path", help="Path to .zip or .shp file")
    inspect_parser.add_argument("--json-out", help="Optional path to write a JSON inspection summary")

    real_parser = subparsers.add_parser(
        "run-real",
        help="Run the monitoring pipeline on real local inputs defined in a JSON config",
    )
    real_parser.add_argument("config", help="Path to a real-run JSON config")

    continuous_parser = subparsers.add_parser(
        "run-continuous",
        help="Run repeated real-data monitoring cycles and write run history",
    )
    continuous_parser.add_argument("config", help="Path to a real-run JSON config")
    continuous_parser.add_argument("--iterations", type=int, default=3)
    continuous_parser.add_argument("--interval-seconds", type=int, default=0)
    continuous_parser.add_argument("--history-path", help="Optional path for continuous history JSON")

    discover_parser = subparsers.add_parser(
        "discover-open-industrial",
        help="Find candidate industrial polygons from open OSM data for a region",
    )
    discover_parser.add_argument("--query", help="Region or place search text, e.g. 'Belagavi, Karnataka, India'")
    discover_parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Explicit bounding box in EPSG:4326",
    )
    discover_parser.add_argument("--output-dir", default="outputs/open_industrial")
    discover_parser.add_argument("--limit", type=int, default=100)
    discover_parser.add_argument("--include-buildings", action="store_true")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the FastAPI monitoring interface",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    subparsers.add_parser(
        "run-ui",
        help="Launch the desktop Tkinter monitoring studio",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-kgis":
        _run_inspection(args.path, args.json_out)
        return

    if args.command == "run-real":
        _run_real(args.config)
        return

    if args.command == "run-continuous":
        _run_continuous(
            config_path=args.config,
            iterations=args.iterations,
            interval_seconds=args.interval_seconds,
            history_path=args.history_path,
        )
        return

    if args.command == "discover-open-industrial":
        _run_open_discovery(
            query=args.query,
            bbox=tuple(args.bbox) if args.bbox else None,
            output_dir=args.output_dir,
            limit=args.limit,
            include_buildings=args.include_buildings,
        )
        return

    if args.command == "serve":
        uvicorn.run("ml3.api:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "run-ui":
        launch_desktop_ui()
        return

    parser.error(f"Unknown command: {args.command}")


def _run_inspection(path: str, json_out: str | None) -> None:
    result = inspect_kgis_workflow(path, json_out)
    print(f"Source: {result['source_path']}")
    print(f"Layer: {result['extracted_layer_path']}")
    print(f"Geometry type: {result['geometry_type']}")
    print(f"Record count: {result['record_count']}")
    print(f"BBox: {tuple(round(value, 3) for value in result['bbox'])}")
    print(f"Fields ({len(result['fields'])}): {', '.join(result['fields'])}")
    print("Sample record:")
    print(json.dumps(result["sample_record"], indent=2))
    if result["notes"]:
        print("Notes:")
        for note in result["notes"]:
            print(f"- {note}")
    if json_out and result.get("json_summary_path"):
        print(f"JSON summary written to {result['json_summary_path']}")


def _run_real(config_path: str) -> None:
    result = run_real_workflow(config_path)
    report = result["report"]
    print(f"Generated real-data monitoring bundle in {result['output_dir']}")
    print(
        "Green cover: "
        f"{report['before_metrics']['green_cover_pct']:.2f}% -> "
        f"{report['after_metrics']['green_cover_pct']:.2f}% "
        f"({report['green_cover_delta_pct_points']:.2f} pp)"
    )
    print(f"Alerts: {len(report['alerts'])}")
    for alert in report["alerts"]:
        print(f"- [{alert['severity']}] {alert['alert_type']}: {alert['message']}")


def _run_continuous(
    *,
    config_path: str,
    iterations: int,
    interval_seconds: int,
    history_path: str | None,
) -> None:
    result = run_continuous_monitoring(
        config_path=config_path,
        iterations=iterations,
        interval_seconds=interval_seconds,
        history_path=history_path,
    )
    print(f"Completed {result.run_count} run(s)")
    print(f"History: {result.history_path}")
    for run in result.runs:
        report = run["report"]
        print(
            "- "
            f"#{run['sequence']} {report['site_id']} "
            f"green={report['green_cover_before_pct']:.2f}%->{report['green_cover_after_pct']:.2f}% "
            f"delta={report['green_cover_delta_pct_points']:.2f}pp "
            f"alerts={report['alert_count']}"
        )


def _run_open_discovery(
    *,
    query: str | None,
    bbox: tuple[float, float, float, float] | None,
    output_dir: str,
    limit: int,
    include_buildings: bool,
) -> None:
    result = discover_open_industrial_workflow(
        query=query,
        bbox=bbox,
        output_dir=output_dir,
        limit=limit,
        include_buildings=include_buildings,
    )
    region = result["region"]
    print(f"Region: {region['display_name']}")
    print(f"Center: {region['lat']:.6f}, {region['lon']:.6f}")
    print(f"BBox: {tuple(round(value, 6) for value in region['bbox'])}")
    print(f"Candidates: {result['candidate_count']}")
    for row in result["candidates"][:10]:
        print(
            "- "
            f"#{row['candidate_index']} {row['display_name']} "
            f"(osm:{row['osm_type']}/{row['osm_id']}, "
            f"centroid={row['centroid_lat']},{row['centroid_lon']})"
        )
    print(f"GeoJSON: {result['artifacts']['geojson']}")
    print(f"Summary CSV: {result['artifacts']['summary_csv']}")
