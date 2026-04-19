from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
USER_AGENT = "ml3-open-monitoring/0.1"


@dataclass
class RegionSearchResult:
    query: str
    display_name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float]
    raw: dict[str, Any]


@dataclass
class IndustrialDiscoveryResult:
    region: RegionSearchResult
    feature_collection: dict[str, Any]
    summary_rows: list[dict[str, Any]]
    raw_overpass: dict[str, Any]


def discover_open_industrial_sites(
    *,
    query: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int = 100,
    include_buildings: bool = False,
) -> IndustrialDiscoveryResult:
    if query:
        region = geocode_region(query)
        effective_bbox = region.bbox
    elif bbox:
        south, west, north, east = bbox
        region = RegionSearchResult(
            query="manual_bbox",
            display_name=f"Manual bbox {south},{west},{north},{east}",
            lat=(south + north) / 2.0,
            lon=(west + east) / 2.0,
            bbox=bbox,
            raw={"bbox": bbox},
        )
        effective_bbox = bbox
    else:
        raise ValueError("Provide either a region query or an explicit bbox.")

    overpass = fetch_industrial_features(
        bbox=effective_bbox,
        include_buildings=include_buildings,
    )
    feature_collection = overpass_to_geojson(
        overpass,
        limit=limit,
    )
    summary_rows = feature_collection_to_summary_rows(feature_collection)
    return IndustrialDiscoveryResult(
        region=region,
        feature_collection=feature_collection,
        summary_rows=summary_rows,
        raw_overpass=overpass,
    )


def geocode_region(query: str) -> RegionSearchResult:
    params = urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
        }
    )
    url = f"{NOMINATIM_SEARCH_URL}?{params}"
    payload = _fetch_json(url)
    if not payload:
        raise ValueError(f"No region match found for query {query!r}.")

    item = payload[0]
    bbox = (
        float(item["boundingbox"][0]),
        float(item["boundingbox"][2]),
        float(item["boundingbox"][1]),
        float(item["boundingbox"][3]),
    )
    return RegionSearchResult(
        query=query,
        display_name=str(item["display_name"]),
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        bbox=bbox,
        raw=item,
    )


def fetch_industrial_features(
    *,
    bbox: tuple[float, float, float, float],
    include_buildings: bool = False,
) -> dict[str, Any]:
    south, west, north, east = bbox
    clauses = [
        f'way["landuse"="industrial"]({south},{west},{north},{east});',
        f'way["man_made"="works"]({south},{west},{north},{east});',
        f'way["industrial"]({south},{west},{north},{east});',
        f'relation["landuse"="industrial"]({south},{west},{north},{east});',
        f'relation["man_made"="works"]({south},{west},{north},{east});',
        f'relation["industrial"]({south},{west},{north},{east});',
    ]
    if include_buildings:
        clauses.append(f'way["building"="industrial"]({south},{west},{north},{east});')
        clauses.append(f'relation["building"="industrial"]({south},{west},{north},{east});')

    query = "\n".join(
        [
            "[out:json][timeout:90];",
            "(",
            *clauses,
            ");",
            "out tags geom;",
        ]
    )
    payload = urlencode({"data": query}).encode("utf-8")
    last_error: Exception | None = None
    for endpoint in OVERPASS_API_URLS:
        request = Request(
            endpoint,
            data=payload,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No Overpass API endpoint was available.")


def overpass_to_geojson(overpass: dict[str, Any], *, limit: int = 100) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for element in overpass.get("elements", []):
        if element.get("type") not in {"way", "relation"}:
            continue
        geometry = element.get("geometry")
        if not geometry:
            continue

        polygon = _element_geometry_to_polygon(geometry)
        if polygon is None:
            continue

        tags = element.get("tags", {})
        centroid = _polygon_centroid(polygon[0])
        area_hint = _polygon_area_hint(polygon[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon,
                },
                "properties": {
                    "osm_type": element.get("type", ""),
                    "osm_id": element.get("id"),
                    "name": tags.get("name", ""),
                    "display_name": tags.get("name")
                    or tags.get("industrial")
                    or tags.get("man_made")
                    or tags.get("landuse")
                    or "unnamed industrial candidate",
                    "landuse": tags.get("landuse", ""),
                    "industrial": tags.get("industrial", ""),
                    "man_made": tags.get("man_made", ""),
                    "building": tags.get("building", ""),
                    "centroid_lon": centroid[0],
                    "centroid_lat": centroid[1],
                    "area_hint_deg2": area_hint,
                    "tags": tags,
                },
            }
        )

    features.sort(key=lambda feature: feature["properties"]["area_hint_deg2"], reverse=True)
    if limit > 0:
        features = features[:limit]

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenStreetMap Overpass API",
            "feature_count": len(features),
            "crs": "EPSG:4326",
        },
    }


def feature_collection_to_summary_rows(feature_collection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_collection.get("features", []), start=1):
        props = feature["properties"]
        rows.append(
            {
                "candidate_index": index - 1,
                "osm_type": props.get("osm_type", ""),
                "osm_id": props.get("osm_id", ""),
                "display_name": props.get("display_name", ""),
                "name": props.get("name", ""),
                "landuse": props.get("landuse", ""),
                "industrial": props.get("industrial", ""),
                "man_made": props.get("man_made", ""),
                "building": props.get("building", ""),
                "centroid_lon": props.get("centroid_lon", ""),
                "centroid_lat": props.get("centroid_lat", ""),
                "area_hint_deg2": props.get("area_hint_deg2", ""),
            }
        )
    return rows


def write_discovery_outputs(result: IndustrialDiscoveryResult, output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    region_path = target / "region_search.json"
    overpass_path = target / "overpass_raw.json"
    geojson_path = target / "industrial_candidates.geojson"
    summary_path = target / "industrial_candidates_summary.csv"

    region_path.write_text(
        json.dumps(
            {
                "query": result.region.query,
                "display_name": result.region.display_name,
                "lat": result.region.lat,
                "lon": result.region.lon,
                "bbox": list(result.region.bbox),
                "raw": result.region.raw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    overpass_path.write_text(json.dumps(result.raw_overpass, indent=2), encoding="utf-8")
    geojson_path.write_text(json.dumps(result.feature_collection, indent=2), encoding="utf-8")

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.summary_rows[0].keys()) if result.summary_rows else [
            "candidate_index",
            "osm_type",
            "osm_id",
            "display_name",
            "name",
            "landuse",
            "industrial",
            "man_made",
            "building",
            "centroid_lon",
            "centroid_lat",
            "area_hint_deg2",
        ])
        writer.writeheader()
        for row in result.summary_rows:
            writer.writerow(row)

    return {
        "region_search": str(region_path),
        "overpass_raw": str(overpass_path),
        "geojson": str(geojson_path),
        "summary_csv": str(summary_path),
    }


def _fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _element_geometry_to_polygon(geometry: list[dict[str, float]]) -> list[list[list[float]]] | None:
    ring = [[float(point["lon"]), float(point["lat"])] for point in geometry]
    if len(ring) < 3:
        return None
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        return None
    return [ring]


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in ring[:-1]]
    ys = [point[1] for point in ring[:-1]]
    return (round(sum(xs) / len(xs), 6), round(sum(ys) / len(ys), 6))


def _polygon_area_hint(ring: list[list[float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += x1 * y2 - x2 * y1
    return round(abs(area / 2.0), 8)
