from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, model_validator

from ml3.workflows import (
    discover_open_industrial_workflow,
    inspect_kgis_workflow,
    run_real_workflow,
    workspace_summary,
)


APP_ROOT = Path.cwd().resolve()
app = FastAPI(
    title="ML3 Environmental Monitoring",
    version="0.1.0",
    description="Factory monitoring interface for GIS, imagery change detection, and compliance reporting.",
)


class KGISInspectRequest(BaseModel):
    path: str
    json_out: str | None = None


class RealRunRequest(BaseModel):
    config_path: str


class DiscoverRequest(BaseModel):
    query: str | None = None
    bbox: list[float] | None = None
    output_dir: str = "outputs/open_industrial"
    limit: int = 50
    include_buildings: bool = False

    @model_validator(mode="after")
    def validate_input(self) -> "DiscoverRequest":
        if not self.query and not self.bbox:
            raise ValueError("Provide either query or bbox.")
        if self.bbox and len(self.bbox) != 4:
            raise ValueError("bbox must contain [south, west, north, east].")
        return self


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _render_homepage()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    return workspace_summary(APP_ROOT)


@app.post("/api/kgis/inspect")
def api_inspect_kgis(payload: KGISInspectRequest) -> dict[str, Any]:
    try:
        return _decorate_response_paths(
            inspect_kgis_workflow(
                path=payload.path,
                json_out=payload.json_out,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/real/run")
def api_run_real(payload: RealRunRequest) -> dict[str, Any]:
    try:
        result = run_real_workflow(payload.config_path)
        return _decorate_response_paths(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/open/discover")
def api_discover_open(payload: DiscoverRequest) -> dict[str, Any]:
    try:
        bbox = tuple(payload.bbox) if payload.bbox else None
        result = discover_open_industrial_workflow(
            query=payload.query,
            bbox=bbox,
            output_dir=payload.output_dir,
            limit=payload.limit,
            include_buildings=payload.include_buildings,
        )
        return _decorate_response_paths(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/files/{file_path:path}")
def serve_file(file_path: str) -> FileResponse:
    target = _safe_resolve(file_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


def _decorate_response_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        decorated: dict[str, Any] = {}
        for key, value in payload.items():
            decorated[key] = _decorate_response_paths(value)
        if "report" in decorated and isinstance(decorated["report"], dict):
            evidence_paths = decorated["report"].get("evidence_paths", {})
            if isinstance(evidence_paths, dict):
                decorated["report"]["artifact_urls"] = {
                    label: _path_to_file_url(path)
                    for label, path in evidence_paths.items()
                }
        return decorated
    if isinstance(payload, list):
        return [_decorate_response_paths(item) for item in payload]
    if isinstance(payload, str):
        try:
            path = Path(payload)
            if path.exists():
                return payload
        except OSError:
            return payload
    return payload


def _path_to_file_url(raw_path: str) -> str | None:
    try:
        resolved = Path(raw_path).expanduser().resolve()
    except OSError:
        return None
    try:
        relative = resolved.relative_to(APP_ROOT)
    except ValueError:
        return None
    return f"/files/{relative.as_posix()}"


def _safe_resolve(raw_path: str) -> Path:
    candidate = (APP_ROOT / raw_path).resolve()
    try:
        candidate.relative_to(APP_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path is outside the workspace") from exc
    return candidate


def _render_homepage() -> str:
    summary = workspace_summary(APP_ROOT)
    configs_json = json.dumps(summary["configs"])
    boundaries_json = json.dumps(summary["boundary_candidates"])
    reports_json = json.dumps(summary["report_jsons"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML3 Monitoring Interface</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    :root {{
      --bg: #eef2e6;
      --panel: #fffef8;
      --ink: #13211c;
      --muted: #5d6d65;
      --line: #d7dfd0;
      --accent: #0f766e;
      --accent-2: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,.12), transparent 32%),
        radial-gradient(circle at top right, rgba(217,119,6,.10), transparent 28%),
        linear-gradient(180deg, #f4f7ef 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,254,248,.95), rgba(245,248,240,.92));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 24px 60px rgba(19,33,28,.08);
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      max-width: 860px;
      color: var(--muted);
      font-size: 17px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 20px 48px rgba(19,33,28,.06);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 21px;
    }}
    .card p {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 15px;
    }}
    label {{
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
    }}
    input {{
      width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
      background: #ffffff;
    }}
    button {{
      margin-top: 14px;
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{ background: var(--accent-2); }}
    .quicklist {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f1f5ef;
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }}
    .result {{
      margin-top: 20px;
      background: #101916;
      color: #def3e7;
      border-radius: 18px;
      padding: 18px;
      min-height: 180px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
    .artifacts {{
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }}
    .artifacts a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .artifact-images {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .artifact-images img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    #osmMap {{
      width: 100%;
      min-height: 380px;
      border-radius: 14px;
      border: 1px solid var(--line);
      overflow: hidden;
    }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.2);
    }}
    @media (max-width: 760px) {{
      .shell {{ padding: 16px; }}
      .hero h1 {{ font-size: 28px; }}
      #osmMap {{ min-height: 320px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Factory Compliance Monitoring Interface</h1>
      <p>
        Real-data only workflow: inspect boundaries, discover industrial premises from OSM,
        run compliance monitoring on real inputs, and review generated evidence artifacts.
      </p>
      <div class="quicklist" id="bootstrapChips"></div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Inspect Boundary Layer</h2>
        <p>Checks whether a KGIS or local boundary file is polygon-based and usable for premises-level monitoring.</p>
        <label>Boundary Path</label>
        <input id="inspectPath" value="01_Belagavi.shp" />
        <label>Optional JSON Output Path</label>
        <input id="inspectJsonOut" value="outputs/inspect_belagavi.json" />
        <button class="secondary" onclick="inspectBoundary()">Inspect</button>
      </div>

      <div class="card">
        <h2>Discover Open Industrial Sites</h2>
        <p>Finds candidate industrial polygons from real OpenStreetMap data and paints them on the map below.</p>
        <label>Region Query</label>
        <input id="discoverQuery" value="Udyambag, Belagavi, Karnataka, India" />
        <label>Output Directory</label>
        <input id="discoverOutput" value="outputs/open_ui_discovery" />
        <label>Candidate Limit</label>
        <input id="discoverLimit" type="number" value="15" />
        <label style="text-transform:none; letter-spacing:0; font-size:14px; margin-top:10px; display:flex; align-items:center; gap:8px;">
          <input id="discoverIncludeBuildings" type="checkbox" style="width:auto; margin:0;" />
          Include building=industrial features
        </label>
        <button onclick="discoverOpen()">Discover</button>
      </div>

      <div class="card">
        <h2>Run Real Monitoring Config</h2>
        <p>Executes only the real-data pipeline using one of the prepared config files or your own JSON config.</p>
        <label>Config Path</label>
        <input id="realConfigPath" value="configs/open_candidate_run.json" />
        <button class="secondary" onclick="runReal()">Run Real Config</button>
        <p class="muted">Available configs load below when the page opens.</p>
      </div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <h2>OpenStreetMap Candidate Overlay</h2>
      <p>Real OSM basemap with color-coded industrial polygons discovered from Overpass output.</p>
      <div id="osmMap"></div>
      <div class="legend" id="mapLegend"></div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <h2>Live Result</h2>
      <p>API output, generated file links, and preview images appear here.</p>
      <div id="artifacts" class="artifacts"></div>
      <div id="artifactImages" class="artifact-images"></div>
      <pre id="result" class="result">Ready.</pre>
    </section>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const configs = {configs_json};
    const boundaries = {boundaries_json};
    const reports = {reports_json};
    const categoryStyles = {{
      industrial: {{ color: "#b91c1c", fillColor: "#ef4444" }},
      building_industrial: {{ color: "#b45309", fillColor: "#f59e0b" }},
      works: {{ color: "#0369a1", fillColor: "#0ea5e9" }},
      other: {{ color: "#1d4ed8", fillColor: "#3b82f6" }},
    }};

    let map;
    let mapLayer;

    function initMap() {{
      map = L.map("osmMap", {{ zoomControl: true }}).setView([15.8497, 74.4977], 12);
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }}).addTo(map);
      renderLegend();
    }}

    function renderLegend() {{
      const entries = [
        ["industrial", "landuse=industrial or industrial=*"],
        ["building_industrial", "building=industrial"],
        ["works", "man_made=works"],
        ["other", "other OSM industrial candidates"],
      ];
      const legend = document.getElementById("mapLegend");
      legend.innerHTML = entries
        .map(([key, label]) => {{
          const style = categoryStyles[key];
          return `<div class="legend-item"><span class="legend-swatch" style="background:${{style.fillColor}}"></span>${{escapeHtml(label)}}</div>`;
        }})
        .join("");
    }}

    function classifyFeature(feature) {{
      const props = feature?.properties || {{}};
      const landuse = String(props.landuse || "").toLowerCase();
      const industrial = String(props.industrial || "").toLowerCase();
      const manMade = String(props.man_made || "").toLowerCase();
      const building = String(props.building || "").toLowerCase();

      if (landuse === "industrial" || industrial) return "industrial";
      if (building === "industrial") return "building_industrial";
      if (manMade === "works") return "works";
      return "other";
    }}

    async function loadGeojsonOverlay(fileUrl) {{
      const response = await fetch(fileUrl);
      if (!response.ok) throw new Error(`Failed to load GeoJSON from ${{fileUrl}}`);
      const geojson = await response.json();

      if (mapLayer) {{
        map.removeLayer(mapLayer);
      }}

      mapLayer = L.geoJSON(geojson, {{
        style: (feature) => {{
          const category = classifyFeature(feature);
          const style = categoryStyles[category] || categoryStyles.other;
          return {{
            color: style.color,
            weight: 2,
            fillColor: style.fillColor,
            fillOpacity: 0.35,
          }};
        }},
        onEachFeature: (feature, layer) => {{
          const props = feature?.properties || {{}};
          const label = props.display_name || props.name || "Industrial candidate";
          const osmType = props.osm_type || "";
          const osmId = props.osm_id || "";
          const category = classifyFeature(feature);
          layer.bindPopup(
            `<strong>${{escapeHtml(label)}}</strong><br/>` +
            `Category: ${{escapeHtml(category)}}<br/>` +
            `OSM: ${{escapeHtml(String(osmType))}}/${{escapeHtml(String(osmId))}}`
          );
        }},
      }}).addTo(map);

      const bounds = mapLayer.getBounds();
      if (bounds.isValid()) {{
        map.fitBounds(bounds.pad(0.15));
      }}
    }}

    function renderBootstrap() {{
      const chips = [];
      chips.push(...configs.map(item => `config: ${{item}}`));
      chips.push(...boundaries.slice(0, 6).map(item => `boundary: ${{item}}`));
      chips.push(...reports.slice(0, 4).map(item => `report: ${{item}}`));
      const target = document.getElementById("bootstrapChips");
      target.innerHTML = chips.map(label => `<span class="chip">${{escapeHtml(label)}}</span>`).join("");
    }}

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      const text = await response.text();
      let data;
      try {{
        data = JSON.parse(text);
      }} catch (_) {{
        data = {{ raw: text }};
      }}
      if (!response.ok) {{
        throw new Error(JSON.stringify(data, null, 2));
      }}
      return data;
    }}

    function setResult(data) {{
      document.getElementById("result").textContent = JSON.stringify(data, null, 2);
      renderArtifacts(data);
      renderMapFromResult(data).catch((error) => {{
        console.error(error);
      }});
    }}

    function renderArtifacts(data) {{
      const artifactBox = document.getElementById("artifacts");
      const imageBox = document.getElementById("artifactImages");
      artifactBox.innerHTML = "";
      imageBox.innerHTML = "";

      const urls = data?.report?.artifact_urls || {{}};
      Object.entries(urls).forEach(([label, url]) => {{
        const row = document.createElement("div");
        row.innerHTML = `<a href="${{url}}" target="_blank">${{escapeHtml(label)}}</a>`;
        artifactBox.appendChild(row);
        if (label.includes("image") || label.includes("panel")) {{
          const figure = document.createElement("div");
          figure.innerHTML = `<div class="muted">${{escapeHtml(label)}}</div><img src="${{url}}" alt="${{escapeHtml(label)}}" />`;
          imageBox.appendChild(figure);
        }}
      }});

      const artifacts = data?.artifacts || {{}};
      Object.entries(artifacts).forEach(([label, path]) => {{
        if (typeof path !== "string") return;
        const fileUrl = `/files/${{path.replace(/^\\.?\\//, "")}}`;
        const row = document.createElement("div");
        row.innerHTML = `<a href="${{fileUrl}}" target="_blank">${{escapeHtml(label)}}</a>`;
        artifactBox.appendChild(row);
      }});
    }}

    async function inspectBoundary() {{
      const payload = {{
        path: document.getElementById("inspectPath").value,
        json_out: document.getElementById("inspectJsonOut").value || null,
      }};
      setResult(await postJson("/api/kgis/inspect", payload));
    }}

    async function discoverOpen() {{
      const payload = {{
        query: document.getElementById("discoverQuery").value,
        output_dir: document.getElementById("discoverOutput").value,
        limit: parseInt(document.getElementById("discoverLimit").value, 10),
        include_buildings: document.getElementById("discoverIncludeBuildings").checked,
      }};
      setResult(await postJson("/api/open/discover", payload));
    }}

    async function runReal() {{
      const payload = {{
        config_path: document.getElementById("realConfigPath").value,
      }};
      setResult(await postJson("/api/real/run", payload));
    }}

    async function renderMapFromResult(data) {{
      const geojsonPath = data?.artifacts?.geojson;
      if (!geojsonPath) return;
      const fileUrl = `/files/${{geojsonPath.replace(/^\\.?\\//, "")}}`;
      await loadGeojsonOverlay(fileUrl);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    initMap();
    renderBootstrap();
  </script>
</body>
</html>"""
