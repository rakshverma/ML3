# Factory Premises Monitoring System

This repository is the starting point for a computer-vision and geospatial monitoring system that will:

- identify factory premises from KGIS/KIADB boundary data,
- monitor green-cover reduction over time,
- detect possible unauthorized construction inside premises,
- generate compliance-ready evidence with coordinates, temporal comparisons, and annotated imagery.

Current status:

- Iteration 0 completed: research, architecture, and data contract definition.
- A runnable baseline pipeline now exists for KGIS inspection and end-to-end real-data monitoring.
- Waiting for Iteration 1 inputs from the KGIS/KIADB website.

## Quick Start

Install and run with `uv`.

Inspect a KGIS boundary file:

```bash
uv run python -m ml3 inspect-kgis 02_Bagalkote.zip
```

Run on real local inputs:

```bash
uv run python -m ml3 run-real path/to/your_run_config.json
```

Run continuous monitoring cycles:

```bash
uv run python -m ml3 run-continuous path/to/your_run_config.json --iterations 12 --interval-seconds 3600
```

Launch the desktop UI (human-readable workflow, no direct JSON editing required in UI):

```bash
uv run python -m ml3 run-ui
```

Run the FastAPI interface:

```bash
uv run python -m ml3 serve --host 127.0.0.1 --port 8000
```

Use [templates/real_run_config_template.json](templates/real_run_config_template.json) as the starting point and replace the placeholder paths first.

For KGIS + Sentinel-2 + Landsat fusion, use:

```bash
uv run python -m ml3 run-real templates/multisource_real_run_config_template.json
```

Discover open industrial candidate polygons without KGIS:

```bash
uv run python -m ml3 discover-open-industrial --query "Udyambag, Belagavi, Karnataka, India" --output-dir outputs/open_udyambag
```

This generates:

- `before_annotated.png`
- `after_annotated.png`
- `comparison_panel.png`
- `compliance_report.json`
- `compliance_report.md`

What the current baseline implements:

- KGIS boundary/shapefile inspection
- config-driven real-data execution
- multi-source Sentinel-2 and Landsat compositing for real runs
- open-source industrial boundary discovery from OpenStreetMap
- NDVI-based vegetation detection
- built-up change screening using NDBI-style logic
- optional trainable pixel-level ML models for vegetation and built-up classification
- alert generation with geo-coordinates
- annotated evidence imagery and compliance reports
- continuous run history logging for scalable monitoring operations
- modern Tkinter desktop studio with guided file selection for boundary and scene inputs

Start here:

- [Research and pipeline design](docs/research_and_pipeline.md)
- [Real data run guide](docs/real_data_run_guide.md)
- [Iteration 1 KGIS data request](docs/iteration_01_kgis_data_request.md)
- [KGIS premises inventory template](templates/kgis_premises_inventory_template.csv)
- [Real run config template](templates/real_run_config_template.json)
- [Multi-source real run config template](templates/multisource_real_run_config_template.json)
- [Open candidate run config](configs/open_candidate_run.json)

Notes:

- The design assumes Karnataka GIS / KIADB industrial premise data.
- Compliance thresholds should be tied to the applicable official rule or consent condition for each site before we hard-code them into the scoring engine.
