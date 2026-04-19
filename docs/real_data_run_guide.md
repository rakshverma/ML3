# Real Data Run Guide

The repository supports a real-data execution path for premises monitoring, reporting, and continuous runs.

Run it with:

```bash
uv run python -m ml3 run-real path/to/your_run_config.json
```

Start from [templates/real_run_config_template.json](../templates/real_run_config_template.json) and replace the placeholder paths with your actual site files first.

For a fused Sentinel-2 + Landsat run, start from [templates/multisource_real_run_config_template.json](../templates/multisource_real_run_config_template.json).

For continuous monitoring operations, run:

```bash
uv run python -m ml3 run-continuous path/to/your_run_config.json --iterations 12 --interval-seconds 3600
```

This writes a cumulative history JSON with run timestamps, key metrics, and alert counts.

Important:

- relative paths in the config are resolved from the config file location

## What the real runner expects

### 1. Premises boundary

Supported now:

- zipped shapefile: `.zip`
- shapefile: `.shp`
- GeoJSON: `.geojson` or `.json`

Important:

- the boundary must be a `POLYGON` layer
- the current Bagalkote village file is `POLYLINE`, so it cannot be used directly for compliance monitoring
- GeoJSON boundaries discovered from OpenStreetMap are typically `EPSG:4326`
- the current runner expects the GeoJSON boundary CRS to match the raster scene CRS

Selection options:

- `feature_index`
- or `filter_field` + `filter_value`

### 2. Before and after multispectral scenes

Each dated scene is currently expected as a local `.npz` file containing:

- `blue`
- `green`
- `red`
- `nir`
- optional `swir`

All bands must:

- have the same shape
- already be aligned to each other
- already be in the same CRS and grid between the before and after scene

The config also stores:

- `acquired_on`
- `origin_x`
- `origin_y`
- `pixel_width`
- `pixel_height`
- `crs`
- `source`

## Suggested export path for satellite imagery

For a real pilot, the easiest route is:

1. Get the actual factory premises polygon from KGIS/KIADB.
2. Export two dates of clipped imagery around the same premises.
3. Convert those aligned bands into `.npz` scene packages with `blue`, `green`, `red`, `nir`, and if possible `swir`.

Recommended first baseline:

- Sentinel-2 derived scenes
- before and after dry-season dates for comparability
- same resolution and same clipping extent

## Multi-source fusion with Sentinel-2 and Landsat

Yes, this improves the KGIS monitoring setup.

Why it helps:

- KGIS gives us the authoritative premises boundary
- Sentinel-2 gives stronger 10 m vegetation sensitivity and more frequent observations
- Landsat adds extra temporal coverage and helps when Sentinel-2 is cloudy or missing for the target window

The runner now supports:

- a single scene per period using `before_scene` and `after_scene`
- or a fused period composite using `before_composite` and `after_composite`

Each composite contains:

- one `acquired_on` date for the period-level composite
- one `source` label
- a `scenes` list with one or more Sentinel-2 and Landsat scene packages
- optional `sensor_weights`

Current fusion method:

- weighted pixel-wise averaging across valid pixels
- default sensor weights favor Sentinel-2 over Landsat
- if `valid_mask` or `cloud_mask` exists in the `.npz`, cloudy pixels are excluded from the composite

Recommended naming:

- use `sensor: "sentinel-2"` for Sentinel-2 scenes
- use `sensor: "landsat-8"` or `sensor: "landsat-9"` for Landsat scenes

## What is real vs synthetic now

- `run-real` is for real local inputs only
- `run-continuous` repeats real runs at your selected interval and stores run history

## Optional ML model path

You can enable a trainable model bundle in your run config under `ml_models`.

Example:

```json
"ml_models": {
	"train_from_current_run": true,
	"bundle_path": "../outputs/site_001_real_run/trained_monitoring_models.json",
	"max_samples_per_scene": 30000
}
```

Behavior:

- if `train_from_current_run` is true, the runner trains vegetation and built-up pixel classifiers and saves them
- if `bundle_path` is set without training, the runner loads and applies the saved models
- if `ml_models` is absent, the runner uses spectral rule logic (NDVI/NDBI thresholds)

## Open-source fallback without KGIS

The repository also supports open industrial candidate discovery:

```bash
uv run python -m ml3 discover-open-industrial --query "Udyambag, Belagavi, Karnataka, India" --output-dir outputs/open_udyambag
```

This uses:

- OpenStreetMap Nominatim for region search
- Overpass API for industrial candidate polygons

Important caveat:

- this is useful for screening and bootstrapping
- it is not as authoritative as KGIS/KIADB plot or cadastral data for compliance boundaries

## Current limitations

- no direct GeoTIFF loader yet
- no direct KML boundary loader yet
- polygon shapefile and GeoJSON boundary inputs are supported
- a line-only village boundary layer can help with context, but not with premise-level monitoring
