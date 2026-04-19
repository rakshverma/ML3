# Iteration 1: KGIS / KIADB Data Request

This is the only data I need from you right now.

## Goal

Create a clean premises registry so we can start monitoring actual factory polygons instead of broad industrial-area extents.

## Minimum required

For each pilot site, please get:

1. A polygon boundary for the factory plot or premise
- preferred formats: `GeoJSON`, `GPKG`, `Shapefile`, or `KML`
- if multiple files come from a shapefile, keep the full set together
- premise-level polygon is required
- industrial-area-only polygon is not enough

2. A metadata sheet using this template
- [kgis_premises_inventory_template.csv](../templates/kgis_premises_inventory_template.csv)

3. Evidence of source
- portal URL or screenshot showing which layer/site the boundary came from
- download date

## Strongly preferred fields from the portal

Please capture these if the portal exposes them:

- `plot_id`
- `industrial_area_name`
- `district`
- `allottee_name`
- `env_category` or pollution category
- `sector`
- `status`
- `total_area_hectares` or area field
- `survey_no` or cadastral reference
- `ec_obtained` if shown

## Pilot scope I recommend

Please start with:

- 20 to 50 premises
- 2 to 3 industrial areas
- a mix of red and orange category sites if available

Why:

- enough to validate the pipeline,
- small enough to clean manually,
- enough variation to test rule logic.

## If the portal does not allow GIS export

Any of these are acceptable fallbacks:

1. Boundary file from the department that owns the layer
2. Boundary file exported by a GIS operator from the portal
3. A manually digitized GeoJSON/KML for a smaller pilot set
4. Screenshots plus exact site identifiers, so I can help define a fallback ingestion path next

## File organization

Please place the data like this when you have it:

```text
data/
  raw/
    kgis/
      boundaries/
      metadata/
      screenshots/
```

Suggested filenames:

- `kgis_premises_inventory.csv`
- `pilot_boundaries.geojson`
- `site_001_portal_screenshot.png`

## Quick validation before you send it

Please check:

- each site has one boundary polygon
- each site has one stable `site_id`
- the metadata row and polygon refer to the same premise
- the CRS is preserved if the file includes one

## What I will do once you bring this

Next iteration I will:

- load and normalize all boundaries,
- validate geometry quality,
- join metadata to polygons,
- prepare the open-satellite monitoring pipeline for vegetation compliance.
