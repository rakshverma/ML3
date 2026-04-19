# Research and Pipeline Design

## 1. What the research says

### A. Premise identification from KGIS/KIADB is feasible

The public KIADB GIS portal exposes an `Industrial Plot Information System` with:

- `Click on the map to get Plot Info`
- search by `District`, `Industrial area`, `Env Category`, and `Allottee Name`
- a visible `Plot Boundary` layer

This is enough to define premise-level areas of interest if we can obtain the polygon boundaries in GIS form.

### B. Green-cover monitoring is a strong fit for open satellite data

For vegetation detection and temporal green-cover analysis, the strongest open baseline is:

- Sentinel-2 Surface Reflectance
- Sentinel-2 cloud probability masking
- NDVI/EVI-style spectral vegetation analysis
- Dynamic World class probabilities for `trees`, `grass`, `crops`, and `built`

Why this is a good baseline:

- Sentinel-2 offers 10 m resolution, 13 spectral bands, and 5-day revisit on the mission side.
- The Earth Engine harmonized Sentinel-2 SR collection is analysis-ready and consistent through time.
- Dynamic World provides near-real-time 10 m land-cover probabilities derived from Sentinel-2.

This is suitable for:

- detecting green-cover decline,
- measuring green-cover percentage inside a factory boundary,
- tracking gradual conversion from vegetated to bare/built land.

### C. Unauthorized construction detection depends on image resolution

This is the most important design constraint.

Inference from the sources:

- Sentinel-2 at 10 m is good for land-cover change and large footprint changes.
- Sentinel-1 SAR is useful as an all-weather complement, especially during cloudy periods.
- Small plot-level structures, sheds, narrow paved surfaces, and compound changes generally require higher-resolution imagery than Sentinel-2.

Practical implication:

- For pilot construction detection, use high-resolution imagery if available.
- If only open data is available at first, we should treat the construction module as a coarse screening model, not a legal-grade detector.

Recommended imagery tiers:

1. Best for construction:
- 0.3 m to 1 m imagery

2. Acceptable for larger construction footprints:
- 3 m imagery such as analysis-ready PlanetScope

3. Useful only for coarse screening:
- 10 m Sentinel-2

### D. Compliance logic must be rule-based, not only model-based

The model can measure green cover and change, but a compliance report also needs an official threshold.

As of the official PIB release posted on December 4, 2025, summarizing the MoEFCC OM dated October 29, 2025:

- inside industrial estates: common green area minimum 10%
- red-category units within estates: 15% of premises as green belt/green cover
- orange-category units within estates: 10%
- outside industrial estates: red 25%, orange 20%, with some reductions depending on air-pollution profile

We should still validate the exact applicable rule per premise because consent conditions or project-specific EC conditions may override a generic rule.

## 2. Recommended system architecture

### Module 1. Premises registry

Inputs:

- KGIS/KIADB plot or factory boundary polygons
- premise metadata such as plot ID, allottee, industrial area, pollution category

Outputs:

- normalized premises table
- one polygon AOI per monitored site

### Module 2. Imagery ingestion

Open-data baseline:

- Sentinel-2 SR
- Sentinel-2 cloud probability
- Sentinel-1 SAR
- Dynamic World

Optional higher-resolution branch:

- licensed 0.3 m to 3 m imagery for construction detection

Core preprocessing:

- CRS normalization
- cloud masking
- seasonal compositing
- clipping to premise AOIs
- date alignment for before/after comparison

### Module 3. Vegetation detection and green-belt scoring

Baseline approach:

- compute NDVI and optionally EVI
- combine vegetation indices with Dynamic World vegetation probabilities
- calculate:
  - green-cover area inside boundary
  - green-cover percentage
  - change from baseline date
  - edge/perimeter green-belt occupancy where required

Suggested outputs per site/date:

- total premise area
- vegetated area
- vegetated percentage
- delta versus baseline
- confidence score

### Module 4. Land-use and construction change detection

Phase 1 baseline:

- compare before/after composites
- detect change from vegetation or bare land to built-up-like surfaces
- use Dynamic World `built` probability change plus spectral differencing
- use Sentinel-1 backscatter change as a secondary signal in cloudy periods

Phase 2 stronger model:

- bi-temporal deep change detection on high-resolution imagery
- building or impervious-surface segmentation before and after
- polygonize changed built-up regions

Important limitation:

- do not market a Sentinel-2-only output as precise unauthorized-construction detection for small premises.
- use it as a screening layer until we have higher-resolution imagery.

### Module 5. Violation engine

Rule engine should map each site to:

- applicable compliance regime
- required green-cover threshold
- allowed pollution category context
- monitoring date window

Trigger examples:

- green-cover percentage below required threshold
- significant green-cover drop between dates
- new built-up polygon appears in previously non-built area
- change occurs inside restricted buffer or earmarked green-belt zone

### Module 6. Report generator

Each report should include:

- site identity and boundary map
- date range analyzed
- rule applied
- before/after statistics
- violation alerts
- geo-coordinates of suspicious polygons
- annotated evidence imagery
- confidence and caveat notes

### Module 7. Continuous monitoring operations

Recommended cadence:

- monthly refresh for screening
- quarterly compliance summaries
- event-triggered reruns after new imagery availability

Scalable implementation path:

- Earth Engine for imagery querying, compositing, and large-scale zonal statistics
- Python pipeline for AOI ingestion, rule logic, evidence packaging, and exports
- PostGIS or GeoParquet for premise registry and results

## 3. Iteration-wise build plan

### Iteration 0. Research and data contract

Done in this repository.

Deliverables:

- system architecture
- data request checklist
- site metadata template

### Iteration 1. KGIS/KIADB premise data ingestion

Goal:

- create the premise registry and verify that each monitored site has a valid polygon boundary

You need to provide:

- plot or premise boundary files from KGIS/KIADB
- a site inventory sheet with metadata

Success criteria:

- we can load all boundaries
- every site has a stable ID
- each boundary can be overlaid on satellite imagery

### Iteration 2. Open satellite baseline

Goal:

- launch the green-cover monitoring baseline using open data

We will build:

- Sentinel-2 compositing
- cloud masking
- NDVI and Dynamic World statistics per premise
- first green-cover compliance dashboard/report

### Iteration 3. Construction screening

Goal:

- detect coarse land-use and built-up changes

We will build:

- before/after change maps
- new built-up candidate polygons
- confidence-ranked alerts

If high-resolution imagery is available, this becomes the first strong unauthorized-construction model.

### Iteration 4. Operational reporting

Goal:

- generate repeatable board-ready compliance outputs

We will build:

- PDF/HTML reports
- CSV/GeoJSON alert exports
- annotated evidence panels
- scheduling and batch monitoring flow

## 4. Data we need from you first

For the next step, only collect the KGIS/KIADB premise data listed in:

- [Iteration 1 KGIS data request](iteration_01_kgis_data_request.md)
- [KGIS premises inventory template](../templates/kgis_premises_inventory_template.csv)

## 5. Key design decisions

### Decision 1. Start with a pilot set, not the full state

Recommended pilot:

- 20 to 50 factory premises
- 2 to 3 industrial areas
- mixed red/orange/green categories if available

This will let us validate:

- boundary quality,
- imagery availability,
- compliance scoring,
- false positives in change detection.

### Decision 2. Separate green-cover compliance from construction detection

These are related but not identical tasks.

- Green-cover compliance can start immediately with open data.
- Construction detection becomes much stronger once we have higher-resolution imagery.

### Decision 3. Keep legal confidence separate from model confidence

The system should explicitly store:

- `model_confidence`
- `data_quality`
- `rule_source`
- `human_review_status`

That prevents overstating what a model-only alert means.

## 6. Sources

Official and primary sources used:

- KIADB industrial GIS portal: https://kiadb.karnataka.gov.in/kiadbgisportal/
- Invest Karnataka page linking the KIADB GIS portal: https://investkarnataka.co.in/find-your-site/kiadb-industrial-area/
- ESA Sentinel-2 mission page: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2
- Earth Engine Sentinel-2 SR Harmonized: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Earth Engine Sentinel-2 Cloud Probability: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY
- Earth Engine Dynamic World V1: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- ESA Sentinel-1 mission page: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1
- Earth Engine Sentinel-1 GRD: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- Planet analysis-ready PlanetScope technical specification: https://docs.planet.com/data/imagery/arps/techspec/
- PIB release on revised greenbelt standards, posted December 4, 2025: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2198750

## 7. Assumptions and caveats

- I assumed `KGIS industrial boundary data` means Karnataka KGIS / KIADB industrial plot data.
- I could verify the presence of plot-boundary and plot-info functionality on the public portal, but I could not verify from the public HTML alone whether direct GIS file export is enabled for every layer.
- Because of that, the Iteration 1 checklist includes a fallback path: if direct export is unavailable, get the boundary file from the owning department or share the exact layer/screenshot workflow so we can adapt.
