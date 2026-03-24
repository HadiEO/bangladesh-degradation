# Bangladesh Degradation Local Processing

This repository contains the production Python pipelines for local processing of Bangladesh `village_union` environmental covariates:

- cropland-weighted annual soil salinity (`ECe`)
- cropland-weighted elevation over ever-cropland
- cropland-weighted monthly and annual TerraClimate weather
- cropland-weighted annual groundwater table depth

The code was extracted from a larger working repository and reduced to the core local-processing workflows only. Tests, exploratory analysis, plotting, and validation scripts are intentionally excluded here.

## Repository Layout

- `code/bangladesh_degradation/`: reusable pipeline modules
- `code/run_salinity.py`: salinity CLI
- `code/run_elevation.py`: elevation CLI
- `code/run_weather.py`: TerraClimate weather CLI
- `code/run_groundwater.py`: groundwater CLI
- `code/run_custom_shapefile_processing.py`: run one or more pipelines on an alternate shapefile and isolated output namespace
- `config/paths.yml`: active machine-specific config
- `config/paths.example.yml`: template for a fresh machine
- `data/intermediate/`: generated intermediate rasters
- `output/`: final CSV outputs

## Reproducible Setup

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate bangladesh-degradation
```

2. Review and edit `config/paths.yml`.

Use `config/paths.example.yml` as the template if you need to rebuild it on a different machine.

3. Run the desired pipeline:

```bash
python code/run_salinity.py
python code/run_elevation.py
python code/run_weather.py
python code/run_groundwater.py
```

## Config

All machine-specific inputs are stored in `config/paths.yml`, including:

- the input shapefile path
- land-cover tile paths
- salinity raster directories
- groundwater source root
- output and intermediate locations
- year range and cropland class definitions

You can also point any CLI to a different config file with `--config-path`.

## Alternate Shapefile Runs

To process a different shapefile without overwriting the default outputs:

```bash
python code/run_custom_shapefile_processing.py --shapefile C:/path/to/other.shp --run-label custom_run
```

That command writes outputs under `output/custom_run/` and intermediates under `data/intermediate/custom_run/`.

## Notes

- The code writes intermediate rasters and final outputs automatically.
- `output/` and `data/intermediate/` are gitignored except for placeholder `.gitkeep` files.
- Weather and elevation depend on Microsoft Planetary Computer access.
- Groundwater uses the same annual cropland weights but a climatological groundwater raster source.
