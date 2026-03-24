from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import xarray as xr
from azure.core.exceptions import ClientAuthenticationError
from exactextract import exact_extract
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.windows import Window

from bangladesh_degradation.config import get_project_settings, get_section_settings, resolve_path
from bangladesh_degradation.pipeline import (
    AnalysisGrid,
    build_cell_area_raster,
    build_cropland_share_raster,
    contributing_cell_count,
    intersecting_cell_count,
    load_village_unions,
    write_float_raster,
)


TERRACLIMATE_ACCOUNT = "cpdataeuwest"
TERRACLIMATE_CONTAINER = "cpdata"
TERRACLIMATE_DATASET_PATH = "cpdata/terraclimate.zarr"


@dataclass(frozen=True)
class WeatherConfig:
    shapefile_path: Path
    lc_tile_paths: Mapping[str, Path]
    years: tuple[int, ...]
    variables: tuple[str, ...]
    monthly_output_csv: Path
    annual_output_csv: Path
    cropland_share_dir: Path
    cropland_weight_dir: Path
    monthly_raster_dir: Path
    join_column: str = "ADM4_PCODE"
    cropland_classes: frozenset[int] = frozenset({10, 11, 12, 20})
    lc_start_year: int = 2000
    terraclimate_account: str = TERRACLIMATE_ACCOUNT
    terraclimate_container: str = TERRACLIMATE_CONTAINER
    terraclimate_dataset_path: str = TERRACLIMATE_DATASET_PATH
    persist_intermediate: bool = False


def open_terraclimate_dataset(config: WeatherConfig) -> xr.Dataset:
    fs = planetary_computer.get_adlfs_filesystem(
        config.terraclimate_account,
        config.terraclimate_container,
    )
    mapper = fs.get_mapper(config.terraclimate_dataset_path)
    return xr.open_zarr(mapper, consolidated=True)


def build_terraclimate_analysis_grid(
    dataset: xr.Dataset,
    shapefile_path: Path,
    join_column: str,
    pad_cells: int = 1,
) -> tuple[AnalysisGrid, gpd.GeoDataFrame]:
    target_crs = CRS.from_epsg(4326)
    unions = load_village_unions(shapefile_path, target_crs, join_column)
    minx, miny, maxx, maxy = unions.geometry.total_bounds

    lon = np.asarray(dataset.lon.values, dtype=np.float64)
    lat = np.asarray(dataset.lat.values, dtype=np.float64)
    x_res = float(lon[1] - lon[0])
    y_res = float(abs(lat[1] - lat[0]))

    lon_mask = (lon >= (minx - pad_cells * x_res)) & (lon <= (maxx + pad_cells * x_res))
    lat_mask = (lat >= (miny - pad_cells * y_res)) & (lat <= (maxy + pad_cells * y_res))
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    if len(lon_idx) == 0 or len(lat_idx) == 0:
        raise ValueError("Village_union extent does not overlap the TerraClimate grid.")

    col_off = int(lon_idx.min())
    row_off = int(lat_idx.min())
    width = int(lon_idx.max() - lon_idx.min() + 1)
    height = int(lat_idx.max() - lat_idx.min() + 1)

    west = float(lon[col_off] - x_res / 2.0)
    north = float(lat[row_off] + y_res / 2.0)
    transform = from_origin(west, north, x_res, y_res)

    grid = AnalysisGrid(
        window=Window(col_off, row_off, width, height),
        transform=transform,
        crs=target_crs,
        width=width,
        height=height,
        bounds=array_bounds(height, width, transform),
    )
    return grid, unions


def _read_terraclimate_month(
    dataset: xr.Dataset,
    variable: str,
    year: int,
    month: int,
    grid: AnalysisGrid,
) -> np.ndarray:
    timestamp = np.datetime64(f"{year}-{month:02d}-01")
    data = (
        dataset[variable]
        .sel(time=timestamp)
        .isel(
            lat=slice(int(grid.window.row_off), int(grid.window.row_off + grid.window.height)),
            lon=slice(int(grid.window.col_off), int(grid.window.col_off + grid.window.width)),
        )
        .load()
    )
    array = np.asarray(data.values, dtype=np.float32)
    if array.shape != (grid.height, grid.width):
        raise ValueError(f"Unexpected TerraClimate subset shape for {variable} {year}-{month:02d}: {array.shape}")
    return array


def extract_terraclimate_month(
    dataset: xr.Dataset,
    config: WeatherConfig,
    variable: str,
    year: int,
    month: int,
    grid: AnalysisGrid,
) -> tuple[np.ndarray, xr.Dataset]:
    try:
        return _read_terraclimate_month(dataset, variable, year, month, grid), dataset
    except ClientAuthenticationError:
        dataset.close()
        refreshed_dataset = open_terraclimate_dataset(config)
        return _read_terraclimate_month(refreshed_dataset, variable, year, month, grid), refreshed_dataset


def weather_weighted_mean(values, coverage, weights) -> float:
    values_arr = np.ma.asarray(values, dtype=np.float64)
    weights_arr = np.ma.asarray(weights, dtype=np.float64)
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = (
        (~np.ma.getmaskarray(values_arr))
        & (~np.ma.getmaskarray(weights_arr))
        & np.isfinite(coverage_arr)
        & (coverage_arr > 0)
        & (weights_arr.data > 0)
    )
    if not np.any(valid):
        return float("nan")
    effective_weights = coverage_arr[valid] * weights_arr.data[valid]
    return float(np.average(values_arr.data[valid], weights=effective_weights))


def cropland_area_weight_sum_km2(values, coverage, weights) -> float:
    values_arr = np.ma.asarray(values)
    weights_arr = np.ma.asarray(weights, dtype=np.float64)
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = (
        (~np.ma.getmaskarray(values_arr))
        & (~np.ma.getmaskarray(weights_arr))
        & np.isfinite(coverage_arr)
        & (coverage_arr > 0)
        & (weights_arr.data > 0)
    )
    return float(np.sum(coverage_arr[valid] * weights_arr.data[valid]) / 1_000_000.0)


def run_exactextract_weather(
    raster_path: Path,
    weight_raster_path: Path,
    unions: gpd.GeoDataFrame,
    join_column: str,
    metadata_columns: list[str],
    value_column: str,
    include_metrics: bool,
) -> pd.DataFrame:
    include_cols = [join_column, *metadata_columns]
    operations = [weather_weighted_mean]
    if include_metrics:
        operations.extend(
            [
                cropland_area_weight_sum_km2,
                contributing_cell_count,
                intersecting_cell_count,
            ]
        )
    result = exact_extract(
        str(raster_path),
        unions,
        operations,
        weights=str(weight_raster_path),
        include_cols=include_cols,
        output="pandas",
        strategy="feature-sequential",
    )
    result = pd.DataFrame(result)
    if len(result) != len(unions):
        raise ValueError("exactextract result row count does not match input village_unions.")
    missing_cols = [column for column in include_cols if column not in result.columns]
    if missing_cols:
        raise ValueError(f"exactextract output is missing expected columns: {missing_cols}")
    if result[join_column].astype(str).nunique() != len(unions):
        raise ValueError(f"exactextract output has non-unique values in {join_column}.")

    rename_map = {"weather_weighted_mean": value_column}
    if include_metrics:
        rename_map.update(
            {
                "cropland_area_weight_sum_km2": "cropland_area_weight_sum_km2",
                "contributing_cell_count": "n_weather_cells_contributing",
                "intersecting_cell_count": "n_weather_cells_intersecting",
            }
        )
    result = result.rename(columns=rename_map)
    result[join_column] = result[join_column].astype(str)
    return result


def aggregate_weather_annual(monthly_df: pd.DataFrame, join_column: str, metadata_columns: list[str]) -> pd.DataFrame:
    group_cols = [join_column, *metadata_columns, "year"]

    def _count_complete(group: pd.DataFrame) -> int:
        return int(group["ppt"].notna().sum())

    annual = (
        monthly_df.groupby(group_cols, dropna=False)
        .agg(
            ppt_annual_sum=("ppt", "sum"),
            tmin_annual_mean=("tmin", "mean"),
            tmax_annual_mean=("tmax", "mean"),
            srad_annual_mean=("srad", "mean"),
            pdsi_annual_mean=("pdsi", "mean"),
            n_months_present=("ppt", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    annual["has_complete_year"] = annual["n_months_present"] == 12
    return annual


def default_weather_config(project_root: Path, config_path: Path | None = None) -> WeatherConfig:
    project_settings = get_project_settings(project_root, config_path)
    weather_settings = get_section_settings(project_root, "weather", config_path)
    return WeatherConfig(
        shapefile_path=project_settings["shapefile_path"],
        lc_tile_paths=project_settings["lc_tile_paths"],
        years=project_settings["years"],
        variables=tuple(str(variable) for variable in weather_settings.get("variables", ("ppt", "tmin", "tmax", "srad", "pdsi"))),
        monthly_output_csv=resolve_path(project_root, weather_settings["monthly_output_csv"]),
        annual_output_csv=resolve_path(project_root, weather_settings["annual_output_csv"]),
        cropland_share_dir=resolve_path(project_root, weather_settings["cropland_share_dir"]),
        cropland_weight_dir=resolve_path(project_root, weather_settings["cropland_weight_dir"]),
        monthly_raster_dir=resolve_path(project_root, weather_settings["monthly_raster_dir"]),
        join_column=project_settings["join_column"],
        cropland_classes=project_settings["cropland_classes"],
        lc_start_year=project_settings["lc_start_year"],
        terraclimate_account=str(weather_settings.get("terraclimate_account", TERRACLIMATE_ACCOUNT)),
        terraclimate_container=str(weather_settings.get("terraclimate_container", TERRACLIMATE_CONTAINER)),
        terraclimate_dataset_path=str(weather_settings.get("terraclimate_dataset_path", TERRACLIMATE_DATASET_PATH)),
    )


def run_weather_pipeline(config: WeatherConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_tiles = [str(path) for path in config.lc_tile_paths.values() if not path.exists()]
    if missing_tiles:
        raise FileNotFoundError(f"Missing LC tile files: {missing_tiles}")

    dataset = open_terraclimate_dataset(config)
    grid, unions = build_terraclimate_analysis_grid(dataset, config.shapefile_path, config.join_column)
    metadata_columns = [column for column in ["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"] if column in unions.columns]

    monthly_rows: list[dict[str, object]] = []

    for year in config.years:
        cropland_share = build_cropland_share_raster(
            config.lc_tile_paths,
            year,
            grid,
            config.cropland_classes,
            lc_start_year=config.lc_start_year,
        )
        share_path = config.cropland_share_dir / f"cropland_share_{year}.tif"
        weight_path = config.cropland_weight_dir / f"cropland_weight_{year}.tif"
        write_float_raster(cropland_share, grid, share_path)
        write_float_raster(cropland_share * build_cell_area_raster(grid), grid, weight_path)

        for month in range(1, 13):
            merged_df: pd.DataFrame | None = None
            for idx, variable in enumerate(config.variables):
                raster_path = config.monthly_raster_dir / f"{variable}_{year}_{month:02d}.tif"
                array, dataset = extract_terraclimate_month(dataset, config, variable, year, month, grid)
                write_float_raster(array, grid, raster_path)
                var_df = run_exactextract_weather(
                    raster_path,
                    weight_path,
                    unions,
                    join_column=config.join_column,
                    metadata_columns=metadata_columns,
                    value_column=variable,
                    include_metrics=(idx == 0),
                )
                if merged_df is None:
                    merged_df = var_df
                else:
                    merged_df = merged_df.merge(
                        var_df[[config.join_column, variable]],
                        on=config.join_column,
                        how="inner",
                        validate="one_to_one",
                    )
                if not config.persist_intermediate and raster_path.exists():
                    raster_path.unlink()

            assert merged_df is not None
            merged_df["year"] = year
            merged_df["month"] = month
            merged_df["has_weather_data"] = merged_df["n_weather_cells_contributing"] > 0
            merged_df["zero_cropland_flag"] = ~merged_df["has_weather_data"]
            merged_df["terraclimate_source"] = "TerraClimate"
            merged_df["terraclimate_resolution_km"] = round(abs(grid.transform.a) * 111.32, 2)
            monthly_rows.extend(merged_df.to_dict(orient="records"))

        if not config.persist_intermediate:
            for path in (share_path, weight_path):
                if path.exists():
                    path.unlink()

    monthly_df = pd.DataFrame(monthly_rows)
    annual_df = aggregate_weather_annual(monthly_df, config.join_column, metadata_columns)
    dataset.close()

    config.monthly_output_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(config.monthly_output_csv, index=False)
    annual_df.to_csv(config.annual_output_csv, index=False)
    return monthly_df, annual_df
