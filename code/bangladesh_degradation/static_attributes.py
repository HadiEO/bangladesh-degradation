from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from planetary_computer import sign
from pystac_client import Client
from rasterio.transform import array_bounds
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from bangladesh_degradation.config import get_project_settings, get_section_settings, resolve_path
from bangladesh_degradation.pipeline import (
    AnalysisGrid,
    build_cell_area_raster,
    clamp_window,
    contributing_cell_count,
    intersecting_cell_count,
    load_village_unions,
    write_float_raster,
)


STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "nasadem"


@dataclass(frozen=True)
class StaticAttributesConfig:
    shapefile_path: Path
    lc_tile_paths: Mapping[str, Path]
    years: tuple[int, ...]
    output_csv: Path
    ever_cropland_path: Path
    dem_aligned_path: Path
    weight_raster_path: Path
    stac_api_url: str = STAC_API_URL
    dem_collection: str = DEM_COLLECTION
    dem_asset_key: str | None = None
    join_column: str = "ADM4_PCODE"
    cropland_classes: frozenset[int] = frozenset({10, 11, 12, 20})
    lc_start_year: int = 2000
    persist_intermediate: bool = True


def get_lc_band_index(year: int, lc_start_year: int = 2000) -> int:
    return year - lc_start_year + 1


def build_cropland_analysis_grid(
    lc_tile_paths: Mapping[str, Path],
    shapefile_path: Path,
    join_column: str,
) -> tuple[AnalysisGrid, gpd.GeoDataFrame]:
    first_tile = next(iter(lc_tile_paths.values()))
    with rasterio.open(first_tile) as ref_src:
        unions = load_village_unions(shapefile_path, ref_src.crs, join_column)
        bounds = unions.geometry.total_bounds
        raw_window = from_bounds(*bounds, transform=ref_src.transform)
        col_off = int(np.floor(raw_window.col_off))
        row_off = int(np.floor(raw_window.row_off))
        col_end = int(np.ceil(raw_window.col_off + raw_window.width))
        row_end = int(np.ceil(raw_window.row_off + raw_window.height))
        width = col_end - col_off
        height = row_end - row_off
        window = Window(col_off, row_off, width, height)
        transform = ref_src.window_transform(window)
        grid = AnalysisGrid(
            window=window,
            transform=transform,
            crs=ref_src.crs,
            width=width,
            height=height,
            bounds=array_bounds(height, width, transform),
        )
    return grid, unions


def build_ever_cropland_raster(
    lc_tile_paths: Mapping[str, Path],
    years: Iterable[int],
    grid: AnalysisGrid,
    cropland_classes: Iterable[int],
    lc_start_year: int = 2000,
) -> np.ndarray:
    cropland_classes = set(cropland_classes)
    ever = np.zeros((grid.height, grid.width), dtype=np.uint8)

    for tile_path in lc_tile_paths.values():
        with rasterio.open(tile_path) as src:
            tile_window = from_bounds(*grid.bounds, transform=src.transform)
            tile_window = clamp_window(tile_window, src.width, src.height)
            if tile_window.width <= 0 or tile_window.height <= 0:
                continue
            tile_transform = src.window_transform(tile_window)

            for year in years:
                band_index = get_lc_band_index(year, lc_start_year=lc_start_year)
                if band_index < 1 or band_index > src.count:
                    raise ValueError(f"Year {year} is outside LC band range for {tile_path.name}.")

                band = src.read(band_index, window=tile_window, masked=True)
                binary = np.isin(band.filled(255), list(cropland_classes)).astype(np.uint8)
                binary[np.ma.getmaskarray(band)] = 255

                tile_dest = np.full((grid.height, grid.width), 255, dtype=np.uint8)
                reproject(
                    source=binary,
                    destination=tile_dest,
                    src_transform=tile_transform,
                    src_crs=src.crs,
                    src_nodata=255,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )
                ever[tile_dest == 1] = 1

    return ever.astype(np.float32)


def query_dem_items(bounds: tuple[float, float, float, float], api_url: str, collection: str):
    client = Client.open(api_url)
    search = client.search(collections=[collection], bbox=bounds)
    if hasattr(search, "items"):
        items = list(search.items())
    elif hasattr(search, "get_items"):
        items = list(search.get_items())
    else:  # pragma: no cover
        raise AttributeError("Unsupported pystac-client ItemSearch interface.")
    if not items:
        raise FileNotFoundError("No DEM items were returned for the requested bounds.")
    return items


def select_dem_asset_href(items, preferred_key: str | None) -> list[str]:
    hrefs: list[str] = []
    for item in items:
        assets = item.assets
        asset_key = preferred_key
        if asset_key is None:
            for candidate in ("data", "image", "dem", "elevation"):
                if candidate in assets:
                    asset_key = candidate
                    break
        if asset_key is None or asset_key not in assets:
            raise KeyError(f"Could not find a DEM asset for item {item.id}.")
        hrefs.append(sign(assets[asset_key].href))
    return hrefs


def read_dem_subset(
    hrefs: list[str],
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, rasterio.Affine, object]:
    datasets = [rasterio.open(href) for href in hrefs]
    try:
        merged, transform = merge(datasets, bounds=bounds, masked=True)
        crs = datasets[0].crs
    finally:
        for dataset in datasets:
            dataset.close()

    dem_band = merged[0].astype(np.float32)
    dem = dem_band.filled(np.nan)
    return dem, transform, crs


def align_dem_to_grid(
    dem_array: np.ndarray,
    dem_transform,
    dem_crs,
    grid: AnalysisGrid,
) -> np.ndarray:
    aligned = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(
        source=dem_array,
        destination=aligned,
        src_transform=dem_transform,
        src_crs=dem_crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return aligned


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


def elevation_weighted_mean(values, coverage, weights) -> float:
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


def run_exactextract_static(
    dem_raster_path: Path,
    weight_raster_path: Path,
    unions: gpd.GeoDataFrame,
    join_column: str,
    metadata_columns: list[str],
) -> pd.DataFrame:
    include_cols = [join_column, *metadata_columns]
    result = exact_extract(
        str(dem_raster_path),
        unions,
        [
            elevation_weighted_mean,
            cropland_area_weight_sum_km2,
            contributing_cell_count,
            intersecting_cell_count,
        ],
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

    result = result.rename(
        columns={
            "elevation_weighted_mean": "elevation_mean_ever_cropland_m",
            "cropland_area_weight_sum_km2": "ever_cropland_area_weight_sum_km2",
            "contributing_cell_count": "n_dem_pixels_contributing",
            "intersecting_cell_count": "n_dem_pixels_intersecting",
        }
    )
    result[join_column] = result[join_column].astype(str)
    result["has_elevation_data"] = result["n_dem_pixels_contributing"] > 0
    result["zero_ever_cropland_flag"] = ~result["has_elevation_data"]
    return result


def default_static_config(project_root: Path, config_path: Path | None = None) -> StaticAttributesConfig:
    project_settings = get_project_settings(project_root, config_path)
    elevation_settings = get_section_settings(project_root, "elevation", config_path)
    return StaticAttributesConfig(
        shapefile_path=project_settings["shapefile_path"],
        lc_tile_paths=project_settings["lc_tile_paths"],
        years=project_settings["years"],
        output_csv=resolve_path(project_root, elevation_settings["output_csv"]),
        ever_cropland_path=resolve_path(project_root, elevation_settings["ever_cropland_path"]),
        dem_aligned_path=resolve_path(project_root, elevation_settings["dem_aligned_path"]),
        weight_raster_path=resolve_path(project_root, elevation_settings["weight_raster_path"]),
        stac_api_url=str(elevation_settings.get("stac_api_url", STAC_API_URL)),
        dem_collection=str(elevation_settings.get("dem_collection", DEM_COLLECTION)),
        dem_asset_key=elevation_settings.get("dem_asset_key"),
        join_column=project_settings["join_column"],
        cropland_classes=project_settings["cropland_classes"],
        lc_start_year=project_settings["lc_start_year"],
    )


def run_static_attributes(config: StaticAttributesConfig) -> pd.DataFrame:
    missing_tiles = [str(path) for path in config.lc_tile_paths.values() if not path.exists()]
    if missing_tiles:
        raise FileNotFoundError(f"Missing LC tile files: {missing_tiles}")

    grid, unions = build_cropland_analysis_grid(
        config.lc_tile_paths,
        config.shapefile_path,
        config.join_column,
    )

    ever_cropland = build_ever_cropland_raster(
        config.lc_tile_paths,
        config.years,
        grid,
        config.cropland_classes,
        lc_start_year=config.lc_start_year,
    )
    write_float_raster(ever_cropland, grid, config.ever_cropland_path)

    bounds_wgs84 = gpd.read_file(config.shapefile_path).to_crs("EPSG:4326").total_bounds
    dem_items = query_dem_items(tuple(bounds_wgs84), config.stac_api_url, config.dem_collection)
    dem_hrefs = select_dem_asset_href(dem_items, config.dem_asset_key)
    dem_array, dem_transform, dem_crs = read_dem_subset(dem_hrefs, grid.bounds)
    dem_aligned = align_dem_to_grid(dem_array, dem_transform, dem_crs, grid)
    write_float_raster(dem_aligned, grid, config.dem_aligned_path)

    weight_raster = ever_cropland * build_cell_area_raster(grid)
    write_float_raster(weight_raster, grid, config.weight_raster_path)

    metadata_columns = [column for column in ["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"] if column in unions.columns]
    result = run_exactextract_static(
        config.dem_aligned_path,
        config.weight_raster_path,
        unions,
        join_column=config.join_column,
        metadata_columns=metadata_columns,
    )
    result["dem_source"] = "NASADEM_HGT_v001"
    result["dem_resolution_m"] = round(abs(grid.transform.a) * 111320.0, 2)
    result["cropland_year_start"] = min(config.years)
    result["cropland_year_end"] = max(config.years)

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv, index=False)

    if not config.persist_intermediate:
        for path in (config.ever_cropland_path, config.dem_aligned_path, config.weight_raster_path):
            if path.exists():
                path.unlink()

    return result
