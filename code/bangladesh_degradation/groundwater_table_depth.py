from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from exactextract import exact_extract
from rasterio.errors import RasterioIOError
from rasterio.crs import CRS
from rasterio.transform import from_origin

from bangladesh_degradation.config import get_project_settings, get_section_settings, resolve_path
from bangladesh_degradation.pipeline import (
    AnalysisGrid,
    build_analysis_grid,
    build_cell_area_raster,
    build_cropland_share_raster,
    contributing_cell_count,
    cropland_area_weight_sum_km2,
    intersecting_cell_count,
    load_village_unions,
    write_float_raster,
)


GROUNDWATER_VARIABLE = "WTD"
GROUNDWATER_CRS = CRS.from_epsg(4326)


@dataclass(frozen=True)
class GroundwaterSource:
    netcdf_path: Path
    raster_path: str
    variable: str
    units: str
    description: str
    resolution_x_deg: float
    resolution_y_deg: float
    nodata: float | None


@dataclass(frozen=True)
class GroundwaterRasterMetadata:
    bounds: tuple[float, float, float, float]
    resolution_x_deg: float
    resolution_y_deg: float
    nodata: float | None
    units: str
    description: str


@dataclass(frozen=True)
class GroundwaterConfig:
    shapefile_path: Path
    groundwater_root: Path
    lc_tile_paths: Mapping[str, Path]
    years: tuple[int, ...]
    output_csv: Path
    cropland_share_dir: Path
    cropland_weight_dir: Path
    join_column: str = "ADM4_PCODE"
    cropland_classes: frozenset[int] = frozenset({10, 11, 12, 20})
    groundwater_variable: str = GROUNDWATER_VARIABLE
    lc_start_year: int = 2000
    persist_intermediate: bool = False


def groundwater_subdataset_path(netcdf_path: Path, variable: str = GROUNDWATER_VARIABLE) -> str:
    return f"netcdf:{netcdf_path}:{variable}"


def discover_groundwater_annual_files(groundwater_root: Path) -> list[Path]:
    if not groundwater_root.exists():
        raise FileNotFoundError(f"Groundwater root does not exist: {groundwater_root}")
    files = sorted(
        path
        for path in groundwater_root.rglob("*.nc")
        if path.is_file() and path.name.upper().endswith("_WTD_ANNUALMEAN.NC")
    )
    if not files:
        raise FileNotFoundError(f"No annual groundwater files were found under {groundwater_root}")
    return files


def _bounds_intersect(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    left_minx, left_miny, left_maxx, left_maxy = left
    right_minx, right_miny, right_maxx, right_maxy = right
    return not (
        left_maxx <= right_minx
        or right_maxx <= left_minx
        or left_maxy <= right_miny
        or right_maxy <= left_miny
    )


def _metadata_from_xarray(netcdf_path: Path, variable: str) -> GroundwaterRasterMetadata:
    with xr.open_dataset(netcdf_path) as ds:
        data_var = ds[variable]
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        x_res = float(abs(lon[1] - lon[0]))
        y_res = float(abs(lat[1] - lat[0]))
        bounds = (
            float(lon.min() - x_res / 2.0),
            float(lat.min() - y_res / 2.0),
            float(lon.max() + x_res / 2.0),
            float(lat.max() + y_res / 2.0),
        )
        nodata = data_var.attrs.get("_FillValue")
        return GroundwaterRasterMetadata(
            bounds=bounds,
            resolution_x_deg=x_res,
            resolution_y_deg=y_res,
            nodata=None if nodata is None else float(nodata),
            units=str(data_var.attrs.get("units", "unknown")),
            description=str(data_var.attrs.get("comment", "annual mean climatology")),
        )


def read_groundwater_metadata(netcdf_path: Path, variable: str = GROUNDWATER_VARIABLE) -> GroundwaterRasterMetadata:
    raster_path = groundwater_subdataset_path(netcdf_path, variable)
    try:
        with rasterio.open(raster_path) as src:
            if src.count != 1:
                raise ValueError(f"Expected a single annual band in {netcdf_path.name}, found {src.count}.")
            value_tags = src.tags(1)
            return GroundwaterRasterMetadata(
                bounds=(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
                resolution_x_deg=float(abs(src.transform.a)),
                resolution_y_deg=float(abs(src.transform.e)),
                nodata=None if src.nodata is None else float(src.nodata),
                units=value_tags.get("units", "unknown"),
                description=value_tags.get("NETCDF_DIM_time_VALUES", "annual mean climatology"),
            )
    except RasterioIOError:
        return _metadata_from_xarray(netcdf_path, variable)


def materialize_groundwater_raster(
    source: GroundwaterSource,
    output_path: Path,
) -> Path:
    try:
        with rasterio.open(source.raster_path):
            return Path(source.raster_path)
    except RasterioIOError:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with xr.open_dataset(source.netcdf_path) as ds:
            data = ds[source.variable]
            if "time" in data.dims:
                data = data.isel(time=0)
            array = np.asarray(data.values, dtype=np.float32)
            fill_value = data.attrs.get("_FillValue")
            if fill_value is not None:
                array[array == float(fill_value)] = np.nan
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            x_res = float(abs(lon[1] - lon[0]))
            y_res = float(abs(lat[1] - lat[0]))
            transform = from_origin(
                float(lon.min() - x_res / 2.0),
                float(lat.max() + y_res / 2.0),
                x_res,
                y_res,
            )
            north_up_array = np.flipud(array)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=north_up_array.shape[0],
            width=north_up_array.shape[1],
            count=1,
            dtype="float32",
            crs=GROUNDWATER_CRS,
            transform=transform,
            nodata=np.nan,
            compress="LZW",
        ) as dst:
            dst.write(north_up_array, 1)
        return output_path


def select_groundwater_source(
    annual_files: list[Path],
    shapefile_path: Path,
    join_column: str,
    variable: str = GROUNDWATER_VARIABLE,
) -> GroundwaterSource:
    unions_wgs84 = load_village_unions(shapefile_path, GROUNDWATER_CRS, join_column)
    union_bounds = tuple(float(value) for value in unions_wgs84.geometry.total_bounds)
    matches: list[GroundwaterSource] = []

    for netcdf_path in annual_files:
        metadata = read_groundwater_metadata(netcdf_path, variable)
        if not _bounds_intersect(union_bounds, metadata.bounds):
            continue
        matches.append(
            GroundwaterSource(
                netcdf_path=netcdf_path,
                raster_path=groundwater_subdataset_path(netcdf_path, variable),
                variable=variable,
                units=metadata.units,
                description=metadata.description,
                resolution_x_deg=metadata.resolution_x_deg,
                resolution_y_deg=metadata.resolution_y_deg,
                nodata=metadata.nodata,
            )
        )

    if not matches:
        raise FileNotFoundError("No annual groundwater file overlaps the village_union extent.")
    if len(matches) > 1:
        match_names = ", ".join(match.netcdf_path.name for match in matches)
        raise ValueError(f"Expected one overlapping annual groundwater file, found multiple: {match_names}")
    return matches[0]


def build_groundwater_analysis_grid(
    source_path: str,
    shapefile_path: Path,
    join_column: str,
) -> tuple[AnalysisGrid, gpd.GeoDataFrame]:
    with rasterio.open(source_path) as src:
        unions = load_village_unions(shapefile_path, GROUNDWATER_CRS, join_column)
        base_grid = build_analysis_grid(src, unions.geometry)
    grid = AnalysisGrid(
        window=base_grid.window,
        transform=base_grid.transform,
        crs=GROUNDWATER_CRS,
        width=base_grid.width,
        height=base_grid.height,
        bounds=base_grid.bounds,
    )
    return grid, unions


def groundwater_weighted_mean(values, coverage, weights) -> float:
    values_arr = np.ma.asarray(values, dtype=np.float64)
    weights_arr = np.ma.asarray(weights, dtype=np.float64)
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = (~np.ma.getmaskarray(values_arr)) & (~np.ma.getmaskarray(weights_arr)) & np.isfinite(coverage_arr)
    effective_weights = coverage_arr[valid] * weights_arr.data[valid]
    if effective_weights.size == 0 or effective_weights.sum() == 0:
        return np.nan
    return float(np.dot(values_arr.data[valid], effective_weights) / effective_weights.sum())


def run_exactextract_groundwater(
    raster_path: str,
    weight_raster_path: Path,
    unions: gpd.GeoDataFrame,
    join_column: str,
    metadata_columns: list[str],
) -> pd.DataFrame:
    include_cols = [join_column, *metadata_columns]
    result = exact_extract(
        raster_path,
        unions,
        [
            groundwater_weighted_mean,
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
    missing_cols = [column for column in include_cols if column not in result.columns]
    if missing_cols:
        raise ValueError(f"exactextract output is missing expected columns: {missing_cols}")
    if result[join_column].astype(str).nunique() != len(unions):
        raise ValueError(f"exactextract output has non-unique values in {join_column}.")

    result = result.rename(
        columns={
            "groundwater_weighted_mean": "groundwater_table_depth_annual_mean_m",
            "cropland_area_weight_sum_km2": "cropland_area_weight_sum_km2",
            "contributing_cell_count": "n_groundwater_cells_contributing",
            "intersecting_cell_count": "n_groundwater_cells_intersecting",
        }
    )
    result[join_column] = result[join_column].astype(str)
    result["has_groundwater_data"] = result["n_groundwater_cells_contributing"] > 0
    result["zero_cropland_flag"] = ~result["has_groundwater_data"]
    return result


def default_groundwater_config(project_root: Path, config_path: Path | None = None) -> GroundwaterConfig:
    project_settings = get_project_settings(project_root, config_path)
    groundwater_settings = get_section_settings(project_root, "groundwater", config_path)
    return GroundwaterConfig(
        shapefile_path=project_settings["shapefile_path"],
        groundwater_root=resolve_path(project_root, groundwater_settings["groundwater_root"]),
        lc_tile_paths=project_settings["lc_tile_paths"],
        years=project_settings["years"],
        output_csv=resolve_path(project_root, groundwater_settings["output_csv"]),
        cropland_share_dir=resolve_path(project_root, groundwater_settings["cropland_share_dir"]),
        cropland_weight_dir=resolve_path(project_root, groundwater_settings["cropland_weight_dir"]),
        join_column=project_settings["join_column"],
        cropland_classes=project_settings["cropland_classes"],
        groundwater_variable=str(groundwater_settings.get("groundwater_variable", GROUNDWATER_VARIABLE)),
        lc_start_year=project_settings["lc_start_year"],
    )


def run_groundwater_pipeline(config: GroundwaterConfig) -> pd.DataFrame:
    missing_tiles = [str(path) for path in config.lc_tile_paths.values() if not path.exists()]
    if missing_tiles:
        raise FileNotFoundError(f"Missing LC tile files: {missing_tiles}")

    annual_files = discover_groundwater_annual_files(config.groundwater_root)
    source = select_groundwater_source(
        annual_files,
        config.shapefile_path,
        config.join_column,
        variable=config.groundwater_variable,
    )
    source_raster_path = materialize_groundwater_raster(
        source,
        config.cropland_weight_dir.parent / "groundwater_source_raster.tif",
    )
    grid, unions = build_groundwater_analysis_grid(
        str(source_raster_path),
        config.shapefile_path,
        config.join_column,
    )
    metadata_columns = [column for column in ["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"] if column in unions.columns]

    rows: list[dict[str, object]] = []
    cell_area_raster = build_cell_area_raster(grid)

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
        write_float_raster(cropland_share * cell_area_raster, grid, weight_path)

        year_df = run_exactextract_groundwater(
            str(source_raster_path),
            weight_path,
            unions,
            join_column=config.join_column,
            metadata_columns=metadata_columns,
        )
        year_df["year"] = year
        year_df["groundwater_source_file"] = source.netcdf_path.name
        year_df["groundwater_source_variable"] = source.variable
        year_df["groundwater_units"] = source.units
        year_df["groundwater_description"] = source.description
        year_df["groundwater_resolution_deg"] = round(source.resolution_x_deg, 6)
        rows.extend(year_df.to_dict(orient="records"))

        if not config.persist_intermediate:
            for path in (share_path, weight_path):
                if path.exists():
                    path.unlink()

    result = pd.DataFrame(rows)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv, index=False)
    return result
