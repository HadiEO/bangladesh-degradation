from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
try:
    from exactextract import exact_extract
except ImportError:  # pragma: no cover
    exact_extract = None
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, box

from bangladesh_degradation.config import get_project_settings, get_section_settings, resolve_path


ECE_FILENAME_RE = re.compile(r"ECe_(\d{4})\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class PipelineConfig:
    shapefile_path: Path
    ece_dirs: tuple[Path, ...]
    lc_tile_paths: Mapping[str, Path]
    years: tuple[int, ...]
    output_csv: Path
    cropland_share_dir: Path
    cropland_weight_dir: Path
    join_column: str = "ADM4_PCODE"
    cropland_classes: frozenset[int] = frozenset({10, 11, 12, 20})
    ece_scale: float = 100.0
    lc_start_year: int = 2000
    persist_cropland_share: bool = True
    persist_cropland_weights: bool = True
    use_exactextract: bool = True


@dataclass(frozen=True)
class AnalysisGrid:
    window: Window
    transform: Affine
    crs: object
    width: int
    height: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class CellCoverage:
    row: int
    col: int
    area_m2: float


@dataclass(frozen=True)
class UnionCoverage:
    coverages: tuple[CellCoverage, ...]
    total_intersecting_pixels: int
    partial_extent: bool


def clamp_window(window: Window, width: int, height: int) -> Window:
    """Clamp a raster window to valid raster bounds."""
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_end = min(width, int(math.ceil(window.col_off + window.width)))
    row_end = min(height, int(math.ceil(window.row_off + window.height)))
    clamped_width = max(0, col_end - col_off)
    clamped_height = max(0, row_end - row_off)
    return Window(col_off, row_off, clamped_width, clamped_height)


def discover_ece_files(ece_dirs: Iterable[Path], years: Iterable[int]) -> tuple[dict[int, Path], list[int]]:
    """Discover yearly ECe files and report missing years."""
    discovered: dict[int, Path] = {}
    for directory in ece_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("ECe_*.tif"):
            match = ECE_FILENAME_RE.search(path.name)
            if match:
                discovered[int(match.group(1))] = path
    expected = list(years)
    missing = [year for year in expected if year not in discovered]
    filtered = {year: discovered[year] for year in expected if year in discovered}
    return filtered, missing


def get_lc_band_index(year: int, lc_start_year: int = 2000) -> int:
    """Map a calendar year to the 1-based band index in the annual LC stacks."""
    return year - lc_start_year + 1


def load_village_unions(shapefile_path: Path, target_crs: object, join_column: str) -> gpd.GeoDataFrame:
    """Load and reproject the union polygons to the analysis CRS."""
    gdf = gpd.read_file(shapefile_path)
    if join_column not in gdf.columns:
        raise KeyError(f"Missing required join column: {join_column}")
    if not gdf[join_column].is_unique:
        raise ValueError(f"Join column {join_column} is not unique.")
    if gdf.crs is None:
        raise ValueError("Village_union shapefile CRS is missing.")
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def build_analysis_grid(reference_src: rasterio.io.DatasetReader, geometries: gpd.GeoSeries, pad_cells: int = 1) -> AnalysisGrid:
    """Build the ECe subset grid covering all village_unions."""
    minx, miny, maxx, maxy = geometries.total_bounds
    padded_bounds = (
        minx - pad_cells * abs(reference_src.transform.a),
        miny - pad_cells * abs(reference_src.transform.e),
        maxx + pad_cells * abs(reference_src.transform.a),
        maxy + pad_cells * abs(reference_src.transform.e),
    )
    window = from_bounds(*padded_bounds, transform=reference_src.transform)
    window = clamp_window(window, reference_src.width, reference_src.height)
    transform = reference_src.window_transform(window)
    bounds = array_bounds(int(window.height), int(window.width), transform)
    return AnalysisGrid(
        window=window,
        transform=transform,
        crs=reference_src.crs,
        width=int(window.width),
        height=int(window.height),
        bounds=bounds,
    )


def _cell_polygon(transform: Affine, row: int, col: int) -> Polygon:
    left = transform.c + col * transform.a
    top = transform.f + row * transform.e
    right = left + transform.a
    bottom = top + transform.e
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def _polygonal_area_geographic(geometry, geod: Geod) -> float:
    if geometry.is_empty:
        return 0.0
    geom_type = geometry.geom_type
    if geom_type in {"Polygon", "MultiPolygon"}:
        return abs(geod.geometry_area_perimeter(geometry)[0])
    if hasattr(geometry, "geoms"):
        return float(sum(_polygonal_area_geographic(part, geod) for part in geometry.geoms))
    return 0.0


def geometry_area_square_meters(geometry, crs: object, geod: Geod | None = None) -> float:
    """Compute polygon area in square meters for projected or geographic CRS."""
    if geometry.is_empty:
        return 0.0
    if hasattr(crs, "is_geographic") and crs.is_geographic:
        if geod is None:
            raise ValueError("Geod is required for geographic CRS area calculations.")
        return _polygonal_area_geographic(geometry, geod)
    return float(geometry.area)


def precompute_union_coverages(
    gdf: gpd.GeoDataFrame,
    grid: AnalysisGrid,
    join_column: str,
) -> dict[str, UnionCoverage]:
    """Precompute area of each village_union intersected with each ECe cell."""
    grid_bounds_polygon = box(*grid.bounds)
    geod = Geod(ellps="WGS84") if hasattr(grid.crs, "is_geographic") and grid.crs.is_geographic else None
    coverages: dict[str, UnionCoverage] = {}

    for row in gdf.itertuples(index=False):
        union_id = getattr(row, join_column)
        geometry = row.geometry
        partial_extent = not grid_bounds_polygon.contains(geometry)
        clipped = geometry.intersection(grid_bounds_polygon)

        if clipped.is_empty:
            coverages[str(union_id)] = UnionCoverage(coverages=(), total_intersecting_pixels=0, partial_extent=True)
            continue

        bbox_window = from_bounds(*clipped.bounds, transform=grid.transform)
        bbox_window = clamp_window(bbox_window, grid.width, grid.height)
        cell_coverages: list[CellCoverage] = []
        total_pixels = 0

        for row_idx in range(int(bbox_window.row_off), int(bbox_window.row_off + bbox_window.height)):
            for col_idx in range(int(bbox_window.col_off), int(bbox_window.col_off + bbox_window.width)):
                cell = _cell_polygon(grid.transform, row_idx, col_idx)
                if not clipped.intersects(cell):
                    continue
                total_pixels += 1
                intersection = clipped.intersection(cell)
                area_m2 = geometry_area_square_meters(intersection, grid.crs, geod=geod)
                if area_m2 > 0:
                    cell_coverages.append(CellCoverage(row=row_idx, col=col_idx, area_m2=area_m2))

        coverages[str(union_id)] = UnionCoverage(
            coverages=tuple(cell_coverages),
            total_intersecting_pixels=total_pixels,
            partial_extent=partial_extent,
        )
    return coverages


def build_cropland_share_raster(
    lc_tile_paths: Mapping[str, Path],
    year: int,
    grid: AnalysisGrid,
    cropland_classes: Iterable[int],
    lc_start_year: int = 2000,
) -> np.ndarray:
    """Aggregate 30 m annual land cover to cropland share on the ECe grid subset."""
    band_index = get_lc_band_index(year, lc_start_year=lc_start_year)
    cropland_classes = set(cropland_classes)
    dest_sum = np.zeros((grid.height, grid.width), dtype=np.float32)
    dest_count = np.zeros((grid.height, grid.width), dtype=np.uint8)

    for tile_path in lc_tile_paths.values():
        with rasterio.open(tile_path) as src:
            if band_index < 1 or band_index > src.count:
                raise ValueError(f"Year {year} is outside LC band range for {tile_path.name}.")
            band = src.read(band_index, masked=True)
            binary = np.isin(band.filled(255), list(cropland_classes)).astype(np.float32)
            binary[np.ma.getmaskarray(band)] = np.nan

            tile_dest = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
            reproject(
                source=binary,
                destination=tile_dest,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            valid = ~np.isnan(tile_dest)
            dest_sum[valid] += tile_dest[valid]
            dest_count[valid] += 1

    result = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    has_data = dest_count > 0
    result[has_data] = dest_sum[has_data] / dest_count[has_data]
    return result


def build_cell_area_raster(grid: AnalysisGrid) -> np.ndarray:
    """Build a raster of full ECe cell area in square meters."""
    areas = np.zeros((grid.height, grid.width), dtype=np.float32)
    if hasattr(grid.crs, "is_geographic") and grid.crs.is_geographic:
        geod = Geod(ellps="WGS84")
        row_areas = np.zeros(grid.height, dtype=np.float32)
        for row_idx in range(grid.height):
            row_areas[row_idx] = float(
                geometry_area_square_meters(_cell_polygon(grid.transform, row_idx, 0), grid.crs, geod=geod)
            )
        areas[:] = row_areas[:, None]
    else:
        cell_area = float(abs(grid.transform.a * grid.transform.e))
        areas.fill(cell_area)
    return areas


def build_cropland_weight_raster(cropland_share: np.ndarray, grid: AnalysisGrid) -> np.ndarray:
    """Convert cropland share on the ECe grid to approximate cropland area per cell."""
    return cropland_share * build_cell_area_raster(grid)


def read_ece_subset(ece_path: Path, grid: AnalysisGrid, ece_scale: float) -> np.ndarray:
    """Read a year-specific ECe subset aligned to the analysis grid."""
    with rasterio.open(ece_path) as src:
        if src.crs != grid.crs:
            raise ValueError(f"ECe CRS mismatch for {ece_path.name}: {src.crs} != {grid.crs}")
        data = src.read(1, window=grid.window).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
    return data / ece_scale


def write_float_raster(array: np.ndarray, grid: AnalysisGrid, output_path: Path) -> None:
    """Persist an intermediate float raster aligned to the ECe subset grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=np.nan,
        compress="LZW",
    ) as dst:
        dst.write(array.astype(np.float32), 1)


def weighted_mean_dsm(values, coverage, weights) -> float:
    """Exactextract PythonOperation for cropland-area-weighted ECe mean in dS/m."""
    values_arr = np.ma.asarray(values, dtype=np.float64)
    weights_arr = np.ma.asarray(weights, dtype=np.float64)
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = (~np.ma.getmaskarray(values_arr)) & (~np.ma.getmaskarray(weights_arr)) & np.isfinite(coverage_arr)
    effective_weights = coverage_arr[valid] * weights_arr.data[valid]
    if effective_weights.size == 0 or effective_weights.sum() == 0:
        return np.nan
    return float(np.dot(values_arr.data[valid] / 100.0, effective_weights) / effective_weights.sum())


def cropland_area_weight_sum_km2(values, coverage, weights) -> float:
    """Exactextract PythonOperation returning weighted cropland area in km^2."""
    values_arr = np.ma.asarray(values)
    weights_arr = np.ma.asarray(weights, dtype=np.float64)
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = (~np.ma.getmaskarray(values_arr)) & (~np.ma.getmaskarray(weights_arr)) & np.isfinite(coverage_arr)
    if not valid.any():
        return 0.0
    return float(np.sum(coverage_arr[valid] * weights_arr.data[valid]) / 1_000_000.0)


def contributing_cell_count(values, coverage, weights) -> int:
    """Exactextract PythonOperation counting valid intersecting cells with positive cropland weight."""
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
    return int(np.count_nonzero(valid))


def intersecting_cell_count(values, coverage) -> int:
    """Exactextract PythonOperation counting all geometrically intersecting cells."""
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    valid = np.isfinite(coverage_arr) & (coverage_arr > 0)
    return int(np.count_nonzero(valid))


def partial_extent_flags(unions: gpd.GeoDataFrame, grid: AnalysisGrid, join_column: str) -> dict[str, bool]:
    """Flag village_unions that are not fully contained in the ECe analysis bounds."""
    grid_bounds_polygon = box(*grid.bounds)
    return {
        str(row[join_column]): (not grid_bounds_polygon.contains(row.geometry))
        for _, row in unions.iterrows()
    }


def validate_ece_grid_alignment(ece_lookup: Mapping[int, Path], years: Iterable[int]) -> None:
    """Ensure all yearly ECe rasters share the same grid definition."""
    years = list(years)
    with rasterio.open(ece_lookup[years[0]]) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
    for year in years[1:]:
        with rasterio.open(ece_lookup[year]) as src:
            same_transform = all(
                np.isclose(a, b, atol=1e-12, rtol=0.0)
                for a, b in zip(ref_transform[:6], src.transform[:6])
            )
            same_grid = (
                src.crs == ref_crs
                and src.width == ref_width
                and src.height == ref_height
                and same_transform
            )
        if not same_grid:
            raise ValueError(f"ECe grid mismatch detected for year {year}.")


def run_exactextract_year(
    ece_path: Path,
    cropland_weight_raster: Path,
    unions: gpd.GeoDataFrame,
    join_column: str,
    metadata_columns: list[str],
    partial_extent: Mapping[str, bool],
) -> pd.DataFrame:
    """Run exactextract for one year using cropland-area weights."""
    if exact_extract is None:
        raise RuntimeError("exactextract is not available in the active Python environment.")

    include_cols = [join_column, *metadata_columns]
    result = exact_extract(
        str(ece_path),
        unions,
        [
            weighted_mean_dsm,
            cropland_area_weight_sum_km2,
            contributing_cell_count,
            intersecting_cell_count,
        ],
        weights=str(cropland_weight_raster),
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
            "weighted_mean_dsm": "ece_mean_cropland",
            "cropland_area_weight_sum_km2": "cropland_area_weight_sum_km2",
            "contributing_cell_count": "n_ece_pixels_contributing",
            "intersecting_cell_count": "n_ece_pixels_intersecting",
        }
    )
    result[join_column] = result[join_column].astype(str)
    result["has_data"] = result["n_ece_pixels_contributing"] > 0
    result["zero_cropland_flag"] = ~result["has_data"]
    result["partial_extent_flag"] = result[join_column].map(partial_extent)
    return result


def extract_union_stats(
    ece_array: np.ndarray,
    cropland_share_array: np.ndarray,
    coverage: UnionCoverage,
) -> dict[str, object]:
    """Compute cropland-share-weighted ECe statistics for one village_union."""
    weights: list[float] = []
    values: list[float] = []

    for cell in coverage.coverages:
        ece_value = float(ece_array[cell.row, cell.col])
        crop_share = float(cropland_share_array[cell.row, cell.col])
        if np.isnan(ece_value) or np.isnan(crop_share) or crop_share <= 0:
            continue
        weight = cell.area_m2 * crop_share
        if weight <= 0:
            continue
        weights.append(weight)
        values.append(ece_value)

    if not weights:
        return {
            "ece_mean_cropland": np.nan,
            "cropland_area_weight_sum_km2": 0.0,
            "n_ece_pixels_contributing": 0,
            "n_ece_pixels_intersecting": coverage.total_intersecting_pixels,
            "has_data": False,
            "zero_cropland_flag": True,
            "partial_extent_flag": coverage.partial_extent,
        }

    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weighted_mean = float(np.dot(values_arr, weights_arr) / weights_arr.sum())
    return {
        "ece_mean_cropland": round(weighted_mean, 6),
        "cropland_area_weight_sum_km2": round(float(weights_arr.sum() / 1_000_000.0), 6),
        "n_ece_pixels_contributing": int(len(weights)),
        "n_ece_pixels_intersecting": coverage.total_intersecting_pixels,
        "has_data": True,
        "zero_cropland_flag": False,
        "partial_extent_flag": coverage.partial_extent,
    }


def default_config(project_root: Path, config_path: Path | None = None) -> PipelineConfig:
    """Return the default project configuration."""
    project_settings = get_project_settings(project_root, config_path)
    salinity_settings = get_section_settings(project_root, "salinity", config_path)
    return PipelineConfig(
        shapefile_path=project_settings["shapefile_path"],
        ece_dirs=tuple(resolve_path(project_root, path) for path in salinity_settings["ece_dirs"]),
        lc_tile_paths=project_settings["lc_tile_paths"],
        years=project_settings["years"],
        output_csv=resolve_path(project_root, salinity_settings["output_csv"]),
        cropland_share_dir=resolve_path(project_root, salinity_settings["cropland_share_dir"]),
        cropland_weight_dir=resolve_path(project_root, salinity_settings["cropland_weight_dir"]),
        join_column=project_settings["join_column"],
        cropland_classes=project_settings["cropland_classes"],
        ece_scale=float(salinity_settings.get("ece_scale", 100.0)),
        lc_start_year=project_settings["lc_start_year"],
    )


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Run the full village_union ECe extraction workflow."""
    ece_lookup, missing_years = discover_ece_files(config.ece_dirs, config.years)
    if missing_years:
        raise FileNotFoundError(f"Missing ECe files for years: {missing_years}")
    validate_ece_grid_alignment(ece_lookup, config.years)

    missing_tiles = [str(path) for path in config.lc_tile_paths.values() if not path.exists()]
    if missing_tiles:
        raise FileNotFoundError(f"Missing LC tile files: {missing_tiles}")

    reference_year = min(config.years)
    with rasterio.open(ece_lookup[reference_year]) as ref_src:
        unions = load_village_unions(config.shapefile_path, ref_src.crs, config.join_column)
        grid = build_analysis_grid(ref_src, unions.geometry)
        coverages = precompute_union_coverages(unions, grid, config.join_column)
        partial_extent = partial_extent_flags(unions, grid, config.join_column)

    rows: list[dict[str, object]] = []
    metadata_columns = [column for column in ["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"] if column in unions.columns]

    for year in config.years:
        cropland_share = build_cropland_share_raster(
            config.lc_tile_paths,
            year,
            grid,
            config.cropland_classes,
            lc_start_year=config.lc_start_year,
        )
        if config.persist_cropland_share:
            write_float_raster(
                cropland_share,
                grid,
                config.cropland_share_dir / f"cropland_share_{year}.tif",
            )

        if config.use_exactextract and exact_extract is not None:
            cropland_weight = build_cropland_weight_raster(cropland_share, grid)
            weight_path = config.cropland_weight_dir / f"cropland_weight_{year}.tif"
            write_float_raster(cropland_weight, grid, weight_path)

            year_df = run_exactextract_year(
                ece_lookup[year],
                weight_path,
                unions,
                join_column=config.join_column,
                metadata_columns=metadata_columns,
                partial_extent=partial_extent,
            )
            year_df["year"] = year
            rows.extend(year_df.to_dict(orient="records"))
            if not config.persist_cropland_weights and weight_path.exists():
                weight_path.unlink()
        else:
            ece_array = read_ece_subset(ece_lookup[year], grid, config.ece_scale)
            for row in unions.itertuples(index=False):
                union_id = str(getattr(row, config.join_column))
                result = extract_union_stats(ece_array, cropland_share, coverages[union_id])
                output_row = {
                    config.join_column: union_id,
                    "year": year,
                    **result,
                }
                for column in metadata_columns:
                    output_row[column] = getattr(row, column)
                rows.append(output_row)

    output_df = pd.DataFrame(rows)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config.output_csv, index=False)
    return output_df
