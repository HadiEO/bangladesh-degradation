from __future__ import annotations

import argparse
from pathlib import Path

from bangladesh_degradation.groundwater_table_depth import (
    default_groundwater_config,
    run_groundwater_pipeline,
)
from bangladesh_degradation.pipeline import default_config, run_pipeline
from bangladesh_degradation.static_attributes import (
    default_static_config,
    run_static_attributes,
)
from bangladesh_degradation.weather_terraclimate import (
    default_weather_config,
    run_weather_pipeline,
)


PIPELINE_CHOICES = ("salinity", "elevation", "groundwater", "weather")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run shapefile-based processing to a dedicated run label and output namespace."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Optional YAML config path. Defaults to <project-root>/config/paths.yml.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Path to the alternate shapefile to process.",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help="Run label used for the output directory, intermediate directory, and filename suffixes.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=PIPELINE_CHOICES,
        default=PIPELINE_CHOICES,
        help="Subset of pipelines to run. Defaults to all pipelines.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    project_root = args.project_root.resolve()
    shapefile = args.shapefile.resolve()
    run_label = args.run_label
    output_dir = project_root / "output" / run_label
    intermediate_root = project_root / "data" / "intermediate" / run_label

    if "salinity" in args.pipelines:
        config = default_config(project_root, args.config_path)
        config = config.__class__(
            **{
                **config.__dict__,
                "shapefile_path": shapefile,
                "output_csv": output_dir / f"village_union_cropland_salinity_2000_2018_{run_label}.csv",
                "cropland_share_dir": intermediate_root / "cropland_share_ece_grid",
                "cropland_weight_dir": intermediate_root / "cropland_weight_ece_grid",
            }
        )
        df = run_pipeline(config)
        print(f"salinity_output={config.output_csv}")
        print(f"salinity_rows={len(df)}")

    if "elevation" in args.pipelines:
        config = default_static_config(project_root, args.config_path)
        config = config.__class__(
            **{
                **config.__dict__,
                "shapefile_path": shapefile,
                "output_csv": output_dir / f"village_union_elevation_{run_label}.csv",
                "ever_cropland_path": intermediate_root / "ever_cropland.tif",
                "dem_aligned_path": intermediate_root / "nasadem_aligned_to_cropland.tif",
                "weight_raster_path": intermediate_root / "ever_cropland_weights.tif",
            }
        )
        df = run_static_attributes(config)
        print(f"elevation_output={config.output_csv}")
        print(f"elevation_rows={len(df)}")

    if "groundwater" in args.pipelines:
        config = default_groundwater_config(project_root, args.config_path)
        config = config.__class__(
            **{
                **config.__dict__,
                "shapefile_path": shapefile,
                "output_csv": output_dir / f"village_union_groundwater_table_depth_{run_label}.csv",
                "cropland_share_dir": intermediate_root / "cropland_share_groundwater_grid",
                "cropland_weight_dir": intermediate_root / "cropland_weight_groundwater_grid",
            }
        )
        df = run_groundwater_pipeline(config)
        print(f"groundwater_output={config.output_csv}")
        print(f"groundwater_rows={len(df)}")

    if "weather" in args.pipelines:
        config = default_weather_config(project_root, args.config_path)
        config = config.__class__(
            **{
                **config.__dict__,
                "shapefile_path": shapefile,
                "monthly_output_csv": output_dir / f"village_union_weather_monthly_2000_2018_{run_label}.csv",
                "annual_output_csv": output_dir / f"village_union_weather_annual_2000_2018_{run_label}.csv",
                "cropland_share_dir": intermediate_root / "cropland_share_terraclimate_grid",
                "cropland_weight_dir": intermediate_root / "cropland_weight_terraclimate_grid",
                "monthly_raster_dir": intermediate_root / "terraclimate_monthly_rasters",
            }
        )
        monthly_df, annual_df = run_weather_pipeline(config)
        print(f"weather_monthly_output={config.monthly_output_csv}")
        print(f"weather_monthly_rows={len(monthly_df)}")
        print(f"weather_annual_output={config.annual_output_csv}")
        print(f"weather_annual_rows={len(annual_df)}")


if __name__ == "__main__":
    main()
