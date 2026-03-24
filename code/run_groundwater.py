from __future__ import annotations

import argparse
from pathlib import Path

from bangladesh_degradation.config import get_default_config_path
from bangladesh_degradation.groundwater_table_depth import (
    default_groundwater_config,
    run_groundwater_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bangladesh groundwater table depth pipeline.")
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
        "--years",
        nargs="+",
        type=int,
        help="Optional list of years to process instead of the full 2000-2018 range.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate cropland-share and cropland-weight rasters.",
    )
    args = parser.parse_args()

    config = default_groundwater_config(args.project_root, args.config_path)
    if args.years:
        config = config.__class__(**{**config.__dict__, "years": tuple(sorted(set(args.years)))})
    if args.keep_intermediate:
        config = config.__class__(**{**config.__dict__, "persist_intermediate": True})

    df = run_groundwater_pipeline(config)
    print(f"Wrote {len(df)} rows to {config.output_csv}")
    print(f"Config used: {args.config_path or get_default_config_path(args.project_root)}")


if __name__ == "__main__":
    main()
