from __future__ import annotations

import argparse
from pathlib import Path

from bangladesh_degradation.config import get_default_config_path
from bangladesh_degradation.pipeline import default_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bangladesh salinity cropland extraction pipeline.")
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
        "--skip-intermediate",
        action="store_true",
        help="Do not keep intermediate cropland-share or exactextract-weight rasters after each year is processed.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use the Python fallback implementation instead of exactextract.",
    )
    args = parser.parse_args()

    config = default_config(args.project_root, args.config_path)
    if args.skip_intermediate:
        config = config.__class__(
            **{
                **config.__dict__,
                "persist_cropland_share": False,
                "persist_cropland_weights": False,
            }
        )
    if args.fallback:
        config = config.__class__(**{**config.__dict__, "use_exactextract": False})

    df = run_pipeline(config)
    print(f"Wrote {len(df)} rows to {config.output_csv}")
    print(f"Config used: {args.config_path or get_default_config_path(args.project_root)}")


if __name__ == "__main__":
    main()
