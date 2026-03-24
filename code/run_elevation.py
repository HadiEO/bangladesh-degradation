from __future__ import annotations

import argparse
from pathlib import Path

from bangladesh_degradation.config import get_default_config_path
from bangladesh_degradation.static_attributes import default_static_config, run_static_attributes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bangladesh static village_union elevation pipeline.")
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
        help="Delete intermediate ever-cropland, aligned DEM, and weight rasters after the run.",
    )
    args = parser.parse_args()

    config = default_static_config(args.project_root, args.config_path)
    if args.skip_intermediate:
        config = config.__class__(**{**config.__dict__, "persist_intermediate": False})

    df = run_static_attributes(config)
    print(f"Wrote {len(df)} rows to {config.output_csv}")
    print(f"Config used: {args.config_path or get_default_config_path(args.project_root)}")


if __name__ == "__main__":
    main()
