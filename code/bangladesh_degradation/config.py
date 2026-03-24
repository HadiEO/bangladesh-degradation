from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/paths.yml")


def get_default_config_path(project_root: Path) -> Path:
    return project_root / DEFAULT_CONFIG_PATH


def load_settings(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    resolved_config_path = config_path.resolve() if config_path else get_default_config_path(project_root)
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved_config_path}. "
            "Create it from config/paths.example.yml or pass --config-path."
        )

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    if not isinstance(settings, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {resolved_config_path}")
    return settings


def _require_mapping(settings: dict[str, Any], key: str) -> dict[str, Any]:
    value = settings.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping in the YAML config.")
    return value


def _require_value(settings: dict[str, Any], key: str) -> Any:
    if key not in settings:
        raise ValueError(f"Missing required config key: {key}")
    return settings[key]


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def get_project_settings(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    settings = load_settings(project_root, config_path)
    project = _require_mapping(settings, "project")
    landcover = _require_mapping(settings, "landcover")
    landcover_tiles = _require_mapping(landcover, "tiles")

    years = tuple(int(year) for year in _require_value(project, "years"))
    cropland_classes = frozenset(int(value) for value in project.get("cropland_classes", [10, 11, 12, 20]))

    return {
        "shapefile_path": resolve_path(project_root, _require_value(project, "shapefile_path")),
        "join_column": str(project.get("join_column", "ADM4_PCODE")),
        "years": years,
        "cropland_classes": cropland_classes,
        "lc_start_year": int(project.get("lc_start_year", 2000)),
        "lc_tile_paths": {
            str(tile_name): resolve_path(project_root, tile_path)
            for tile_name, tile_path in landcover_tiles.items()
        },
    }


def get_section_settings(project_root: Path, section_name: str, config_path: Path | None = None) -> dict[str, Any]:
    settings = load_settings(project_root, config_path)
    return _require_mapping(settings, section_name)

