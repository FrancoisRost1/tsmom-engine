"""
Load YAML configuration and return as dict.

Single entry point for all config access across the project.
"""

import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml and return as a plain dict.

    Args:
        path: Path to the YAML config file, relative or absolute.

    Returns:
        Dict with all configuration parameters.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_all_tickers(config: dict) -> list[str]:
    """Extract flat list of all tickers from the universe config.

    Args:
        config: Full config dict.

    Returns:
        List of ticker strings, e.g. ['SPY', 'EFA', ...].
    """
    universe = config["universe"]
    tickers = []
    for asset_class in ["equities", "bonds", "commodities", "fx"]:
        tickers.extend(universe.get(asset_class, []))
    return tickers


def get_asset_class_map(config: dict) -> dict[str, str]:
    """Build ticker → asset class mapping.

    Args:
        config: Full config dict.

    Returns:
        Dict mapping ticker to asset class name,
        e.g. {'SPY': 'equities', 'TLT': 'bonds', ...}.
    """
    universe = config["universe"]
    mapping = {}
    for asset_class in ["equities", "bonds", "commodities", "fx"]:
        for ticker in universe.get(asset_class, []):
            mapping[ticker] = asset_class
    return mapping
