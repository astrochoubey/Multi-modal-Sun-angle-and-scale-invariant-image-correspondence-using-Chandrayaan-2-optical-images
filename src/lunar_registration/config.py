"""
Configuration loader and validator for lunar image registration.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and parse a YAML pipeline configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration format in {config_path}")

    return config
