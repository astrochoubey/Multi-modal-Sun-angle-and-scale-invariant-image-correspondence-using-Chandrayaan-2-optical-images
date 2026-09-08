"""
Logging utilities for lunar registration.
"""

import logging
import sys


def get_logger(name: str = "lunar_registration", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
