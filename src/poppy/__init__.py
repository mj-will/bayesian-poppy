"""
bayesian-poppy: Bayesian posterior post-processing in python
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .poppy import Poppy

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Poppy",
]
