"""
Top-level package for DMelon.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


# from . import ml, ocean, plotting, spectral, statistics, utils

__all__ = ["ml", "ocean", "plotting", "spectral", "statistics", "utils"]
