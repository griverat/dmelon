"""
Top-level package for DMelon.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


from . import ocean, plotting, spectral, statistics, utils

__all__ = ["ocean", "plotting", "spectral", "statistics", "utils"]
