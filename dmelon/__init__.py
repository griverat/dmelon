# -*- coding: utf-8 -*-
"""
Top-level package for DMelon.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


from . import ocean, spectral, statistics


__all__ = ["ocean", "spectral", "statistics"]
