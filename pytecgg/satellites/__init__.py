import warnings

from .ephemeris import prepare_ephemeris
from .positions import satellite_coordinates
from .ipp import calculate_ipp
from .constants import (
    CONSTELLATION_PARAMS,
    EPHEMERIS_FIELDS,
    GNSS_CONSTANTS,
    TOL_KEPLER,
    RE,
)

__all__ = [
    "prepare_ephemeris",
    "satellite_coordinates",
    "calculate_ipp",
    "CONSTELLATION_PARAMS",
    "EPHEMERIS_FIELDS",
    "GNSS_CONSTANTS",
    "TOL_KEPLER",
    "RE",
]


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = custom_formatwarning
