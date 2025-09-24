import datetime
import warnings
from pathlib import Path
from importlib.resources import files
from typing import Sequence, Union

import numpy as np
from ppigrf import igrf
from scipy.interpolate import RegularGridInterpolator

from .constants import ALTITUDE_KM, LONGITUDES, LATITUDES


def _calculate_modip_grid(year: int, altitude_km: float = ALTITUDE_KM) -> np.ndarray:
    """
    Calculate MODIP grid for a specific year.

    MODIP (Modified Dip Latitude) is calculated from the IGRF magnetic field model.

    Parameters:
    -----------
    year : int
        Year for magnetic field calculation
    altitude_km : float
        Altitude in kilometers above Earth's surface

    Returns:
    --------
    tuple: (modip_grid, longitudes, latitudes)
        MODIP grid values and corresponding longitude/latitude arrays
    """
    dates = datetime.datetime(year, 1, 1)
    lon_grid, lat_grid = np.meshgrid(LONGITUDES, LATITUDES, indexing="ij")

    # Magnetic field components (east, north, upward) using IGRF model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Be, Bn, Bu = igrf(lon_grid, lat_grid, altitude_km, dates)

    # Horizontal component and the magnetic inclination
    Bh = np.sqrt(Be**2 + Bn**2)
    inclination_rad = -np.arctan2(Bu, Bh)

    # MODIP (Modified Dip Latitude)
    lat_rad = np.deg2rad(lat_grid)
    modip = np.rad2deg(np.arctan(inclination_rad / np.sqrt(np.cos(lat_rad))))

    # At the poles, cos(latitude) approaches zero
    modip = _polar_nan_values(modip, lat_grid)

    return modip[0]


def _polar_nan_values(modip: np.ndarray, lats_grid: np.ndarray) -> np.ndarray:
    """
    Handle NaN values at the poles by replacing them with ±90°.

    At the poles, the calculation involves division by very small numbers
    which results in NaN values. These are replaced with the appropriate
    pole values (+90° for North pole, -90° for South pole).

    Parameters:
    -----------
    modip_deg : np.ndarray
        MODIP grid with potential NaN values at poles
    lats_grid : np.ndarray
        Latitude grid for identifying pole locations

    Returns:
    --------
    modip_deg_clean : np.ndarray
        MODIP grid with NaN values replaced by pole values
    """
    for i in range(modip.shape[0]):
        nan_mask = np.isnan(modip[i])
        modip[i][nan_mask] = np.sign(lats_grid[nan_mask]) * 90.0
    return modip


def _save_modip_grid(modip_grid: np.ndarray, year: int) -> None:
    outdir = Path(__file__).parent / "modip_grids"
    outfile = outdir / f"modip_{year}.npz"
    np.savez_compressed(outfile, modip_grid, allow_pickle=False)


def _load_modip_grid(year: int) -> np.ndarray:
    fname = f"modip_{year}.npz"
    modip_grids = files("pytecgg.tec_calibration.modip_grids").joinpath(fname)

    try:
        with modip_grids.open("rb") as f:
            npz = np.load(f, allow_pickle=False)
            return npz["arr_0"]
    except FileNotFoundError:
        return _calculate_modip_grid(year)


def extract_modip(
    coords: Union[
        tuple[np.ndarray, np.ndarray],
        Sequence[tuple[float, float]],
        np.ndarray,
    ],
    modip_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate MODIP values for specific coordinates. Accepts either:
        - Two arrays (lon, lat) as a tuple
        - A single array-like of shape (N, 2) with (lon, lat) pairs

    Uses linear interpolation to calculate MODIP values at arbitrary
    longitude/latitude points based on a precomputed grid.

    Parameters
    ----------
    coords : tuple of arrays or array-like
        Either (lon_array, lat_array) or array-like [[lon1, lat1], [lon2, lat2], ...]
    modip_grid : np.ndarray
        Precomputed MODIP grid

    Returns:
    --------
    modip_values : np.ndarray
        Interpolated MODIP values for the specified coordinates
    """
    if isinstance(coords, tuple) and len(coords) == 2:
        lon, lat = coords
        points = np.column_stack([lon, lat])
    else:
        points = np.array(coords)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "coords must be either a tuple (lon, lat) or an array of shape (N, 2)"
            )

    interpolator = RegularGridInterpolator(
        (LONGITUDES, LATITUDES),
        modip_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return interpolator(points)
