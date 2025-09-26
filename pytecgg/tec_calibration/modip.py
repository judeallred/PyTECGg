import datetime
import warnings
from pathlib import Path
from importlib.resources import files
from typing import Sequence, Union, Literal

import numpy as np
from ppigrf import igrf
from pymap3d import ecef2geodetic
from scipy.interpolate import RegularGridInterpolator

from .constants import ALTITUDE_KM, LONGITUDES, LATITUDES


def _calculate_modip_grid(year: int, altitude_km: float = ALTITUDE_KM) -> np.ndarray:
    """
    Calculate MoDip grid for a specific year.

    MoDip (Modified Dip Latitude) is calculated from the IGRF magnetic field model.

    Parameters:
    -----------
    year : int
        Year for magnetic field calculation
    altitude_km : float
        Altitude in kilometers above Earth's surface

    Returns:
    --------
    np.ndarray
        MoDip grid values
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

    # MoDip (Modified Dip Latitude)
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
        MoDip grid with potential NaN values at poles
    lats_grid : np.ndarray
        Latitude grid for identifying pole locations

    Returns:
    --------
    modip : np.ndarray
        MoDip grid with NaN values replaced by pole values
    """
    for i in range(modip.shape[0]):
        nan_mask = np.isnan(modip[i])
        modip[i][nan_mask] = np.sign(lats_grid[nan_mask]) * 90.0
    return modip


def _save_modip_grid(modip_grid: np.ndarray, year: int) -> None:
    """
    Save a MoDip grid for a given year to a compressed .npz file.
    For internal use to precompute MoDip grids distributed with the package.
    """
    outdir = Path(__file__).parent / "modip_grids"
    outfile = outdir / f"modip_{year}.npz"
    np.savez_compressed(outfile, modip_grid, allow_pickle=False)


def _load_modip_grid(year: int) -> np.ndarray:
    """
    Load a precomputed MoDip grid for a given year, or compute it if missing.
    """
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
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        Sequence[tuple[float, float, float]],
        np.ndarray,
    ],
    year: int,
    coord_type: Literal["ecef", "geo"],
) -> np.ndarray:
    """
    Interpolate MODIP values for given coordinates in ECEF (x, y, z).

    The function converts input ECEF coordinates to geodetic
    latitude/longitude and interpolates MODIP values from the
    precomputed grid.

    Parameters
    ----------
    coords : tuple of arrays, sequence of tuples, or np.ndarray
        Input coordinates in ECEF system. Either:
        - Tuple of arrays (x, y, z),
        - Sequence of (x, y, z) tuples,
        - Array of shape (N, 3) with (x, y, z).
    year : int
        Year for which the MODIP grid is used.
    coord_type : Literal["ecef", "geo"]
        Type of input coordinates:
        - "ecef" for ECEF coordinates.
        - "geo" for geographic coordinates; lon, lat expected.
    Returns:
    --------
    np.ndarray
        Interpolated MoDip values for the specified coordinates
    """
    if coord_type == "geo":
        if isinstance(coords, tuple) and len(coords) == 2:
            lon, lat = coords
            points_geo = np.column_stack([lon, lat])
        else:
            points_geo = np.array(coords)
            if points_geo.ndim != 2 or points_geo.shape[1] != 2:
                raise ValueError(
                    "For coord_type='geo', coords must be tuple (lon, lat) or array of shape (N, 2)"
                )
    elif coord_type == "ecef":
        if isinstance(coords, tuple) and len(coords) == 3:
            x, y, z = coords
            points_ecef = np.column_stack([x, y, z])
        else:
            points_ecef = np.array(coords)
            if points_ecef.ndim != 2 or points_ecef.shape[1] != 3:
                raise ValueError(
                    "coords must be either a tuple (x, y, z) or an array of shape (N, 3)"
                )
        lat, lon, _ = ecef2geodetic(
            points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2]
        )
        points_geo = np.column_stack([lon, lat])

    modip_grid = _load_modip_grid(year)
    interpolator = RegularGridInterpolator(
        (LONGITUDES, LATITUDES),
        modip_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return interpolator(points_geo).round(2)
