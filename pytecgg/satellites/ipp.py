from typing import Tuple

import numpy as np
from pymap3d import ecef2geodetic, ecef2aer

from .constants import RE


def calculate_ipp(
    rec_ecef: Tuple[float, float, float],
    sat_ecef_array: np.ndarray,
    h_ipp: float = 350_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Ionospheric Pierce Point (IPP) location.

    Parameters:
    - rec_ecef: Receiver ECEF coordinates (x, y, z) in meters
    - sat_ecef_array: (N,3) array of satellite ECEF coordinates in meters
    - h_ipp: Mean height of the ionosphere shell in meters, default is 350.000 meters

    Returns:
    - Tuple of (N,) NumPy arrays containing:
        - lat: Latitude of IPP in degrees (None, if calculation fails)
        - lon: Longitude of IPP in degrees (None, if calculation fails)
        - azi: Azimuth angle from receiver to satellite in degrees (None, if calculation fails)
        - ele: Elevation angle from receiver to satellite in degrees (None, if calculation fails)
    """
    if h_ipp < 0:
        raise ValueError("h_ipp must be non-negative")

    if sat_ecef_array.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xA, yA, zA = map(np.asarray, rec_ecef)
    xB, yB, zB = sat_ecef_array[:, 0], sat_ecef_array[:, 1], sat_ecef_array[:, 2]

    dx, dy, dz = xB - xA, yB - yA, zB - zA

    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * xA + dy * yA + dz * zA)
    c = xA**2 + yA**2 + zA**2 - (RE + h_ipp) ** 2

    disc = b**2 - 4 * a * c
    mask = disc >= 0

    # Init arrays with NaN
    size_ = sat_ecef_array.shape[0]
    lat_ipp = np.full(size_, np.nan, dtype=float)
    lon_ipp = np.full(size_, np.nan, dtype=float)
    azi = np.full(size_, np.nan, dtype=float)
    ele = np.full(size_, np.nan, dtype=float)

    # If no valid solutions, return NaNs
    if not np.any(mask):
        return lat_ipp, lon_ipp, azi, ele

    # Compute valid solutions
    sqrt_disc = np.sqrt(disc[mask])
    denom = 2 * a[mask]

    t1 = (-b[mask] + sqrt_disc) / denom
    t2 = (-b[mask] - sqrt_disc) / denom

    # Choose valid t (0 <= t <= 1), preferring the smaller one
    t1_valid = (0 <= t1) & (t1 <= 1)
    t2_valid = (0 <= t2) & (t2 <= 1)

    t = np.select(
        [t1_valid & t2_valid, t1_valid, t2_valid],
        [np.minimum(t1, t2), t1, t2],
        default=np.nan,
    )

    valid = ~np.isnan(t)
    if not np.any(valid):
        return lat_ipp, lon_ipp, azi, ele

    # Compute IPP coordinates
    idx_out = np.flatnonzero(mask)[valid]
    x_ipp = xA + dx[mask][valid] * t[valid]
    y_ipp = yA + dy[mask][valid] * t[valid]
    z_ipp = zA + dz[mask][valid] * t[valid]

    # Convert to geodetic coordinates
    latv, lonv, _ = ecef2geodetic(x_ipp, y_ipp, z_ipp)
    lat_ipp[idx_out] = latv
    lon_ipp[idx_out] = lonv

    # Compute azimuth and elevation
    rec_geodetic = ecef2geodetic(*rec_ecef)
    aziv, elev, _ = ecef2aer(
        xB[mask][valid], yB[mask][valid], zB[mask][valid], *rec_geodetic, deg=True
    )
    azi[idx_out] = aziv
    ele[idx_out] = elev

    return lat_ipp, lon_ipp, azi, ele
