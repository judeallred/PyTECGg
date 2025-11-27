from typing import Any, Callable, Literal

import numpy as np
import polars as pl

from pytecgg.satellites.kepler.coordinates import _kepler_satellite_coordinates
from pytecgg.satellites.state_vector.coordinates import (
    _state_vector_satellite_coordinates,
)


def _compute_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict: dict[str, dict[str, Any]],
    coord_func: Callable[..., np.ndarray],
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Compute satellite coordinates for multiple satellites and epochs.

    Parameters
    ----------
    sv_ids : pl.Series
        Series containing satellite identifiers (e.g., 'G01', 'E23', 'R01')
    epochs : pl.Series
        Series containing observation times as datetime objects
    ephem_dict : dict[str, dict[str, Any]]
        Dictionary containing ephemeris data for all satellites
    coord_func : Callable[..., np.ndarray]
        Function to compute coordinates for a single satellite and epoch
    **kwargs : Any
        Additional keyword arguments to pass to the coordinate function

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: 'sv', 'epoch', 'sat_x', 'sat_y', 'sat_z'
        containing satellite ECEF coordinates in meters.
    """
    sv_arr = sv_ids.to_numpy()
    size = len(sv_arr)

    x = np.full(size, np.nan)
    y = np.full(size, np.nan)
    z = np.full(size, np.nan)

    for i, (sv, epoch) in enumerate(zip(sv_arr, epochs)):
        try:
            if sv not in ephem_dict:
                continue

            pos = coord_func(ephem_dict, sv, epoch, **kwargs)
            if pos.size == 3:
                x[i], y[i], z[i] = pos

        except Exception as e:
            print(f"Error processing {sv} at {epoch}: {e}")
            continue

    return pl.DataFrame(
        {
            "sv": sv_arr,
            "epoch": epochs,
            "sat_x": x,
            "sat_y": y,
            "sat_z": z,
        }
    )


def satellite_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict: dict[str, dict[str, Any]],
    gnss_system: Literal["GPS", "Galileo", "QZSS", "BeiDou", "GLONASS"],
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Compute Earth-Centered Earth-Fixed (ECEF) coordinates for GNSS satellites.

    The function supports GPS, Galileo, QZSS, BeiDou (using Keplerian orbits)
    and GLONASS (using state-vector propagation).

    Parameters
    ----------
    sv_ids : pl.Series
        Series containing satellite identifiers (e.g., 'G01', 'E23', 'R01')
    epochs : pl.Series
        Series containing observation times as datetime objects
    ephem_dict : dict[str, dict[str, Any]]
        Dictionary containing ephemeris data for all satellites
    gnss_system : Literal["GPS", "Galileo", "QZSS", "BeiDou", "GLONASS"]
        GNSS constellation identifier
    **kwargs : Any
        Additional parameters for GLONASS state-vector propagation:
        - t_res : float, optional
            Time resolution for ODE solver in seconds (default: 15.0)
        - error_estimate : Literal["coarse", "normal", "fine"], optional
            Error tolerance level:
            - "coarse": ~2000 meters precision, faster
            - "normal": ~200 meters precision, balanced (default)
            - "fine": ~20 meters precision, slower

        These parameters are ignored for non-GLONASS systems.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: 'sv', 'epoch', 'sat_x', 'sat_y', 'sat_z'
        containing satellite ECEF coordinates in meters.
    """

    if gnss_system == "GLONASS":
        coord_func = _state_vector_satellite_coordinates
    elif gnss_system in {"GPS", "Galileo", "QZSS", "BeiDou"}:
        coord_func = lambda ephem, sv, t, **kw: _kepler_satellite_coordinates(
            ephem, sv, gnss_system, t, **kw
        )
    return _compute_coordinates(sv_ids, epochs, ephem_dict, coord_func, **kwargs)
