from typing import Any
import datetime
import logging

import numpy as np
import polars as pl

from pytecgg.satellites.kepler.coordinates import _kepler_satellite_coordinates
from pytecgg.satellites.state_vector.coordinates import (
    _state_vector_satellite_coordinates,
)


def _glonass_coordinates(
    ephem_dict: dict[str, dict[str, Any]],
    sv_id: str,
    obs_time: datetime.datetime | None = None,
) -> np.ndarray:
    """
    Uniform interface to compute ECEF coordinates of GLONASS satellite
    """
    if sv_id not in ephem_dict:
        raise KeyError(f"Satellite {sv_id} not found in ephemeris data")

    if obs_time is not None and obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=datetime.timezone.utc)

    try:
        pos, _ = _state_vector_satellite_coordinates(ephem_dict, sv_id, obs_time)
        return pos
    except (KeyError, ValueError) as e:
        logging.warning(f"Warning while processing GLONASS satellite coordinates: {e}")
        return np.array([], dtype=float)


def _satellite_coordinates_sv(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict,
) -> list[pl.Expr]:
    """
    Compute GNSS satellite positions in Earth-Centered Earth-Fixed (ECEF) coordinates
    for multiple epochs using broadcast ephemeris

    Parameters:
    ----------
    sv_ids : pl.Series
        Polars Series containing satellite identifiers (e.g., 'G12', 'E19') for each epoch
    epochs : pl.Series
        Polars Series of datetime values (in ns precision) corresponding to each satellite observation
    ephem_dict : dict[str, dict[str, Any]]
        Dictionary containing broadcast ephemeris parameters for each satellite

    Returns:
    -------
    list[pl.Expr]
        A list of three Polars expressions representing the ECEF coordinates in meters
    """
    sv_arr = sv_ids.to_numpy()
    size_ = sv_arr.shape[0]
    x = np.full(size_, np.nan, dtype=float)
    y = np.full(size_, np.nan, dtype=float)
    z = np.full(size_, np.nan, dtype=float)

    for i, (sv, epoch_) in enumerate(zip(sv_arr, epochs)):
        try:
            if sv not in ephem_dict:
                continue

            pos = _glonass_coordinates(ephem_dict, sv, epoch_)
            if pos.size > 0:
                x[i], y[i], z[i] = pos
        except Exception as e:
            print(f"Error processing {sv} at {epoch_}: {str(e)}")
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


def _satellite_coordinates_kp(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict: dict[str, dict[str, Any]],
    gnss_system: str,
) -> list[pl.Expr]:
    """
    Compute GNSS satellite positions in Earth-Centered Earth-Fixed (ECEF) coordinates
    for multiple epochs using broadcast ephemeris

    Parameters:
    ----------
    sv_ids : pl.Series
        Polars Series containing satellite identifiers (e.g., 'G12', 'E19') for each epoch
    epochs : pl.Series
        Polars Series of datetime values (in ns precision) corresponding to each satellite observation
    ephem_dict : dict[str, dict[str, Any]]
        Dictionary containing broadcast ephemeris parameters for each satellite
    gnss_system : str
        GNSS constellation ('GPS', 'Galileo', 'QZSS' or 'BeiDou')

    Returns:
    -------
    list[pl.Expr]
        A list of three Polars expressions representing the ECEF coordinates in meters
    """
    sv_arr = sv_ids.to_numpy()
    time_arr = epochs.dt.cast_time_unit("ns").to_numpy()

    size_ = sv_arr.shape[0]
    x = np.full(size_, np.nan, dtype=float)
    y = np.full(size_, np.nan, dtype=float)
    z = np.full(size_, np.nan, dtype=float)

    for i, (sv, epoch_ns) in enumerate(zip(sv_arr, time_arr)):
        try:
            if sv not in ephem_dict:
                continue

            epoch_dt = datetime.datetime.fromtimestamp(
                epoch_ns.astype("int64") / 1e9, datetime.timezone.utc
            )

            pos = _kepler_satellite_coordinates(ephem_dict, sv, gnss_system, epoch_dt)
            if pos.size > 0:
                x[i], y[i], z[i] = pos
        except Exception as e:
            print(f"Error processing {sv} at {epoch_dt}: {str(e)}")
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
    gnss_system: str,
) -> np.ndarray:
    """Dispatcher"""

    if gnss_system in ["GPS", "Galileo", "QZSS", "BeiDou"]:
        return _satellite_coordinates_kp(
            sv_ids=sv_ids, epochs=epochs, ephem_dict=ephem_dict, gnss_system=gnss_system
        )
    elif gnss_system == "GLONASS":
        return _satellite_coordinates_sv(
            sv_ids=sv_ids, epochs=epochs, ephem_dict=ephem_dict
        )
