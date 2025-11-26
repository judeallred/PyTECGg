from functools import partial
from typing import Any, Callable
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
    **kwargs,
) -> pl.DataFrame:
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
    gnss_system: str,  # TODO
) -> pl.DataFrame:
    """Dispatcher selecting correct GNSS model."""  # TODO docstring

    if gnss_system == "GLONASS":
        coord_func = lambda eph, sv, t, **_: _state_vector_satellite_coordinates(
            eph, sv, t
        )
        return _compute_coordinates(sv_ids, epochs, ephem_dict, coord_func)

    if gnss_system in {"GPS", "Galileo", "QZSS", "BeiDou"}:
        coord_func = lambda eph, sv, t, system, **_: _kepler_satellite_coordinates(
            eph, sv, system, t
        )
        return _compute_coordinates(
            sv_ids, epochs, ephem_dict, coord_func, system=gnss_system
        )
