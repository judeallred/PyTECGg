import warnings
from collections import defaultdict, Counter
from typing import Any, Callable, Literal, Union

import numpy as np
import polars as pl

from pytecgg.satellites.kepler.coordinates import _kepler_satellite_coordinates
from pytecgg.satellites.state_vector.coordinates import (
    _state_vector_satellite_coordinates,
)


def _compute_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_data: Union[dict[str, Any], pl.DataFrame],
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
    ephem_data : Union[dict[str, Any], pl.DataFrame]
        Dictionary (Keplerian) or DataFrame (GLONASS) containing ephemeris data.
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

    x, y, z = np.full(size, np.nan), np.full(size, np.nan), np.full(size, np.nan)

    # Track missing ephemeris for logging
    total_eph_counts = Counter(sv_arr)
    missing_eph_counts = defaultdict(int)

    # GLONASS
    if isinstance(ephem_data, pl.DataFrame):
        data_rows = ephem_data.to_dicts()
        for i, row in enumerate(data_rows):

            sv_id = row["sv"]
            # If satPosX is None, no ephemeris was found within tolerance
            if row.get("satPosX") is None:
                missing_eph_counts[sv_id] += 1
                continue

            try:
                pos = coord_func(row, row["epoch"], **kwargs)
                if pos is not None and pos.size == 3:
                    x[i], y[i], z[i] = pos
            except Exception:
                missing_eph_counts[sv_id] += 1
                continue

        if missing_eph_counts:
            lines = [
                f"  {sv:3}: {lost:4}/{total_eph_counts[sv]:4} epochs lost ({(lost/total_eph_counts[sv])*100:5.1f}%)"
                for sv, lost in sorted(missing_eph_counts.items())
            ]
            summary_report = "\n".join(lines)
            warnings.warn(
                f"\nGLONASS Missing ephemeris summary:\n{summary_report}\n"
                f"Satellite positions set to NaN (check RINEX NAV coverage).",
                RuntimeWarning,
            )

    # Keplerian systems (GPS, Galileo, BeiDou)
    else:
        for i, (sv, epoch) in enumerate(zip(sv_arr, epochs)):
            if sv not in ephem_data:
                missing_eph_counts[sv] += 1
                continue

            try:
                pos = coord_func(ephem_data, sv, epoch, **kwargs)
                if pos is not None and pos.size == 3:
                    x[i], y[i], z[i] = pos
                else:
                    missing_eph_counts[sv] += 1
            except Exception:
                missing_eph_counts[sv] += 1
                continue

        if missing_eph_counts:
            summary = ", ".join(sorted(missing_eph_counts.keys()))
            warnings.warn(
                f"Missing central ephemeris for satellites: {summary}. "
                "Satellite positions set to NaN.",
                RuntimeWarning,
            )

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
    ephem_dict: dict[str, Union[dict[str, Any], list[dict[str, Any]]]],
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
    ephem_dict : dict
        Dictionary containing ephemeris data
        Expected format: {sv_id: dict} for Keplerian systems or {sv_id: list[dict]}
        for GLONASS.
    gnss_system : str
        GNSS constellation identifier ('GPS', 'Galileo', 'QZSS', 'BeiDou', 'GLONASS')
    **kwargs : Any
        Additional parameters for GLONASS state-vector propagation:
        - t_res : float or None, optional
            Time resolution (in seconds) for sampling the ODE solver solution
            - if float: the trajectory is sampled at fixed intervals
            - if None (default): the solver selects internal time steps automatically.
        - error_estimate : Literal["coarse", "normal", "fine"], optional
            Error tolerance level:
            - "coarse": ~2000 meters precision, faster
            - "normal": ~200 meters precision, balanced (default)
            - "fine": ~20 meters precision, slower

        These additional parameters are ignored for non-GLONASS systems

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: 'sv', 'epoch', 'sat_x', 'sat_y', 'sat_z'
        containing satellite ECEF coordinates in meters
    """

    if gnss_system == "GLONASS":
        eph_list = []
        for sv_ in ephem_dict:
            eph_list.extend(ephem_dict[sv_])

        df_ = pl.DataFrame({"sv": sv_ids, "epoch": epochs}).sort("epoch")
        df_eph = pl.DataFrame(eph_list).sort("datetime")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sortedness of columns cannot be checked when 'by' groups provided",
            )
            df_joined = df_.join_asof(
                df_eph,
                left_on="epoch",
                right_on="datetime",
                by="sv",
                strategy="nearest",
                tolerance="20m",
            )

        return _compute_coordinates(
            sv_ids, epochs, df_joined, _state_vector_satellite_coordinates, **kwargs
        )

    else:
        coord_func = lambda ephem, sv, t, **kw: _kepler_satellite_coordinates(
            ephem, sv, gnss_system, t, **kw
        )
        return _compute_coordinates(sv_ids, epochs, ephem_dict, coord_func, **kwargs)
