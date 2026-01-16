from datetime import datetime, timedelta, timezone
from typing import Any, Union

import polars as pl

from .constants import CONSTELLATION_PARAMS

Ephem = dict[str, Union[dict[str, Any], list[dict[str, Any]]]]


def _parse_time(time_str: str, time_system: str, time_offset: timedelta) -> datetime:
    """
    Parse RINEX time string with time system awareness.

    Parameters
    ----------
    time_str : str
        Time string from RINEX file.
    time_system : str
        Time system identifier ('GPST', 'BDT', etc.).
    time_offset : timedelta
        Offset to apply for conversion to UTC.

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.
    """
    if isinstance(time_str, str):
        # Remove time system suffix if present
        clean_str = time_str.split(f" {time_system}")[0].strip()

        try:
            # Parse naive datetime
            dt = datetime.fromisoformat(clean_str)

            # Apply time system offset and convert to UTC
            if time_offset:
                dt = dt - time_offset

            # Make timezone-aware (UTC)
            return dt.replace(tzinfo=timezone.utc)

        except ValueError as e:
            raise ValueError(f"Failed to parse time string '{time_str}': {e}")

    raise TypeError(f"Unsupported time format: {type(time_str)}")


def _greg2gps(dt: datetime) -> tuple[int, float]:
    """
    Convert Gregorian date to GPS week and seconds.

    Parameters
    ----------
    dt : datetime
        Datetime object to convert.

    Returns
    -------
    tuple[int, float]
        (GPS week, GPS seconds).
    """
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = dt - epoch
    return (
        delta.days // 7,
        (delta.days % 7) * 86400 + delta.seconds + delta.microseconds / 1e6,
    )


def prepare_ephemeris(nav: dict[str, pl.DataFrame], constellation: str) -> Ephem:
    """
    Prepare ephemeris data for the specified constellation from RINEX navigation data.

    The logic depends on the orbit propagation model required:

    1.  Keplerian Orbits (GPS, Galileo, QZSS, BeiDou):
        Only a single 'central' ephemeris message of the day is selected per satellite.

    2.  State-Vector Orbits (GLONASS):
        All available ephemeris messages are collected for the satellite. This is
        required because GLONASS messages contain instantaneous state vectors (position/
        velocity/acceleration) valid only for short periods (typically ± 15 minutes),
        requiring numerical integration from the closest epoch.

    Parameters:
    ----------
    nav : dict[str, pl.DataFrame]
        Dictionary containing navigation data (Polars DataFrames) from a RINEX
        file, keyed by constellation identifier (e.g., 'GPS', 'GLONASS').
    constellation : str
        GNSS constellation name (e.g., 'GPS', 'GLONASS').

    Returns:
    -------
    Ephem
        Dictionary with prepared ephemeris data, keyed by normalized satellite ID
        (e.g., 'R09', 'G01').

        * For Keplerian systems: `dict[sv_id, dict[str, Any]]` (a single ephemeris dictionary).
        * For GLONASS: `dict[sv_id, list[dict[str, Any]]]` (a list of all available ephemeris dictionaries for the day).
    """

    if constellation not in CONSTELLATION_PARAMS or constellation not in nav:
        return {}

    params = CONSTELLATION_PARAMS[constellation]
    ephem_dict: Ephem = {}

    is_state_vector = constellation == "GLONASS"

    for sat_id in nav[constellation]["sv"].unique().to_list():
        normalised_sat_id = f"{params.prefix}{int(sat_id):02d}"
        sat_data = nav[constellation].filter(pl.col("sv") == sat_id)
        if sat_data.is_empty():
            continue

        if is_state_vector:
            sat_data = sat_data.sort("epoch")
            sat_ephems_list = []

            for row in sat_data.to_dicts():
                ephe_time = _parse_time(
                    row["epoch"], params.time_system, params.time_offset
                )
                gps_week, gps_sec = _greg2gps(ephe_time)

                ephem = {
                    "constellation": constellation,
                    "sv": normalised_sat_id,
                    "datetime": ephe_time,
                    "gps_week": gps_week,
                    "gps_seconds": gps_sec,
                    **{field: row.get(field) for field in params.fields},
                }
                sat_ephems_list.append(ephem)

            ephem_dict[normalised_sat_id] = sat_ephems_list

        else:
            # GPS, Galileo, BeiDou: only the central ephemeris (Keplerian approach)
            ephe_row = sat_data.row(len(sat_data) // 2, named=True)

            ephe_time = _parse_time(
                ephe_row["epoch"], params.time_system, params.time_offset
            )
            gps_week, gps_sec = _greg2gps(ephe_time)

            # Base structure
            ephem = {
                "constellation": constellation,
                "sv": normalised_sat_id,
                "datetime": ephe_time,
                "gps_week": gps_week,
                "gps_seconds": gps_sec,
                **{field: ephe_row.get(field) for field in params.fields},
            }
            ephem_dict[normalised_sat_id] = ephem

    return ephem_dict
