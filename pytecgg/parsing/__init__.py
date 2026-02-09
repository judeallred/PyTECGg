from pathlib import Path
from typing import Union

import polars as pl

from ..pytecgg import (
    read_rinex_obs as _read_rinex_obs,
    read_rinex_nav as _read_rinex_nav,
)

__all__ = ["read_rinex_obs", "read_rinex_nav"]


def read_rinex_obs(
    path: Union[str, Path],
) -> tuple[pl.DataFrame, tuple[float, float, float], str]:
    """
    Parses a RINEX observation file and returns the extracted observation data as a Polars DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the RINEX observation file (.rnx, .crx, or .gz).

    Returns
    -------
    tuple
        - pl.DataFrame: DataFrame with columns 'epoch', 'sv', 'observable', 'value'
        - tuple[float, float, float]: Receiver's position in ECEF coordinates (meters)
        - str: RINEX version
    """
    path_str = str(path)
    df, rec_pos, rinex_version = _read_rinex_obs(path_str)
    return (
        df.with_columns(pl.col("epoch").dt.replace_time_zone("UTC")),
        rec_pos,
        rinex_version,
    )


def read_rinex_nav(path: Union[str, Path]) -> dict[str, pl.DataFrame]:
    """
    Parses a RINEX navigation file into a dictionary of DataFrames.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the RINEX navigation file.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary keyed by constellation (e.g., 'GPS'), containing
        DataFrames with 'epoch' as datetime[μs, UTC] and orbital parameters.
    """
    path_str = str(path)
    nav_dict = _read_rinex_nav(path_str)
    return {
        const: df.with_columns(pl.col("epoch").dt.replace_time_zone("UTC"))
        for const, df in nav_dict.items()
    }
