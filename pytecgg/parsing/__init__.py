import polars as pl

from ..pytecgg import read_rinex_obs as _read_rinex_obs, read_rinex_nav

__all__ = ["read_rinex_obs", "read_rinex_nav"]


def read_rinex_obs(path: str):
    """
    Parses a RINEX observation file and returns the extracted observation data as a Polars DataFrame.

    Parameters
    ----------
    path : str
        Path to the RINEX observation file.

    Returns
    -------
    tuple
        - pl.DataFrame: DataFrame with columns 'epoch', 'sv', 'observable', 'value'
        - tuple[float, float, float]: Receiver's position in ECEF coordinates (meters)
        - str: RINEX version

    Notes
    -----
    This is a Python wrapper around the Rust implementation. The 'epoch' column is automatically
    converted from string to datetime in UTC.
    """
    df, rec_pos, rinex_version = _read_rinex_obs(path)
    return (
        df.with_columns(pl.col("epoch").dt.replace_time_zone("UTC")),
        rec_pos,
        rinex_version,
    )
