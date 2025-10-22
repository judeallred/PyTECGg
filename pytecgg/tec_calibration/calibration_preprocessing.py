import numpy as np
import polars as pl
from pymap3d import ecef2geodetic

from pytecgg.tec_calibration.modip import extract_modip


def _polynomial_expansion(
    modip_ipp: np.ndarray,
    modip_rec: np.ndarray,
    lon_ipp: np.ndarray,
    lon_rec: np.ndarray,
    max_degree: int,
) -> np.ndarray:
    delta_lon = lon_ipp - lon_rec
    delta_modip = modip_ipp - modip_rec
    const_norm = 1.0 / (1.0 + np.abs(delta_modip) ** (max_degree + 1))

    n_points = modip_ipp.shape[0]
    n_terms = 2 + max_degree
    pterms_matrix = np.zeros((n_points, n_terms))

    # Constant and longitude terms (2 terms)
    pterms_matrix[:, 0] = const_norm
    pterms_matrix[:, 1] = delta_lon * const_norm
    # MoDip terms (max_degree terms)
    for j in range(1, max_degree + 1):
        pterms_matrix[:, 1 + j] = (delta_modip**j) * const_norm

    return pterms_matrix


def _mapping_function(elevation: np.ndarray, h_ipp: float) -> np.ndarray:
    """
    Mapping function to convert slant to vertical TEC.
    """
    return np.cos(
        np.arcsin((6_371 / (6_371 + h_ipp / 1_000)) * np.cos(np.radians(elevation)))
    )


def _preprocessing(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    h_ipp: float,
) -> pl.DataFrame:
    year = df["epoch"][0].year

    modip_ipp = extract_modip(
        coords=(df["lon_ipp"].to_numpy(), df["lat_ipp"].to_numpy()),
        year=year,
        coord_type="geo",
    )
    modip_rec = extract_modip(coords=receiver_position, year=year, coord_type="ecef")[0]
    _, lon_rec, _ = ecef2geodetic(*receiver_position)

    mapping = _mapping_function(df["ele"].to_numpy(), h_ipp)
    gflc_vert = df["gflc_levelled"].to_numpy() * mapping

    return df.with_columns(
        [
            pl.Series("mapping", mapping),
            pl.Series("gflc_vert", gflc_vert),
            pl.Series("modip_ipp", modip_ipp),
            pl.lit(modip_rec).alias("modip_rec"),
            pl.lit(lon_rec).alias("lon_rec"),
        ]
    )
