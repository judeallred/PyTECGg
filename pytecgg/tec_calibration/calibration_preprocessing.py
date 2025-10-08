import numpy as np
import polars as pl
from scipy.linalg import qr
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
    return 1.0 / np.cos(
        np.arcsin(
            (6_371 / (6_371 + h_ipp / 1_000)) * np.sin(np.radians(90 - elevation))
        )
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
    gflc_vert = df["gflc_levelled"].to_numpy() / mapping

    return df.with_columns(
        [
            pl.Series("mapping", mapping),
            pl.Series("gflc_vert", gflc_vert),
            pl.Series("modip_ipp", modip_ipp),
            pl.lit(modip_rec).alias("modip_rec"),
            pl.lit(lon_rec).alias("lon_rec"),
        ]
    )


def _ensure_R(qr_result):
    """
    Normalisation of scipy.linalg.qr result to return R as ndarray.
    """
    if isinstance(qr_result, np.ndarray):
        return qr_result
    if isinstance(qr_result, tuple):
        for elem in reversed(qr_result):
            if isinstance(elem, np.ndarray) and getattr(elem, "ndim", 0) >= 2:
                return elem
        for elem in qr_result:
            if isinstance(elem, np.ndarray):
                return elem
        return qr_result[0]
    return qr_result


def _create_processing_batches(
    df_preprocessed: pl.DataFrame, n_epochs: int = 30, max_degree: int = 3
) -> tuple[list[dict], list[str]]:
    """
    Create processing batches from preprocessed data.
    Returns both the batch results and complete arc list.
    """
    times = df_preprocessed["epoch"].unique().sort()
    batches: list[dict] = []
    all_arcs: list[str] = []
    seen_arcs: set = set()

    for start_idx in range(0, len(times), n_epochs):
        end_idx = min(start_idx + n_epochs, len(times))
        selected_epochs = times[start_idx:end_idx]
        batch_data = df_preprocessed.filter(pl.col("epoch").is_in(selected_epochs))

        if batch_data.is_empty():
            continue

        arcs_summary = (
            batch_data.group_by("id_arc_valid")
            .agg(
                [
                    pl.col("modip_ipp").first(),
                    pl.col("modip_rec").first(),
                    pl.col("lon_ipp").first(),
                    pl.col("lon_rec").first(),
                    pl.col("mapping").first(),
                    pl.col("gflc_vert").first(),
                ]
            )
            .filter(
                pl.col("modip_ipp").is_not_null()
                & pl.col("modip_ipp").is_finite()
                & pl.col("modip_rec").is_finite()
                & pl.col("mapping").is_finite()
                & pl.col("gflc_vert").is_finite()
            )
        )

        if arcs_summary.is_empty():
            continue

        arc_ids = arcs_summary["id_arc_valid"].to_list()
        modip_ipp = arcs_summary["modip_ipp"].to_numpy()
        modip_rec = arcs_summary["modip_rec"].to_numpy()
        lon_ipp = arcs_summary["lon_ipp"].to_numpy()
        lon_rec = arcs_summary["lon_rec"].to_numpy()
        mapping_funcs = arcs_summary["mapping"].to_numpy()
        observations = arcs_summary["gflc_vert"].to_numpy()

        poly_matrix = _polynomial_expansion(
            modip_ipp, modip_rec, lon_ipp, lon_rec, max_degree
        )

        # Bias matrix: one-hot encoding of arcs
        n_arcs = len(arc_ids)
        n_obs = arcs_summary.height
        bias_matrix = np.zeros((n_obs, n_arcs))
        arc_to_idx = {arc_id: i for i, arc_id in enumerate(arc_ids)}

        for i, arc_id in enumerate(arc_ids):
            bias_matrix[i, arc_to_idx[arc_id]] = mapping_funcs[i]

        # Combined system: [Poly terms | Bias terms | Observations]
        system_matrix = np.column_stack(
            [poly_matrix, bias_matrix, observations.reshape(-1, 1)]
        )

        # QR decomposition
        R = qr(system_matrix, mode="r")
        R = _ensure_R(R)

        if R is not None and isinstance(R, np.ndarray):
            batches.append(
                {"R_matrix": R, "arcs": arc_ids, "n_poly_terms": poly_matrix.shape[1]}
            )

            # Update arc registry
            for arc in arc_ids:
                if arc not in seen_arcs:
                    all_arcs.append(arc)
                    seen_arcs.add(arc)

    return batches, all_arcs


### NEW ###


def _create_processing_batches_fix(
    df_preprocessed: pl.DataFrame, n_epochs: int = 30, max_degree: int = 3
) -> tuple[list[dict], list[str]]:
    times = df_preprocessed["epoch"].unique().sort()
    batches: list[dict] = []
    all_arcs: list[str] = []
    seen_arcs: set = set()

    for start_idx in range(0, len(times), n_epochs):
        end_idx = min(start_idx + n_epochs, len(times))
        selected_epochs = times[start_idx:end_idx]
        batch_data = df_preprocessed.filter(pl.col("epoch").is_in(selected_epochs))

        if batch_data.is_empty():
            continue

        arcs_summary = (
            batch_data.group_by("id_arc_valid")
            .agg(
                [
                    pl.col("modip_ipp").first(),
                    pl.col("modip_rec").first(),
                    pl.col("lon_ipp").first(),
                    pl.col("lon_rec").first(),
                    pl.col("mapping").first(),
                    pl.col("gflc_vert").first(),
                ]
            )
            .filter(
                pl.col("modip_ipp").is_not_null()
                & pl.col("modip_ipp").is_finite()
                & pl.col("modip_rec").is_finite()
                & pl.col("mapping").is_finite()
                & pl.col("gflc_vert").is_finite()
            )
        )

        if arcs_summary.is_empty():
            continue

        arc_ids = arcs_summary["id_arc_valid"].to_list()
        modip_ipp = arcs_summary["modip_ipp"].to_numpy()
        modip_rec = arcs_summary["modip_rec"].to_numpy()
        lon_ipp = arcs_summary["lon_ipp"].to_numpy()
        lon_rec = arcs_summary["lon_rec"].to_numpy()
        mapping_funcs = arcs_summary["mapping"].to_numpy()
        observations = arcs_summary["gflc_vert"].to_numpy()

        poly_matrix = _polynomial_expansion(
            modip_ipp, modip_rec, lon_ipp, lon_rec, max_degree
        )

        n_arcs = len(arc_ids)
        n_obs = arcs_summary.height
        bias_matrix = np.zeros((n_obs, n_arcs))
        arc_to_idx = {arc_id: i for i, arc_id in enumerate(arc_ids)}

        for i, arc_id in enumerate(arc_ids):
            bias_matrix[i, arc_to_idx[arc_id]] = mapping_funcs[i]

        batches.append(
            {
                "arcs": arc_ids,
                "n_poly_terms": poly_matrix.shape[1],
                "poly_matrix": poly_matrix,
                "bias_matrix": bias_matrix,
                "observations": observations,
            }
        )

        for arc in arc_ids:
            if arc not in seen_arcs:
                all_arcs.append(arc)
                seen_arcs.add(arc)

    return batches, all_arcs
