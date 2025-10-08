import numpy as np
import polars as pl
from scipy.linalg import qr

from pytecgg.tec_calibration.constants import ALTITUDE_KM
from pytecgg.tec_calibration.calibration_preprocessing import (
    _ensure_R,
    _preprocessing,
    _create_processing_batches,
    _create_processing_batches_fix,
)


def _combine_interval_systems(
    interval_results: list[dict], all_arcs: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine triangular matrices from all intervals.
    """
    if not interval_results:
        raise ValueError("No data to combine")

    # print(f"🔍 DEBUG: {len(interval_results)} intervalli da processare")
    # print(f"🔍 DEBUG: {len(all_arcs)} archi totali")

    n_arcs_total = len(all_arcs)

    # Estimate of the number of equations that can be formed
    total_equations_est = 0
    for res_ in interval_results:
        Rm = _ensure_R(res_["R_matrix"])
        if Rm is None or not isinstance(Rm, np.ndarray):
            continue
        total_equations_est += max(0, Rm.shape[0] - res_["n_poly_terms"] - 1)

    if total_equations_est == 0:
        print("⚠️ No valid equations found in results")
        return np.zeros((0, n_arcs_total)), np.zeros((0,))

    global_bias_matrix = np.zeros((total_equations_est, n_arcs_total))
    global_obs_vector = np.zeros(total_equations_est)

    equation_counter = 0

    for res_ in interval_results:
        R_matrix = res_["R_matrix"]

        # print(f"🔍 DEBUG Intervallo {interval_idx}:")
        # print(f"   R_matrix type: {type(R_matrix)}")

        R_matrix = _ensure_R(R_matrix)
        if R_matrix is None or not isinstance(R_matrix, np.ndarray):
            # print(f"   ❌ Impossibile estrarre R, skippo intervallo {interval_idx}")
            continue
        # print(f"   ✅ R_matrix normalizzata, shape: {getattr(R_matrix, 'shape', None)}")

        arc_list = res_["arcs"]
        n_poly_terms_local = res_["n_poly_terms"]
        # print(f"   n_poly_terms_local: {n_poly_terms_local}")
        # print(f"   arc_list: {len(arc_list)} archi")

        if R_matrix.ndim != 2:
            # print(f"   ⚠️ R_matrix non 2D, skippo")
            continue

        last_nonzero_row = np.where(R_matrix[:, -1] != 0)[0]
        # print(f"   last_nonzero_row: {last_nonzero_row}")

        if len(last_nonzero_row) == 0:
            # print(f"   ⚠️  Nessuna riga non-zero")
            continue

        UR = last_nonzero_row[-1] + 1
        # print(f"   UR: {UR}")

        if UR <= n_poly_terms_local:
            # print(f"   ⚠️  UR ({UR}) <= n_poly_terms ({n_poly_terms_local})")
            continue

        # Extract submatrix for biases + observations
        RR = R_matrix[n_poly_terms_local:UR, n_poly_terms_local:]
        # print(f"   RR shape: {RR.shape}")

        last_nonzero_rr = np.where(RR[:, -1] != 0)[0]
        # print(f"   last_nonzero_rr: {last_nonzero_rr}")

        if len(last_nonzero_rr) == 0:
            # print(f"   ⚠️  RR nessuna riga non-zero")
            continue

        URR = last_nonzero_rr[-1] + 1
        # print(f"   URR: {URR}")

        if URR <= 1:
            # print(f"   ⚠️  URR ({URR}) <= 1")
            continue

        # print(f"   ✅ Aggiungo {URR-1} equazioni")

        # Reconstruct equations
        for row_idx in range(URR - 1):
            if equation_counter >= global_obs_vector.shape[0]:
                # print("   ⚠️ superato spazio allocato, interrompo l'aggiunta")
                break

            global_obs_vector[equation_counter] = RR[row_idx, URR - 1]

            for col_idx in range(row_idx, URR - 1):
                arc_id = arc_list[col_idx]
                if arc_id in all_arcs:
                    arc_position = all_arcs.index(arc_id)
                    global_bias_matrix[equation_counter, arc_position] = RR[
                        row_idx, col_idx
                    ]

            equation_counter += 1

    global_bias_matrix = global_bias_matrix[:equation_counter, :]
    global_obs_vector = global_obs_vector[:equation_counter]

    # print(f"🔗 Combinazione completata: {equation_counter} equazioni")
    return global_bias_matrix, global_obs_vector


def _solve_final_system(
    global_bias_matrix: np.ndarray, global_obs_vector: np.ndarray
) -> np.ndarray:
    """
    Solve the combined system for the biases using QR decomposition.
    """
    if global_bias_matrix.size == 0:
        raise ValueError("Empty system - no valid equations")

    # Combine matrices
    final_system = np.column_stack([global_bias_matrix, global_obs_vector])

    # Final QR decomposition
    R_final = qr(final_system, mode="r")
    R_final = _ensure_R(R_final)

    # Find solvable system
    last_nonzero = np.where(R_final[:, -1] != 0)[0]
    if len(last_nonzero) == 0:
        raise ValueError("The final system is degenerate - no valid equations")
    UR_BC = last_nonzero[-1] + 1

    if UR_BC <= 1:
        raise ValueError("The final system is under-determined - not enough equations")

    # Extract and solve
    RRBC = R_final[: UR_BC - 1, : UR_BC - 1]
    bb = R_final[: UR_BC - 1, -1]

    offsets = np.linalg.lstsq(RRBC, bb, rcond=None)[0]

    print(f"✅ System solved: {len(offsets)} offest have been estimated")
    return offsets


def estimate_bias(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    max_degree: int = 3,
    n_epochs: int = 30,
    h_ipp: float = ALTITUDE_KM,
) -> dict[str, float]:
    """
    Perform GNSS-derived TEC calibration using QR decomposition to estimate
    instrumental offsets for each satellite-receiver arc.

    This function implements a divide-and-conquer approach where data is processed
    in temporal intervals, then combined to solve for arc-specific biases while
    modeling ionospheric variations with polynomial functions.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing GNSS observations with required columns:
        - epoch: observation timestamps
        - id_arc_valid: unique arc identifier
        - ele: satellite elevation angles
        - gflc_levelled: geometry-free linear combination observations #FIXME qui potremmo usare un campo fra gflc di fase e quello livellato, lasciando alla funzione usare il primo che trova
        - lat_ipp: latitude at IPP
        - lon_ipp: longitude at IPP
    receiver_position : tuple[float, float, float]
        ECEF coordinates of the receiver station (x, y, z) in meters
    max_degree : int, optional
        Maximum degree for polynomial expansion of ionospheric model, by default 3
    n_epochs : int, optional
        Number of epochs to process in each batch, by default 30
    h_ipp : float, optional
        Height of Ionospheric Pierce Point in meters, by default 350_000 (350 km)

    Returns
    -------
    dict[str, float]
        Dictionary mapping arc_id to estimated bias values
    """
    # Batch processing
    df_processed = _preprocessing(df, receiver_position=receiver_position, h_ipp=h_ipp)
    batches, all_arcs = _create_processing_batches(
        df_processed, n_epochs=n_epochs, max_degree=max_degree
    )
    # Solve for biases
    global_bias_matrix, global_obs_vector = _combine_interval_systems(batches, all_arcs)
    offsets = _solve_final_system(global_bias_matrix, global_obs_vector)

    return dict(zip(all_arcs, offsets))


### NEW by chatty ###


def _combine_batches_correctly_fix(batches: list[dict], all_arcs: list[str]):
    n_arcs_total = len(all_arcs)
    global_rows = []
    global_obs = []

    arc_to_idx_global = {arc: i for i, arc in enumerate(all_arcs)}

    for batch in batches:
        bias_matrix_local = batch["bias_matrix"]
        observations = batch["observations"]
        arcs_local = batch["arcs"]

        n_obs = bias_matrix_local.shape[0]
        bias_matrix_global = np.zeros((n_obs, n_arcs_total))

        for j, arc_id in enumerate(arcs_local):
            col_idx_global = arc_to_idx_global[arc_id]
            bias_matrix_global[:, col_idx_global] = bias_matrix_local[:, j]

        global_rows.append(bias_matrix_global)
        global_obs.append(observations)

    global_bias_matrix = np.vstack(global_rows)
    global_obs_vector = np.concatenate(global_obs)

    return global_bias_matrix, global_obs_vector


def estimate_bias_fix(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    max_degree: int = 3,
    n_epochs: int = 30,
    h_ipp: float = 350_000,
) -> dict[str, float]:
    df_processed = _preprocessing(df, receiver_position=receiver_position, h_ipp=h_ipp)
    batches, all_arcs = _create_processing_batches_fix(
        df_processed, n_epochs=n_epochs, max_degree=max_degree
    )
    global_bias_matrix, global_obs_vector = _combine_batches_correctly_fix(
        batches, all_arcs
    )
    offsets, residuals, rank, s = np.linalg.lstsq(
        global_bias_matrix, global_obs_vector, rcond=None
    )

    return dict(zip(all_arcs, offsets))
