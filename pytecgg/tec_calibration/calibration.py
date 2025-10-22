import numpy as np
import polars as pl
from scipy.linalg import qr

from pytecgg.tec_calibration.constants import ALTITUDE_M
from pytecgg.tec_calibration.calibration_preprocessing import (
    _polynomial_expansion,
    _preprocessing,
)


def _gg_calibration(
    df_clean: pl.DataFrame,
    interval: int = 30,
    nmax: int = 3,
):
    # 1. Loop over batches of interval epochs
    epoch_times = df_clean["epoch"].unique().sort()
    qr_results_dict = {}
    arc_lists_dict = {}
    global_arcs_list = []
    global_arc_idx_map = {}

    # 2. Loop over epochs in the batch
    num_epochs = len(epoch_times)
    for batch_start_idx in range(0, num_epochs, interval):
        batch_arcs_list = []
        batch_arc_idx_map = {}
        design_matrix_rows = []
        observations = []
        arc_columns = []
        mapping_values = []

        batch_epoch_indices = range(
            max(0, batch_start_idx - interval), min(batch_start_idx, num_epochs)
        )

        for epoch_idx in batch_epoch_indices:
            current_time = epoch_times[epoch_idx]
            epoch_data = df_clean.filter(pl.col("epoch") == current_time)

            if epoch_data.is_empty():
                continue

            valid_arcs = (
                epoch_data.filter(pl.col("id_arc_valid").is_not_null())["id_arc_valid"]
                .unique(maintain_order=True)
                .to_list()
            )

            for arc_id in valid_arcs:
                arc_data = epoch_data.filter(pl.col("id_arc_valid") == arc_id)

                if arc_data.is_empty():
                    continue

                mapping = arc_data["mapping"][0]
                obs_val = arc_data["gflc_vert"][0]

                polynomial_terms = _polynomial_expansion(
                    arc_data["modip_ipp"].to_numpy(),
                    arc_data["modip_rec"].to_numpy(),
                    arc_data["lon_ipp"].to_numpy(),
                    arc_data["lon_rec"].to_numpy(),
                    nmax,
                )

                design_matrix_rows.append(polynomial_terms)

                # Batch arcs
                if arc_id not in batch_arc_idx_map:
                    batch_arc_idx_map[arc_id] = len(batch_arcs_list)
                    batch_arcs_list.append(arc_id)

                # Global arcs
                if arc_id not in global_arc_idx_map:
                    global_arc_idx_map[arc_id] = len(global_arcs_list)
                    global_arcs_list.append(arc_id)

                arc_columns.append(batch_arc_idx_map[arc_id])
                mapping_values.append(mapping)
                observations.append(obs_val)

        if not design_matrix_rows:
            continue

        design_matrix = np.vstack(design_matrix_rows)
        observation_vector = np.array(observations).reshape(-1, 1)

        num_arcs_in_batch = len(batch_arcs_list)
        num_observations = len(observations)
        beta_matrix = np.zeros((num_observations, num_arcs_in_batch))

        for i, arc_col_idx in enumerate(arc_columns):
            beta_matrix[i, arc_col_idx] = mapping_values[i]

        full_matrix = np.hstack([design_matrix, beta_matrix, observation_vector])
        _, R_matrix = qr(full_matrix, mode="full")

        qr_results_dict[str(batch_start_idx)] = R_matrix
        arc_lists_dict[str(batch_start_idx)] = batch_arcs_list

    num_global_arcs = len(global_arcs_list)
    num_coefficients = nmax + 2

    total_rows = sum(len(arcs) for arcs in arc_lists_dict.values())
    design_matrix_global = np.zeros((total_rows, num_global_arcs))
    observation_vector_global = np.zeros(total_rows)

    current_row = 0

    for batch_key, R_matrix in qr_results_dict.items():
        batch_arcs = arc_lists_dict[batch_key]

        total_cols = R_matrix.shape[1]
        relevant_block = R_matrix[num_coefficients:total_cols, num_coefficients:]

        num_block_rows, num_block_cols = relevant_block.shape

        for row_idx in range(num_block_rows - 1):
            observation_vector_global[current_row] = relevant_block[
                row_idx, num_block_cols - 1
            ]

            for col_idx in range(row_idx, num_block_cols - 1):
                arc_id = batch_arcs[col_idx]
                global_idx = global_arc_idx_map[arc_id]
                design_matrix_global[current_row, global_idx] = relevant_block[
                    row_idx, col_idx
                ]

            current_row += 1

    final_augmented = np.hstack(
        [design_matrix_global, observation_vector_global.reshape(-1, 1)]
    )

    _, final_R = qr(final_augmented, mode="full")

    num_final_eqs = final_R.shape[1] - 1
    triangular_system = final_R[:num_final_eqs, :-1]
    rhs_vector = final_R[:num_final_eqs, -1]

    biases = np.linalg.solve(triangular_system, rhs_vector)

    return biases, global_arcs_list


def estimate_bias(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    max_degree: int = 3,
    n_epochs: int = 30,
    h_ipp: float = ALTITUDE_M,
):
    df_clean = _preprocessing(df, receiver_position=receiver_position, h_ipp=h_ipp)
    bias, id_arc = _gg_calibration(df_clean, interval=n_epochs, nmax=max_degree)
    return bias, id_arc
