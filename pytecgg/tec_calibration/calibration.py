import numpy as np
import polars as pl
from scipy.linalg import qr
from pymap3d import ecef2geodetic

from pytecgg.tec_calibration.modip import extract_modip
from pytecgg.tec_calibration.polynomial_expansion import polynomial_expansion


def _ensure_R(qr_result):
    """
    Normalizza il risultato di scipy.linalg.qr restituendo R come ndarray.
    Gestisce tuple nidificate e casi (R,), (Q, R), (Q, R, P), ecc.
    """
    if isinstance(qr_result, np.ndarray):
        return qr_result
    if isinstance(qr_result, tuple):
        # cerca dall'ultimo elemento un ndarray con dim>=2 (probabile R)
        for elem in reversed(qr_result):
            if isinstance(elem, np.ndarray) and getattr(elem, "ndim", 0) >= 2:
                return elem
        # fallback: se ci sono ndarray prendi il primo
        for elem in qr_result:
            if isinstance(elem, np.ndarray):
                return elem
        # last resort: ritorna il primo elemento (qualsiasi)
        return qr_result[0]
    return qr_result


def _mapping_function(elevation: np.ndarray, h_ipp: float) -> np.ndarray:
    """
    Mapping function to convert slant to vertical TEC.

    Parameters
    ----------
    elevation : np.ndarray
        _description_
    h_ipp : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    return 1.0 / np.cos(
        np.arcsin((6371 / (6371 + h_ipp / 1000)) * np.sin(np.radians(90 - elevation)))
    )


def _preprocessing(df: pl.DataFrame, rec_pos, h_ipp: float = 350_000) -> pl.DataFrame:
    year = df["epoch"][0].year

    modip_ipp = extract_modip(
        coords=(df["lon_ipp"].to_numpy(), df["lat_ipp"].to_numpy()),
        year=year,
        coord_type="geo",
    )
    modip_rec = extract_modip(coords=rec_pos, year=year, coord_type="ecef")[0]
    _, lon_rec, _ = ecef2geodetic(*rec_pos)

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


def _combine_interval_systems(interval_results: list, all_arcs: list, nmax: int = 3):
    """
    Combina tutte le matrici triangolari dagli intervalli.
    all_arcs DEVE essere una lista ordinata (non un set).
    """
    if not interval_results:
        raise ValueError("Nessun dato da combinare")

    print(f"🔍 DEBUG: {len(interval_results)} intervalli da processare")
    print(f"🔍 DEBUG: {len(all_arcs)} archi totali")

    n_arcs_total = len(all_arcs)
    n_poly_terms = nmax + 2

    # stima prudente del numero massimo di equazioni (si usa per allocazione)
    total_equations_est = 0
    for result in interval_results:
        Rm = _ensure_R(result["R_matrix"])
        if Rm is None or not isinstance(Rm, np.ndarray):
            continue
        total_equations_est += max(0, Rm.shape[0] - result["n_poly_terms"] - 1)

    if total_equations_est == 0:
        print("⚠️ Stima equazioni = 0, nessuna equazione valida trovata nei risultati")
        return np.zeros((0, n_arcs_total)), np.zeros((0,))

    global_bias_matrix = np.zeros((total_equations_est, n_arcs_total))
    global_obs_vector = np.zeros(total_equations_est)

    equation_counter = 0

    for interval_idx, result in enumerate(interval_results):
        R_matrix = result["R_matrix"]

        print(f"🔍 DEBUG Intervallo {interval_idx}:")
        print(f"   R_matrix type: {type(R_matrix)}")

        # Normalizza R_matrix
        R_matrix = _ensure_R(R_matrix)
        if R_matrix is None or not isinstance(R_matrix, np.ndarray):
            print(f"   ❌ Impossibile estrarre R, skippo intervallo {interval_idx}")
            continue
        print(f"   ✅ R_matrix normalizzata, shape: {getattr(R_matrix, 'shape', None)}")

        arc_list = result["arcs"]
        n_poly_terms_local = result["n_poly_terms"]
        print(f"   n_poly_terms_local: {n_poly_terms_local}")
        print(f"   arc_list: {len(arc_list)} archi")

        # Controlli shape
        if R_matrix.ndim != 2:
            print(f"   ⚠️ R_matrix non 2D, skippo")
            continue

        # Trova ultima riga utile basata sull'ultima colonna
        last_nonzero_row = np.where(R_matrix[:, -1] != 0)[0]
        print(f"   last_nonzero_row: {last_nonzero_row}")

        if len(last_nonzero_row) == 0:
            print(f"   ⚠️  Nessuna riga non-zero")
            continue

        UR = last_nonzero_row[-1] + 1
        print(f"   UR: {UR}")

        if UR <= n_poly_terms_local:
            print(f"   ⚠️  UR ({UR}) <= n_poly_terms ({n_poly_terms_local})")
            continue

        # Estrai sottomatrice bias+osservazioni
        RR = R_matrix[n_poly_terms_local:UR, n_poly_terms_local:]
        print(f"   RR shape: {RR.shape}")

        # Trova ultima riga non-zero nella sottomatrice
        last_nonzero_rr = np.where(RR[:, -1] != 0)[0]
        print(f"   last_nonzero_rr: {last_nonzero_rr}")

        if len(last_nonzero_rr) == 0:
            print(f"   ⚠️  RR nessuna riga non-zero")
            continue

        URR = last_nonzero_rr[-1] + 1
        print(f"   URR: {URR}")

        if URR <= 1:
            print(f"   ⚠️  URR ({URR}) <= 1")
            continue

        print(f"   ✅ Aggiungo {URR-1} equazioni")

        # Ricostruisci equazioni
        for row_idx in range(URR - 1):
            if equation_counter >= global_obs_vector.shape[0]:
                # dovrebbe essere impossibile se stima è corretta, ma sicuro
                print("   ⚠️ superato spazio allocato, interrompo l'aggiunta")
                break

            global_obs_vector[equation_counter] = RR[row_idx, URR - 1]

            for col_idx in range(row_idx, URR - 1):
                arc_id = arc_list[col_idx]
                # all_arcs ora si assume lista: cerco indice in lista
                if arc_id in all_arcs:
                    arc_position = all_arcs.index(arc_id)
                    global_bias_matrix[equation_counter, arc_position] = RR[
                        row_idx, col_idx
                    ]

            equation_counter += 1

    # Truncate alle righe effettivamente usate
    global_bias_matrix = global_bias_matrix[:equation_counter, :]
    global_obs_vector = global_obs_vector[:equation_counter]

    print(f"🔗 Combinazione completata: {equation_counter} equazioni")
    return global_bias_matrix, global_obs_vector


def _solve_final_system(global_bias_matrix: np.ndarray, global_obs_vector: np.ndarray):
    """
    Solve the combined system for the biases using QR decomposition.

    Parameters
    ----------
    global_bias_matrix : np.ndarray
        _description_
    global_obs_vector : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
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


def calibration_qr(
    df_preprocessed: pl.DataFrame, nmax: int = 3, n_epochs: int = 30
) -> tuple[dict[str, float], list[dict]]:
    """
    Perform GNSS-derived TEC calibration using QR decomposition to estimate
    instrumental offsets for each satellite-receiver arc.

    This function implements a divide-and-conquer approach where data is processed
    in temporal intervals, then combined to solve for arc-specific biases while
    modeling ionospheric variations with polynomial functions.

    Parameters
    ----------
    df_preprocessed : pl.DataFrame
        Preprocessed DataFrame containing GNSS observations
    nmax : int, optional
        Maximum degree for polynomial expansion of ionospheric model, by default 3
    n_epochs : int, optional
        Number of epochs for processing batches, by default 30

    Returns
    -------
    Tuple[Dict[str, float], List[Dict]]
        - arc_to_offset: Dictionary mapping arc_id to estimated bias
        - interval_results: List of intermediate results for each time interval
    """
    times = df_preprocessed["epoch"].unique().sort()
    interval_results: list[dict] = []
    all_arcs: list[str] = []
    _seen_arcs: set = set()

    for start_idx in range(0, len(times), n_epochs):
        end_idx = min(start_idx + n_epochs, len(times))
        selected_epochs = times[start_idx:end_idx]
        interval_data = df_preprocessed.filter(pl.col("epoch").is_in(selected_epochs))

        if interval_data.is_empty():
            continue

        arcs_summary = (
            interval_data.group_by("id_arc_valid")
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

        poly_matrix = polynomial_expansion(modip_ipp, modip_rec, lon_ipp, lon_rec, nmax)

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
        if R is None or not isinstance(R, np.ndarray):
            # print(f"❌ Intervallo {interval_idx}: impossibile ottenere R, skip")
            continue

        interval_results.append(
            {"R_matrix": R, "arcs": arc_ids, "n_poly_terms": poly_matrix.shape[1]}
        )

        for arc in arc_ids:
            if arc not in _seen_arcs:
                all_arcs.append(arc)
                _seen_arcs.add(arc)

        # print(f"✅ Intervallo {interval_idx}: {len(arc_ids)} archi processati")

    # print("🔗 Combinando tutti gli intervalli...")
    global_bias_matrix, global_obs_vector = _combine_interval_systems(
        interval_results, all_arcs, nmax
    )

    # print("🧮 Risolvendo sistema finale...")
    offsets = _solve_final_system(global_bias_matrix, global_obs_vector)
    arc_to_offset = dict(zip(all_arcs, offsets))

    # print(f"🎯 Calibrazione completata: {len(offsets)} offset stimati")
    return arc_to_offset, interval_results
