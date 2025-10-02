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
    return 1.0 / np.cos(
        np.arcsin((6371 / (6371 + h_ipp / 1000)) * np.sin(np.radians(90 - elevation)))
    )


def _preprocessing(df_: pl.DataFrame, rec_pos, h_ipp: float = 350_000) -> pl.DataFrame:
    year = df_["epoch"][0].year

    modip_ipp = extract_modip(
        coords=(df_["lon_ipp"].to_numpy(), df_["lat_ipp"].to_numpy()),
        year=year,
        coord_type="geo",
    )
    modip_rec = extract_modip(coords=rec_pos, year=year, coord_type="ecef")[0]
    _, lon_rec, _ = ecef2geodetic(*rec_pos)

    mapping = _mapping_function(df_["ele"].to_numpy(), h_ipp)
    gflc_vert = df_["gflc_levelled"].to_numpy() * mapping

    return df_.with_columns(
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
    if global_bias_matrix.size == 0:
        raise ValueError("Sistema vuoto - nessuna equazione valida")

    # Combina matrici (equivalente MATLAB: BB_CC = [BB, CC])
    final_system = np.column_stack([global_bias_matrix, global_obs_vector])

    # QR finale (equivalente MATLAB: [~, r_BC] = qr(BB_CC))
    R_final = qr(final_system, mode="r")
    R_final = _ensure_R(R_final)

    # Trova sistema risolvibile (equivalente MATLAB: UR_BC = find(r_BC(:, end), 1, 'last'))
    last_nonzero = np.where(R_final[:, -1] != 0)[0]
    if len(last_nonzero) == 0:
        raise ValueError("Sistema finale degenere")
    UR_BC = last_nonzero[-1] + 1

    if UR_BC <= 1:
        raise ValueError("Sistema finale sotto-determinato")

    # Estrai e risolvi (equivalente MATLAB: offset = RRBC \ bb)
    RRBC = R_final[: UR_BC - 1, : UR_BC - 1]
    bb = R_final[: UR_BC - 1, -1]

    offsets = np.linalg.lstsq(RRBC, bb, rcond=None)[0]

    print(f"✅ Sistema risolto: {len(offsets)} offset stimati")
    return offsets


def calibration_qr(df_preprocessed: pl.DataFrame, nmax: int = 3, interval: int = 30):
    times = df_preprocessed["epoch"].unique().sort()
    interval_results = []
    # Manteniamo una lista ordinata di archi + set per check di esistenza rapido
    all_arcs = []
    _seen_arcs = set()

    # 2. PROCESSAMENTO INTERVALLI - Batch processing
    for interval_idx, start_idx in enumerate(range(0, len(times), interval)):
        end_idx = min(start_idx + interval, len(times))
        interval_epochs = times[start_idx:end_idx]
        interval_data = df_preprocessed.filter(pl.col("epoch").is_in(interval_epochs))

        if interval_data.is_empty():
            continue

        # 3. RAGGRUPPAMENTO PER ARCO
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

        # 4. CALCOLO POLINOMIALE - Batch
        arc_data = arcs_summary.to_dicts()
        arc_ids = [arc["id_arc_valid"] for arc in arc_data]

        # Prepara arrays per calcolo vettoriale
        modip_ipp = np.array([arc["modip_ipp"] for arc in arc_data])
        modip_rec = np.array([arc["modip_rec"] for arc in arc_data])
        lon_ipp = np.array([arc["lon_ipp"] for arc in arc_data])
        lon_rec = np.array([arc["lon_rec"] for arc in arc_data])
        mapping_funcs = np.array([arc["mapping"] for arc in arc_data])
        observations = np.array([arc["gflc_vert"] for arc in arc_data])

        # Calcolo polinomiale batch
        poly_matrix = polynomial_expansion(modip_ipp, modip_rec, lon_ipp, lon_rec, nmax)

        # 5. COSTRUZIONE SISTEMA
        n_arcs = len(arc_ids)
        n_obs = len(arc_data)

        # Matrice bias: one-hot encoding degli archi
        bias_matrix = np.zeros((n_obs, n_arcs))
        arc_to_idx = {arc_id: i for i, arc_id in enumerate(arc_ids)}

        for i, arc_id in enumerate(arc_ids):
            bias_matrix[i, arc_to_idx[arc_id]] = mapping_funcs[i]

        # Sistema combinato
        system_matrix = np.column_stack(
            [poly_matrix, bias_matrix, observations.reshape(-1, 1)]
        )

        # 6. FATTORIZZAZIONE QR
        R = qr(system_matrix, mode="r")
        R = _ensure_R(R)
        if R is None or not isinstance(R, np.ndarray):
            print(f"❌ Intervallo {interval_idx}: impossibile ottenere R, skip")
            continue

        # 7. ACCUMULA RISULTATI
        interval_results.append(
            {"R_matrix": R, "arcs": arc_ids, "n_poly_terms": poly_matrix.shape[1]}
        )

        # aggiorna all_arcs mantenendo ordine e unicità
        for arc in arc_ids:
            if arc not in _seen_arcs:
                all_arcs.append(arc)
                _seen_arcs.add(arc)

        print(f"✅ Intervallo {interval_idx}: {len(arc_ids)} archi processati")

    # 8. COMBINAZIONE FINALE
    print("🔗 Combinando tutti gli intervalli...")
    global_bias_matrix, global_obs_vector = _combine_interval_systems(
        interval_results, all_arcs, nmax
    )

    print("🧮 Risolvendo sistema finale...")
    offsets = _solve_final_system(global_bias_matrix, global_obs_vector)

    # Crea dizionario arc_id -> offset
    arc_to_offset = {arc_id: offset for arc_id, offset in zip(all_arcs, offsets)}

    print(f"🎯 Calibrazione completata: {len(offsets)} offset stimati")
    return arc_to_offset, interval_results
