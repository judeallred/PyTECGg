import numpy as np


def polynomial_expansion(
    modip: np.ndarray,
    stationmodip: np.ndarray,
    lon: np.ndarray,
    station_lon: np.ndarray,
    nmax: int,
) -> np.ndarray:
    delta_lon = lon - station_lon
    delta_modip = modip - stationmodip
    const_norm = 1.0 / (1.0 + np.abs(delta_modip) ** (nmax + 1))

    n_points = len(modip)
    n_terms = 2 + nmax
    pterms_matrix = np.zeros((n_points, n_terms))

    # Constant and longitude terms (2 terms)
    pterms_matrix[:, 0] = const_norm
    pterms_matrix[:, 1] = delta_lon * const_norm
    # MoDip terms (nmax terms)
    for j in range(1, nmax + 1):
        pterms_matrix[:, 1 + j] = (delta_modip**j) * const_norm

    return pterms_matrix
