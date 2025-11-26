import datetime
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import GNSS_CONSTANTS
from pytecgg.satellites.kepler.orbits import _is_ephemeris_valid
from pytecgg.satellites.state_vector.orbits import _glonass_derivatives, _get_gmst


def _state_vector_satellite_coordinates(
    ephem_dict: dict[str, dict[str, Any]],
    sv_id: str,
    obs_time: datetime.datetime | None = None,
    t_res: float = 60.0,
    rtol: float = 1e-8,
    atol: float = 1e-11,
) -> np.ndarray:
    """
    Compute GLONASS satellite position from ephemeris data only,
    propagating motion for a given number of seconds from ephemeris time.

    Parameters:
        ephem_dict: Dictionary containing ephemeris data
        sv_id: Satellite identifier (e.g., 'R01')
        obs_time: Observation time (datetime); if None, uses ephemeris timestamp
        t_res: Time resolution for ODE solver output
        rtol: Relative tolerance for solver
        atol: Absolute tolerance for solver

    Returns:
        pos: [3] array of ECEF coordinates [X, Y, Z] (meters)
    """
    REQUIRED_KEYS = {
        "satPosX": "Satellite Position X (km)",
        "satPosY": "Satellite Position Y (km)",
        "satPosZ": "Satellite Position Z (km)",
        "velX": "Velocity X (km/s)",
        "velY": "Velocity Y (km/s)",
        "velZ": "Velocity Z (km/s)",
        "datetime": "Ephemeris datetime",
    }
    const = GNSS_CONSTANTS["GLONASS"]

    # Validation
    if sv_id not in ephem_dict:
        raise KeyError(f"Satellite {sv_id} not found in ephemeris data")

    data = ephem_dict[sv_id]
    if not _is_ephemeris_valid(data, sv_id, REQUIRED_KEYS):
        return np.array([], dtype=float), {}

    # Converting km/s → m/s
    re = np.array([data["satPosX"], data["satPosY"], data["satPosZ"]]) * 1000
    ve = np.array([data["velX"], data["velY"], data["velZ"]]) * 1000

    # Converting km/s² → m/s²
    ae = (
        np.array(
            [
                0.0 if data["accelX"] is None else data["accelX"],
                0.0 if data["accelY"] is None else data["accelY"],
                0.0 if data["accelZ"] is None else data["accelZ"],
            ]
        )
        * 1000
    )

    eph_time = data["datetime"]
    if obs_time is not None and obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=datetime.timezone.utc)

    delta_seconds = (obs_time - eph_time).total_seconds() if obs_time else 0.0
    te = eph_time.hour * 3600 + eph_time.minute * 60 + eph_time.second
    ymd = [eph_time.year, eph_time.month, eph_time.day]

    # GMST at eph_time
    theta_ge = _get_gmst(ymd) + const.we * (te % 86400)
    rot_matrix = np.array(
        [
            [np.cos(theta_ge), -np.sin(theta_ge), 0],
            [np.sin(theta_ge), np.cos(theta_ge), 0],
            [0, 0, 1],
        ]
    )

    # Inertial coordinates at epoch
    ra = rot_matrix @ re
    va = rot_matrix @ ve + const.we * np.array([-ra[1], ra[0], 0])
    initial_state = np.concatenate([ra, va])

    # Integration time span
    t_span = (0, delta_seconds) if delta_seconds >= 0 else (delta_seconds, 0)
    n_steps = max(2, int(abs(t_span[1] - t_span[0]) / t_res) + 1)

    sol = solve_ivp(
        fun=lambda t, y: _glonass_derivatives(t, y, const, ae),
        t_span=t_span,
        y0=initial_state,
        t_eval=np.linspace(t_span[0], t_span[1], n_steps),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Rotate back to ECEF after delta_seconds
    theta_gi = theta_ge + const.we * delta_seconds
    rot_matrix_obs = np.array(
        [
            [np.cos(theta_gi), np.sin(theta_gi), 0],
            [-np.sin(theta_gi), np.cos(theta_gi), 0],
            [0, 0, 1],
        ]
    )

    inertial_pos = sol.y[:3, -1]
    ecef_pos = rot_matrix_obs @ inertial_pos

    return ecef_pos
