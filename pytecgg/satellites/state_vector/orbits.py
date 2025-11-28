import datetime

import numpy as np


def _glonass_derivatives(state, gm, c20, a, ae):
    """Compute derivatives for GLONASS satellite motion"""
    r = state[0:3]
    v = state[3:6]
    r_norm = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
    z_r = r[2] / r_norm
    z_r2 = z_r * z_r

    # Earth gravity + zonal harmonic
    earth_grav = -gm * r / r_norm**3
    zonal_term = 1.5 * c20 * gm * a**2 / r_norm**5
    zonal_correction = zonal_term * np.array(
        [
            r[0] * (1 - 5 * z_r2),
            r[1] * (1 - 5 * z_r2),
            r[2] * (3 - 5 * z_r2),
        ]
    )

    # Total acceleration (Earth gravity + zonal harmonic + lunisolar)
    acceleration = earth_grav + zonal_correction + ae

    out = np.empty(6)
    out[0:3], out[3:6] = v, acceleration
    return out


def _get_gmst(ymd: list) -> float:
    """Compute Greenwich Mean Sidereal Time (simplified version)

    Parameters:
        ymd: [year, month, day] list

    Returns:
        GMST in radians
    """
    dt = datetime.datetime(ymd[0], ymd[1], ymd[2])
    jd = dt.toordinal() + 1721425.5  # Convert to Julian Date
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t**2
        - t**3 / 38710000
    )
    return np.radians(gmst % 360)
