import datetime

import numpy as np

from ..constants import GNSS_CONSTANTS

const = GNSS_CONSTANTS["GLONASS"]


def _glonass_derivatives(t, state, const, ae):
    """Compute derivatives for GLONASS satellite motion"""
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)

    # Earth gravity + zonal harmonic
    earth_grav = -const.gm * r / r_norm**3
    zonal_term = 1.5 * const.c20 * const.gm * const.a**2 / r_norm**5
    zonal_correction = zonal_term * np.array(
        [
            r[0] * (1 - 5 * (r[2] / r_norm) ** 2),
            r[1] * (1 - 5 * (r[2] / r_norm) ** 2),
            r[2] * (3 - 5 * (r[2] / r_norm) ** 2),
        ]
    )

    # Total acceleration (Earth gravity + zonal harmonic + lunisolar)
    acceleration = earth_grav + zonal_correction + ae

    return np.concatenate([v, acceleration])


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
