from typing import Literal
from datetime import timedelta

import polars as pl
import numpy as np

from pytecgg.linear_combinations import FREQ_BANDS, C


def detect_cs_lol(
    df: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    threshold_std: float = 5.0,
    threshold_abs: float = 5.0,
    max_gap: timedelta = timedelta(seconds=30),
) -> pl.DataFrame:
    lambda_w = C / (FREQ_BANDS[system]["L1"] - FREQ_BANDS[system]["L2"])
    sigma_0 = lambda_w / 2
    result = []

    for sv in df.get_column("sv").unique():
        df_sv = df.filter(pl.col("sv") == sv).sort("epoch")

        k = 0
        m_mw = None
        sigma2_mw = None
        last_valid_epoch = None
        m_prev = None

        for row in df_sv.iter_rows(named=True):
            epoch = row["epoch"]
            mw = row["mw"]

            # Melbourne-Wübbena value missing: loss of lock
            if mw is None:
                result.append(
                    {
                        "epoch": epoch,
                        "sv": sv,
                        "is_lol": True,
                        "is_cycle_slip": None,
                    }
                )
                k = 0
                m_mw = None
                sigma2_mw = None
                m_prev = None
                continue

            # Gap in the current epoch: loss of lock/sunset
            is_lol = False
            if last_valid_epoch is not None:
                gap = epoch - last_valid_epoch
                if gap > max_gap:
                    is_lol = True
                    result.append(
                        {
                            "epoch": epoch,
                            "sv": sv,
                            "is_lol": True,
                            "is_cycle_slip": None,
                        }
                    )
                    k = 0
                    m_mw = None
                    sigma2_mw = None
                    m_prev = None

            if is_lol:
                last_valid_epoch = epoch
                continue

            # First valid point after a gap or NaN
            if k == 0 or m_mw is None or sigma2_mw is None or m_prev is None:
                m_mw = mw
                sigma2_mw = sigma_0**2
                is_cycle_slip = False
            else:
                sigma = np.sqrt(sigma2_mw)
                deviation = np.abs(mw - m_mw)
                delta = np.abs(mw - m_prev)

                if (deviation > threshold_std * sigma) and (delta > threshold_abs):
                    is_cycle_slip = True
                    k = 0
                    m_mw = None
                    sigma2_mw = None
                    m_prev = None
                else:
                    is_cycle_slip = False
                    m_prev = m_mw
                    m_mw = (k * m_mw + mw) / (k + 1)
                    sigma2_mw = (k * sigma2_mw + (mw - m_prev) ** 2) / (k + 1)

            result.append(
                {
                    "epoch": epoch,
                    "sv": sv,
                    "is_lol": is_lol,
                    "is_cycle_slip": is_cycle_slip if not is_lol else None,
                }
            )

            last_valid_epoch = epoch
            if not is_cycle_slip and not is_lol:
                k += 1
                m_prev = mw

    return pl.DataFrame(result)
