from typing import Literal
from datetime import timedelta

import polars as pl
import numpy as np

from .constants import FREQ_BANDS, C


def _infer_temporal_resolution(df: pl.DataFrame) -> timedelta:
    epochs = df.get_column("epoch")
    return epochs.unique().sort().diff().drop_nulls().mode()[0]


def detect_cs_lol(
    df: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    threshold_std: float = 5.0,
    threshold_abs: float = 5.0,
    max_gap: timedelta = None,
) -> pl.DataFrame:
    """
    Detect cycle slip (CS) and loss-of-lock (LoL) in GNSS observations using
    Melbourne-Wübbena combination

    Parameters:
        df (pl.DataFrame): DataFrame containing GNSS observation data with Melbourne-Wübbena values
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        threshold_std (float): Threshold in standard deviations for CS detection (default: 5.0)
        threshold_abs (float): Absolute threshold in meters for CS detection (default: 5.0)
        max_gap (timedelta): Maximum allowed time gap between observations before declaring
            LoL (default: inferred from df's temporal resolution)

    Returns:
        pl.DataFrame: DataFrame with CS and LoL detections, containing:
            - epoch: Observation timestamp
            - sv: Satellite PRN number
            - is_loss_of_lock: Boolean indicating LoL (or satellite setting)
            - is_cycle_slip: Boolean indicating CS (None, when LoL occurs)

    Notes:
        This function implements a CS detection algorithm based on the Melbourne-Wübbena
        combination, following the methodology described in:
        ESA Navipedia (https://gssc.esa.int/navipedia/index.php?title=Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination)
    """
    lambda_w = C / (FREQ_BANDS[system]["L1"] - FREQ_BANDS[system]["L2"])
    sigma_0 = lambda_w / 2
    if max_gap is None:
        max_gap = _infer_temporal_resolution(df)

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
                        "is_loss_of_lock": True,
                        "is_cycle_slip": None,
                    }
                )
                k = 0
                m_mw = None
                sigma2_mw = None
                m_prev = None
                continue

            # Gap in the current epoch: loss of lock/satellite setting
            is_loss_of_lock = False
            if last_valid_epoch is not None:
                gap = epoch - last_valid_epoch
                if gap > max_gap:
                    is_loss_of_lock = True
                    result.append(
                        {
                            "epoch": epoch,
                            "sv": sv,
                            "is_loss_of_lock": True,
                            "is_cycle_slip": None,
                        }
                    )
                    k = 0
                    m_mw = None
                    sigma2_mw = None
                    m_prev = None

            if is_loss_of_lock:
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
                    "is_loss_of_lock": is_loss_of_lock,
                    "is_cycle_slip": is_cycle_slip if not is_loss_of_lock else None,
                }
            )

            last_valid_epoch = epoch
            if not is_cycle_slip and not is_loss_of_lock:
                k += 1
                m_prev = mw

    return pl.DataFrame(result)
