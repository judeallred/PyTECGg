from datetime import datetime, timedelta

import polars as pl

from pytecgg.linear_combinations.mw import _calculate_melbourne_wubbena
from pytecgg.linear_combinations.gflc import _calculate_gflc_phase
from pytecgg.linear_combinations.cs_lol_detection import detect_cs_lol


def test_mw_cycle_slip():
    """Test MW sensitivity to cycle slips"""
    freq1 = 1575.42e6
    freq2 = 1227.60e6

    # No cycle slip
    df_no_slip = pl.DataFrame(
        {
            "phase1": [1000.0, 1000.1, 1000.2],
            "phase2": [800.0, 800.08, 800.16],
            "code1": [20000000.0, 20000000.1, 20000000.2],
            "code2": [20000000.0, 20000000.1, 20000000.2],
        }
    )

    # 1 wide-lane cycle slip on L1
    df_slip = pl.DataFrame(
        {
            "phase1": [1000.0, 1000.1, 1001.1],  # slip of 1 cycle on L1
            "phase2": [800.0, 800.08, 800.08],  # no slip on L2
            "code1": [20000000.0, 20000000.1, 20000000.2],
            "code2": [20000000.0, 20000000.1, 20000000.2],
        }
    )

    mw_no_slip = df_no_slip.with_columns(
        mw=_calculate_melbourne_wubbena(
            pl.col("phase1"),
            pl.col("phase2"),
            pl.col("code1"),
            pl.col("code2"),
            freq1,
            freq2,
        )
    )["mw"]

    mw_slip = df_slip.with_columns(
        mw=_calculate_melbourne_wubbena(
            pl.col("phase1"),
            pl.col("phase2"),
            pl.col("code1"),
            pl.col("code2"),
            freq1,
            freq2,
        )
    )["mw"]

    # Stability against small changes (no slip)
    assert abs(mw_no_slip[1] - mw_no_slip[0]) < 0.1
    assert abs(mw_no_slip[2] - mw_no_slip[1]) < 0.1

    # Jump due to cycle slip
    assert abs(mw_slip[2] - mw_slip[1]) > 0.5


def test_gflc_phase_iono():
    """Test GFLC phase sensitivity to ionospheric changes"""
    freq1 = 1575.42e6  # GPS L1
    freq2 = 1227.60e6  # GPS L2

    # Data with ionospheric variation, namely a larger change on L2
    df = pl.DataFrame(
        {
            "phase1": [1000.0, 1000.5],
            "phase2": [
                800.0,
                801.0,
            ],
        }
    )

    result = df.with_columns(
        gflc=_calculate_gflc_phase(pl.col("phase1"), pl.col("phase2"), freq1, freq2)
    )["gflc"]

    # GFLC should show significant variation due to different ionospheric sensitivity
    assert abs(result[1] - result[0]) > 1  # TECu change


def test_detect_cs():
    """Test cycle slip detection"""
    # Data with a clear cycle slip at the 5th epoch
    df_mock = pl.DataFrame(
        {
            "epoch": [
                datetime(2023, 1, 1, 0, 0, 0),
                datetime(2023, 1, 1, 0, 0, 30),
                datetime(2023, 1, 1, 0, 1, 0),
                datetime(2023, 1, 1, 0, 1, 30),
                datetime(2023, 1, 1, 0, 2, 0),
                datetime(2023, 1, 1, 0, 2, 30),
            ],
            "sv": ["G01", "G01", "G01", "G01", "G01", "G01"],
            "mw": [10.0, 10.1, 10.2, 10.1, 15.5, 15.4],
        }
    )

    result = detect_cs_lol(df_mock, "G", threshold_std=5, threshold_abs=5)
    cs_detections = result.filter(pl.col("is_cycle_slip") == True)
    assert len(cs_detections) == 1
    assert cs_detections["epoch"][0] == datetime(2023, 1, 1, 0, 2, 0)


def test_detect_lol():
    """Test loss-of-lock detection"""
    df = pl.DataFrame(
        {
            "epoch": [
                datetime(2023, 1, 1, 0, 0, 0),
                datetime(2023, 1, 1, 0, 0, 30),
                datetime(2023, 1, 1, 0, 6, 0),
            ],
            "sv": ["G01", "G01", "G01"],
            "mw": [10.0, 10.1, 10.2],
        }
    )

    result = detect_cs_lol(df, "G", max_gap=timedelta(minutes=1))

    lol_detections = result.filter(pl.col("is_loss_of_lock") == True)
    assert len(lol_detections) == 1
    assert lol_detections["epoch"][0] == datetime(2023, 1, 1, 0, 6, 0)
