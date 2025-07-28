from typing import Literal, Optional

import polars as pl

from . import C, OBS_MAPPING, FREQ_BANDS


def _calculate_melbourne_wubbena(
    phase1: pl.Expr,
    phase2: pl.Expr,
    code1: pl.Expr,
    code2: pl.Expr,
    freq1: pl.Expr,
    freq2: pl.Expr,
) -> pl.Expr:
    """
    Calculate the Melbourne-Wübbena (MW) combination for cycle-slip detection

    Parameters:
        phase1 (pl.Expr): Phase observation (in cycles) for frequency 1
        phase2 (pl.Expr): Phase observation (in cycles) for frequency 2
        code1 (pl.Expr): Code observation (in meters) for frequency 1
        code2 (pl.Expr): Code observation (in meters) for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz

    Returns:
        pl.Expr: MW combination (in meters)
    """
    lambda1 = C / freq1
    lambda2 = C / freq2
    # Phase wide-lane (in meters)
    lw = (freq1 * phase1 * lambda1 - freq2 * phase2 * lambda2) / (freq1 - freq2)
    # Narrow-lane code combination (in meters)
    pn = (freq1 * code1 + freq2 * code2) / (freq1 + freq2)
    return lw - pn


def calculate_melbourne_wubbena(
    obs_data: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Calculate the Melbourne-Wübbena (MW) combination for cycle-slip detection

    Parameters:
        obs_data (pl.DataFrame): Input GNSS observation data
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        glonass_freq (Optional[dict[str, int]]): Dictionary mapping GLONASS
            satellite IDs to their frequency channel numbers; required only
            if system == "R"

    Returns:
        pl.DataFrame: DataFrame containing the computed MW combination (in meters)
        for each satellite and epoch
    """
    phase_mapping = OBS_MAPPING[system]["phase"]
    code_mapping = OBS_MAPPING[system]["code"]

    phase_keys = list(phase_mapping.keys())
    code_keys = list(code_mapping.keys())

    phase1, phase2 = phase_mapping[phase_keys[0]], phase_mapping[phase_keys[1]]
    code1, code2 = code_mapping[code_keys[0]], code_mapping[code_keys[1]]

    df_sys = obs_data.filter(
        (pl.col("sv").str.starts_with(system))
        & (pl.col("observable").is_in([phase1, phase2, code1, code2]))
    )

    if df_sys.is_empty():
        return pl.DataFrame()

    df_pivot = df_sys.pivot(
        values="value",
        index=["epoch", "sv"],
        columns="observable",
        aggregate_function="first",
    )

    required_cols = {phase1, phase2, code1, code2}
    if not required_cols.issubset(df_pivot.columns):
        missing = required_cols - set(df_pivot.columns)
        print(f"Warning: Missing observations: {missing}")
        return pl.DataFrame()

    if system == "R":
        if glonass_freq is None:
            raise ValueError("glonass_freq is required for GLONASS")
        df_pivot = df_pivot.with_columns(
            pl.col("sv").map_dict(glonass_freq).alias("freq_number")
        )
        freq1 = (1602 + pl.col("freq_number") * 0.5625) * 1e6
        freq2 = (1246 + pl.col("freq_number") * 0.4375) * 1e6
    else:
        phase_to_band = {v: k for k, v in phase_mapping.items()}
        band1 = phase_to_band[phase1]
        band2 = phase_to_band[phase2]
        freq1 = FREQ_BANDS[system][band1]
        freq2 = FREQ_BANDS[system][band2]

    return df_pivot.with_columns(
        _calculate_melbourne_wubbena(
            pl.col(phase1), pl.col(phase2), pl.col(code1), pl.col(code2), freq1, freq2
        ).alias("melbourne_wubbena")
    ).drop([phase1, phase2, code1, code2])
