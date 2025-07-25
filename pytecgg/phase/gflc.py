import polars as pl
from typing import Optional, Literal

from . import OBS_MAPPING, FREQ_BANDS, C


def _calculate_gflc_phase(
    phase1: pl.Expr, phase2: pl.Expr, freq1: pl.Expr, freq2: pl.Expr
) -> pl.Expr:
    """
    Calculate the geometry-free linear combination (GFLC) from two phase observations

    Parameters:
        phase1 (pl.Expr): Phase observation for frequency 1
        phase2 (pl.Expr): Phase observation for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz
    Returns:
        pl.Expr: Expression for the calculated GFLC
    """
    lambda1 = C / freq1
    lambda2 = C / freq2
    pr_to_tec = (1 / 40.308) * (freq1**2 * freq2**2) / (freq1**2 - freq2**2) / 1e16
    return (phase1 * lambda1 - phase2 * lambda2) * pr_to_tec


def _calculate_gflc_code(
    code1: pl.Expr, code2: pl.Expr, freq1: pl.Expr, freq2: pl.Expr
) -> pl.Expr:
    """
    Calculate the geometry-free linear combination (GFLC) from two code observations

    Parameters:
        code1 (pl.Expr): Code observation for frequency 1
        code2 (pl.Expr): Code observation for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz
    Returns:
        pl.Expr: Expression for the calculated GFLC
    """
    pr_to_tec = (1 / 40.308) * (freq1**2 * freq2**2) / (freq1**2 - freq2**2) / 1e16
    return (code2 - code1) * pr_to_tec


def calculate_gflc(
    obs_data: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Process observations for a specific GNSS system to calculate GFLC (phase and code)

    Parameters:
        obs_data (pl.DataFrame): DataFrame containing observation data
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        glonass_freq (Optional[dict[str, int]]): Frequency mapping for GLONASS, required if system is "R"

    Returns:
        pl.DataFrame: DataFrame with calculated GFLC values (phase and code)
    """
    # Get phase and code mappings
    phase_mapping = OBS_MAPPING[system]["phase"]
    code_mapping = OBS_MAPPING[system]["code"]

    # Get observation keys
    phase_keys = list(phase_mapping.keys())  # e.g. ["L1", "L2"]
    code_keys = list(code_mapping.keys())  # e.g. ["C1", "C2"]

    phase1, phase2 = phase_mapping[phase_keys[0]], phase_mapping[phase_keys[1]]
    code1, code2 = code_mapping[code_keys[0]], code_mapping[code_keys[1]]

    # Filter system data with both phase and code observations
    df_sys = obs_data.filter(
        (pl.col("sv").str.starts_with(system))
        & (pl.col("observable").is_in([phase1, phase2, code1, code2]))
    )

    if df_sys.is_empty():
        return pl.DataFrame()

    # Pivot to get phase and code in separate columns
    df_pivot = df_sys.pivot(
        values="value",
        index=["epoch", "sv"],
        columns="observable",
        aggregate_function="first",
    )

    # Check if we have all required observations
    required_cols = {phase1, phase2, code1, code2}
    if not required_cols.issubset(df_pivot.columns):
        missing = required_cols - set(df_pivot.columns)
        print(f"Warning: Missing observations: {missing}")
        return pl.DataFrame()

    # Frequency handling
    if system == "R":
        # FIXME
        if glonass_freq is None:
            raise ValueError("glonass_freq is required for GLONASS processing")
        df_pivot = df_pivot.with_columns(
            pl.col("sv").map_dict(glonass_freq).alias("freq_number")
        )
        freq1 = (1602 + pl.col("freq_number") * 0.5625) * 1e6
        freq2 = (1246 + pl.col("freq_number") * 0.4375) * 1e6
    elif system in ["G", "E", "C"]:
        phase_to_band = {v: k for k, v in phase_mapping.items()}
        band1 = phase_to_band.get(phase1)
        band2 = phase_to_band.get(phase2)

        try:
            freq1 = FREQ_BANDS[system][band1]
            freq2 = FREQ_BANDS[system][band2]
        except KeyError as e:
            raise KeyError(
                f"Missing frequency for band '{e.args[0]}' in system '{system}'"
            )

    # Calculate both GFLC combinations
    return df_pivot.with_columns(
        _calculate_gflc_phase(pl.col(phase1), pl.col(phase2), freq1, freq2).alias(
            "gflc_phase"
        ),
        _calculate_gflc_code(pl.col(code1), pl.col(code2), freq1, freq2).alias(
            "gflc_code"
        ),
    ).drop([phase1, phase2, code1, code2])
