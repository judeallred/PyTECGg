import polars as pl
from typing import Optional, Literal

from .constants import OBS_MAPPING_BY_VERSION, FREQ_BANDS
from .gflc import _calculate_gflc_code, _calculate_gflc_phase
from .iflc import _calculate_iflc_code, _calculate_iflc_phase
from .mw import _calculate_melbourne_wubbena


def _map_obs_by_version(rinex_version: str) -> dict:
    """
    Map RINEX version to observation mapping

    Parameters:
        rinex_version (str): RINEX version string (e.g., '2.11', '3.04')

    Returns:
        dict: Observation mapping for the specified version
    """
    v_ = rinex_version.split(".")
    major_version = v_[0]

    if major_version not in OBS_MAPPING_BY_VERSION:
        raise ValueError(
            f"Unsupported RINEX version: {rinex_version}. "
            f"Supported major versions: {list(OBS_MAPPING_BY_VERSION.keys())}"
        )
    return OBS_MAPPING_BY_VERSION[major_version]


def calculate_linear_combinations(
    obs_data: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    rinex_version: str,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ] = ["gflc_phase", "gflc_code", "mw"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Process observations for a specific GNSS system to calculate specific linear combinations

    Parameters:
        obs_data (pl.DataFrame): DataFrame containing observation data
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        rinex_version (str): RINEX version string (e.g., '2.11', '3.04')
        combinations (list[Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]]):
            List of combinations to calculate. Options:
                - "gflc_phase": Geometry-Free Linear Combination (Phase)
                - "gflc_code": Geometry-Free Linear (Code)
                - "mw": Melbourne-Wübbena combination
                - "iflc_phase": Ionosphere-Free Linear Combination (Phase)
                - "iflc_code": Ionosphere-Free Linear Combination (Code)
            Defaults to ["gflc_phase", "gflc_code", "mw"]
        glonass_freq (Optional[dict[str, int]]): Frequency mapping for GLONASS, required if system is "R"

    Returns:
        pl.DataFrame: DataFrame with the requested linear combinations
    """
    # Get the appropriate mapping for the RINEX version
    obs_mapping = _map_obs_by_version(rinex_version)

    # Get phase and code mappings
    phase_mapping = obs_mapping[system]["phase"]
    code_mapping = obs_mapping[system]["code"]

    # Get observation keys
    phase_keys = list(phase_mapping.keys())  # e.g. ["L1", "L2"]
    code_keys = list(code_mapping.keys())  # e.g. ["C1", "C2"]

    phase1, phase2 = phase_mapping[phase_keys[0]], phase_mapping[phase_keys[1]]
    code1, code2 = code_mapping[code_keys[0]], code_mapping[code_keys[1]]

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

    df_result = df_pivot

    if "gflc_phase" in combinations:
        df_result = df_result.with_columns(
            _calculate_gflc_phase(pl.col(phase1), pl.col(phase2), freq1, freq2).alias(
                "gflc_phase"
            )
        )

    if "gflc_code" in combinations:
        df_result = df_result.with_columns(
            _calculate_gflc_code(pl.col(code1), pl.col(code2), freq1, freq2).alias(
                "gflc_code"
            )
        )

    if "mw" in combinations:
        df_result = df_result.with_columns(
            _calculate_melbourne_wubbena(
                pl.col(phase1),
                pl.col(phase2),
                pl.col(code1),
                pl.col(code2),
                freq1,
                freq2,
            ).alias("mw")
        )

    if "iflc_phase" in combinations:
        df_result = df_result.with_columns(
            _calculate_iflc_phase(pl.col(phase1), pl.col(phase2), freq1, freq2).alias(
                "iflc_phase"
            )
        )

    if "iflc_code" in combinations:
        df_result = df_result.with_columns(
            _calculate_iflc_code(pl.col(code1), pl.col(code2), freq1, freq2).alias(
                "iflc_code"
            )
        )

    return df_result.drop([phase1, phase2, code1, code2])
