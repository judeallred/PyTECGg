import polars as pl
from typing import Optional, Literal, Any

from .constants import FREQ_BANDS
from .observables import retrieve_observable_pairs, _extract_band
from .gflc import _calculate_gflc_code, _calculate_gflc_phase
from .iflc import _calculate_iflc_code, _calculate_iflc_phase
from .mw import _calculate_melbourne_wubbena


def calculate_linear_combinations(
    obs_data: pl.DataFrame,
    rinex_version: str,
    systems: Optional[list[Literal["G", "E", "C", "R"]]] = None,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ] = ["gflc_phase", "gflc_code", "mw"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Process observations for multiple GNSS systems to calculate specific linear combinations

    Parameters:
        obs_data (pl.DataFrame): DataFrame containing observation data
        rinex_version (str): RINEX version string (e.g., '2.11', '3.04')
        systems (Optional[list[Literal["G", "E", "C", "R"]]]):
            List of GNSS systems to process. If None, all systems in obs_data are processed.
        combinations (list[Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]]):
            List of combinations to calculate. Options:
                - "gflc_phase": Geometry-Free Linear Combination (Phase)
                - "gflc_code": Geometry-Free Linear (Code)
                - "mw": Melbourne-Wübbena combination
                - "iflc_phase": Ionosphere-Free Linear Combination (Phase)
                - "iflc_code": Ionosphere-Free Linear Combination (Code)
            Defaults to ["gflc_phase", "gflc_code", "mw"]
        glonass_freq (Optional[dict[str, int]]): Frequency mapping for GLONASS, required if "R" is processed

    Returns:
        tuple[pl.DataFrame, dict[str, Any]]:
            - DataFrame with the requested linear combinations
            - freq_meta: Dictionary containing frequencies (MHz) for each system.
                Format: {system: (f1, f2)} or {system: {sv: (f1, f2)}} for GLONASS.
    """
    if systems is None:
        systems = obs_data["sv"].str.slice(0, 1).unique().to_list()  # type: ignore

    results = []
    freq_meta = {}

    for system in systems:
        best_pairs = retrieve_observable_pairs(
            obs_data, system=system, rinex_version=rinex_version, prefer_by_suffix=True
        )

        if best_pairs is None:
            continue

        (phase1, phase2), (code1, code2) = best_pairs

        df_sys = obs_data.filter(
            (pl.col("sv").str.starts_with(system))
            & (pl.col("observable").is_in([phase1, phase2, code1, code2]))
        )

        if df_sys.is_empty():
            continue

        df_pivot = df_sys.pivot(
            values="value",
            index=["epoch", "sv"],
            columns="observable",
            aggregate_function="first",
        )

        required_cols = {phase1, phase2, code1, code2}
        if not required_cols.issubset(df_pivot.columns):
            continue

        # Handle system-specific frequency mapping
        if system == "R":
            if glonass_freq is None:
                continue

            # Map GLONASS channels to frequencies per SV
            f1_map = FREQ_BANDS["R"][_extract_band(phase1)]
            f2_map = FREQ_BANDS["R"][_extract_band(phase2)]

            # MHz for metadata
            sv_freqs = {
                sv: (f1_map(k) / 1e6, f2_map(k) / 1e6)
                for sv, k in glonass_freq.items()
                if k is not None
            }
            freq_meta[system] = sv_freqs

            df_step = df_pivot.with_columns(
                pl.col("sv").replace(glonass_freq).cast(pl.Float32).alias("_k")
            )
            freq1, freq2 = f1_map(pl.col("_k")), f2_map(pl.col("_k"))
        else:
            try:
                f1 = FREQ_BANDS[system][_extract_band(phase1)]
                f2 = FREQ_BANDS[system][_extract_band(phase2)]

                # MHz for metadata
                freq_meta[system] = (f1 / 1e6, f2 / 1e6)
                freq1, freq2 = pl.lit(f1), pl.lit(f2)
                df_step = df_pivot
            except KeyError:
                continue

        # Linear combinations calculation
        if "gflc_phase" in combinations:
            df_step = df_step.with_columns(
                _calculate_gflc_phase(
                    pl.col(phase1), pl.col(phase2), freq1, freq2
                ).alias("gflc_phase")
            )
        if "gflc_code" in combinations:
            df_step = df_step.with_columns(
                _calculate_gflc_code(pl.col(code1), pl.col(code2), freq1, freq2).alias(
                    "gflc_code"
                )
            )
        if "mw" in combinations:
            df_step = df_step.with_columns(
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
            df_step = df_step.with_columns(
                _calculate_iflc_phase(
                    pl.col(phase1), pl.col(phase2), freq1, freq2
                ).alias("iflc_phase")
            )
        if "iflc_code" in combinations:
            df_step = df_step.with_columns(
                _calculate_iflc_code(pl.col(code1), pl.col(code2), freq1, freq2).alias(
                    "iflc_code"
                )
            )

        drop_cols = list(required_cols)
        if "_k" in df_step.columns:
            drop_cols.append("_k")

        results.append(df_step.drop(drop_cols))

    if not results:
        return pl.DataFrame(), {}

    return pl.concat(results), freq_meta
