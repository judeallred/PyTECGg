import polars as pl
from typing import Optional, Literal, Any

from .constants import FREQ_BANDS
from .observables import retrieve_observable_pairs, _extract_band
from .gflc import _calculate_gflc_code, _calculate_gflc_phase
from .iflc import _calculate_iflc_code, _calculate_iflc_phase
from .mw import _calculate_melbourne_wubbena
from pytecgg.context import GNSSContext


def calculate_linear_combinations(
    obs_data: pl.DataFrame,
    ctx: GNSSContext,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ] = ["gflc_phase", "gflc_code", "mw"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Process observations for multiple GNSS systems to calculate specific linear combinations

    Parameters
    ----------
    obs_data : pl.DataFrame
        DataFrame containing observation data.
    ctx : GNSSContext
        Execution context containing GNSS systems, RINEX version, and support lookups.
    combinations : list[Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]]
        List of combinations to calculate. Options:
            - "gflc_phase": Geometry-Free Linear Combination (Phase)
            - "gflc_code": Geometry-Free Linear (Code)
            - "mw": Melbourne-Wübbena combination
            - "iflc_phase": Ionosphere-Free Linear Combination (Phase)
            - "iflc_code": Ionosphere-Free Linear Combination (Code)
        Defaults to ["gflc_phase", "gflc_code", "mw"]

    Returns
    -------
    pl.DataFrame
        DataFrame with the requested linear combinations.
    """
    results = []

    for system_ in ctx.systems:
        best_pairs = retrieve_observable_pairs(
            obs_data,
            system=system_,
            rinex_version=ctx.rinex_version,
            prefer_by_suffix=True,
        )

        if best_pairs is None:
            continue

        (phase1, phase2), (code1, code2) = best_pairs

        df_sys = obs_data.filter(
            (pl.col("sv").str.starts_with(system_))
            & (pl.col("observable").is_in([phase1, phase2, code1, code2]))
        )

        if df_sys.is_empty():
            continue

        df_pivot = df_sys.pivot(
            values="value",
            index=["epoch", "sv"],
            on="observable",
            aggregate_function="first",
        )

        required_cols = {phase1, phase2, code1, code2}
        if not required_cols.issubset(df_pivot.columns):
            continue

        if system_ == "R":
            # GLONASS frequencies are SV-dependent based on the channel lookup
            if not ctx.glonass_channels:
                continue

            # Map GLONASS channels to frequencies per SV
            f1_map = FREQ_BANDS["R"][_extract_band(phase1)]
            f2_map = FREQ_BANDS["R"][_extract_band(phase2)]

            # Metadata in MHz
            ctx.freq_meta[system_] = {
                sv: (f1_map(k) / 1e6, f2_map(k) / 1e6)
                for sv, k in ctx.glonass_channels.items()
                if k is not None
            }

            df_step = df_pivot.with_columns(
                pl.col("sv").replace(ctx.glonass_channels).cast(pl.Float32).alias("_k")
            )
            freq1, freq2 = f1_map(pl.col("_k")), f2_map(pl.col("_k"))
        else:
            try:
                f1 = FREQ_BANDS[system_][_extract_band(phase1)]
                f2 = FREQ_BANDS[system_][_extract_band(phase2)]

                # MHz for metadata
                ctx.freq_meta[system_] = (f1 / 1e6, f2 / 1e6)
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
        return pl.DataFrame()

    return pl.concat(results)
