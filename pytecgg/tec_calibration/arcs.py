from warnings import warn
from datetime import timedelta

import polars as pl

from pytecgg.linear_combinations import detect_cs_lol


def _add_arc_id(
    min_arc_length: int = 30, receiver_acronym: str = None
) -> list[pl.Expr]:
    """
    Identify continuous TEC arcs in GNSS observations

    Arcs are defined as sequences of observations without loss-of-lock events.
    Cycle slips do not break arcs, as they can be repaired in subsequent processing.
    Arcs shorter than the minimum length (in epochs) are discarded.

    Parameters:
    ----------
    min_arc_length : int
        Minimum number of consecutive valid observations required for an arc to be considered valid.
        Default: 30 epochs.
    receiver_acronym : str, optional
        Acronym of the receiver to prepend to the arc identifier.
        If provided, the arc ID format will be "<receiver>_<sv>_<YYYYMMDD>_<arcnumber>".
        Otherwise, the format will be "<sv>_<YYYYMMDD>_<arcnumber>".
        Default: None.

    Returns:
    -------
    list[pl.Expr]
        List of Polars expressions representing:
        - id_arc: Arc identifier
        - id_arc_valid: Valid arc identifier (None for arcs shorter than `min_arc_length`)
    """
    _id_arc = pl.col("is_loss_of_lock").cum_sum().over("sv")
    _arc_length = pl.col("gflc_code").is_not_null().sum().over(["sv", _id_arc])

    if receiver_acronym:
        id_arc = pl.format(
            "{}_{}_{}_{}",
            pl.lit(receiver_acronym),
            pl.col("sv").str.to_lowercase(),
            pl.col("epoch").dt.strftime("%Y%m%d"),
            _id_arc,
        )
    else:
        id_arc = pl.format(
            "{}_{}_{}",
            pl.col("sv").str.to_lowercase(),
            pl.col("epoch").dt.strftime("%Y%m%d"),
            _id_arc,
        )
    id_arc_valid = pl.when(_arc_length >= min_arc_length).then(id_arc).otherwise(None)

    return [id_arc.alias("id_arc"), id_arc_valid.alias("id_arc_valid")]


def _remove_cs_jumps(df: pl.DataFrame, threshold_jump: float = 10.0) -> pl.DataFrame:
    """
    Fix GNSS combinations by removing cycle-slip jumps within valid arcs

    For each column in `linear_combinations`, the function:
    1. Calculates differences between consecutive observations within valid arcs
    2. Identifies cycle slip contributions
    3. Computes cumulative sum of cycle slip jumps within each arc
    4. Fixes linear combinations by removing cumulative cycle slip effects
    5. Additional check: detects and corrects jumps above threshold between consecutive epochs

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - id_arc_valid: Valid arc identifiers
        - is_cycle_slip: Cycle slip indicators
        - linear_combinations, as calculated in `calculate_linear_combinations`
    threshold_jump : float, optional
        Threshold for detecting significant jumps between consecutive epochs.
        If the absolute difference between consecutive values exceeds this threshold,
        it will be treated as a jump and corrected. Default is 10

    Returns
    -------
    pl.DataFrame
        DataFrame with fixed columns (suffix '_fix') added
    """
    predefined_lin_combs = ["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    lin_combs = [lc_ for lc_ in predefined_lin_combs if lc_ in df.columns]

    if len(lin_combs) == 0:
        warn(
            "No linear combinations found in DataFrame columns; expected at least one of: "
            + ", ".join(predefined_lin_combs)
        )
        return df

    df_ = df.clone()

    for lc_ in lin_combs:
        # Calculate differences within valid arcs
        df_ = df_.with_columns(
            pl.when(pl.col("id_arc_valid").is_not_null())
            .then(pl.col(lc_) - pl.col(lc_).shift(1))
            .otherwise(None)
            .alias(f"_delta_{lc_}")
        )

        # Identify cycle slip contributions
        df_ = df_.with_columns(
            pl.when(pl.col("is_cycle_slip"))
            .then(pl.col(f"_delta_{lc_}"))
            .otherwise(0)
            .alias(f"_cs_delta_{lc_}")
        )

        # Additional check: detect significant jumps between consecutive epochs
        # This catches jumps that might not be flagged as cycle slips
        df_ = df_.with_columns(
            pl.when(
                (pl.col("id_arc_valid").is_not_null())
                & (pl.col(f"_delta_{lc_}").abs() > threshold_jump)
                & (
                    ~pl.col("is_cycle_slip")
                )  # Only if not already flagged as cycle slip
            )
            .then(pl.col(f"_delta_{lc_}"))
            .otherwise(0)
            .alias(f"_jump_delta_{lc_}")
        )

        # Combine cycle slip and jump contributions
        total_jump = pl.col(f"_cs_delta_{lc_}") + pl.col(f"_jump_delta_{lc_}")

        # Cumulative sum of cycle slips and jumps within each arc
        df_ = df_.with_columns(
            total_jump.cum_sum().over("id_arc_valid").alias(f"_total_cumsum_{lc_}")
        )

        df_ = df_.with_columns(
            (pl.col(lc_) - pl.col(f"_total_cumsum_{lc_}")).alias(f"{lc_}_fix")
        )

    return df_.drop(pl.col("^_.*$"))


def _level_phase_to_code(df: pl.DataFrame) -> pl.DataFrame:
    """
    Level phase measurements to code measurements within valid arcs

    For each valid arc, the function computes the mean difference between
    phase and code linear combinations and adjusts the phase measurements accordingly.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - id_arc_valid: Valid arc identifiers
        - gflc_phase_fix: Fixed phase linear combination
        - gflc_code_fix: Fixed code linear combination

    Returns
    -------
    pl.DataFrame
        DataFrame with leveled phase column added
    """
    if "gflc_phase_fix" not in df.columns or "gflc_code_fix" not in df.columns:
        warn(
            "Both 'gflc_phase_fix' and 'gflc_code_fix' must be present in DataFrame columns to level phase to code."
        )
        return df

    df_ = df.with_columns(
        (pl.col("gflc_phase_fix") - pl.col("gflc_code_fix")).alias("_phase_code_diff")
    )

    # Calculate the mean of (phase - code) over each valid arc
    df_ = df_.with_columns(
        pl.when(pl.col("id_arc_valid").is_not_null())
        .then(pl.col("_phase_code_diff").mean().over("id_arc_valid"))
        .otherwise(None)
        .alias("_mean_phase_code_diff")
    )

    # Calculate the final, levelled value
    df_ = df_.with_columns(
        pl.when(pl.col("id_arc_valid").is_not_null())
        .then(pl.col("gflc_phase_fix") - pl.col("_mean_phase_code_diff"))
        .otherwise(None)
        .alias("gflc_levelled")
    )

    return df_.drop(pl.col("^_.*$"))


def extract_arcs(
    df: pl.DataFrame,
    const_symb: str,
    threshold_abs: float = 5.0,
    threshold_std: float = 5.0,
    min_arc_length: int = 30,
    receiver_acronym: str = None,
    max_gap: timedelta = None,
    threshold_jump: float = 10.0,
) -> pl.DataFrame:
    """
    Extract continuous TEC arcs and fix GNSS linear combinations

    The function performs the following steps:
    1. Detects loss-of-lock events and cycle slips
    2. Identifies valid arcs, discarding short ones
    3. Removes cycle-slip jumps within valid arcs
    4. Additional check: corrects significant jumps between consecutive epochs
    5. Calculates arc-levelled GFLC values

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - epoch: Observation epochs
        - sv: Satellite identifiers
        - TEC-related linear combinations (e.g., gflc_code, gflc_phase, etc.)
    const_symb : str
        Constellation symbol (e.g., 'G' for GPS, 'E' for Galileo) used in cycle slip detection.
    threshold_abs : float, optional
        Absolute threshold for detecting cycle slips; default is 5
    threshold_std : float, optional
        Standard deviation multiplier threshold for detecting cycle slips; default is 5
    min_arc_length : int, optional
        Minimum number of consecutive valid observations required for an arc to be considered valid.
        Default: 30 epochs.
    receiver_acronym : str, optional
        Acronym of the receiver to prepend to arc identifiers.
        If provided, the arc ID format will be "<receiver>_<sv>_<YYYYMMDD>_<arcnumber>".
        Otherwise, the format will be "<sv>_<YYYYMMDD>_<arcnumber>".
        Default: None.
    max_gap : timedelta, optional
        Maximum allowed time gap between observations before declaring LoL (default: inferred
        from df's temporal resolution)
    threshold_jump : float, optional
        Threshold for detecting significant jumps between consecutive epochs.
        If the absolute difference between consecutive values exceeds this threshold,
        it will be treated as a jump and corrected. Default is 10

    Returns
    -------
    pl.DataFrame
        DataFrame with:
        - cycle slip and loss-of-lock flags
        - arc identifiers (id_arc, id_arc_valid)
        - fixed linear combinations (suffix '_fix')
        - gflc_levelled: arc-levelled GFLC values
    """
    df_ = detect_cs_lol(
        df,
        system=const_symb,
        threshold_abs=threshold_abs,
        threshold_std=threshold_std,
        max_gap=max_gap,
    )

    df_lc_arcs = df.join(
        df_,
        on=["epoch", "sv"],
    ).with_columns(
        _add_arc_id(min_arc_length=min_arc_length, receiver_acronym=receiver_acronym)
    )

    df_lc_arcs_fix = _remove_cs_jumps(df=df_lc_arcs, threshold_jump=threshold_jump)

    return _level_phase_to_code(df=df_lc_arcs_fix)
