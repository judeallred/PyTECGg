from collections.abc import Iterable

import polars as pl


def add_arc_id(min_arc_length: int = 30, receiver_acronym: str = None) -> list[pl.Expr]:
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


def remove_cs_jumps(
    df: pl.DataFrame, linear_combinations: Iterable[str]
) -> pl.DataFrame:
    """
    Compute leveled GNSS fields by removing cycle-slip jumps within valid arcs

    For each column in `linear_combinations`, the function:
    1. Calculates differences between consecutive observations within valid arcs
    2. Identifies cycle slip contributions
    3. Computes cumulative sum of cycle slip jumps within each arc
    4. Generates levelled columns by removing cumulative cycle slip effects

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - id_arc_valid: Valid arc identifiers
        - is_cycle_slip: Cycle slip indicators
        - Columns specified in linear_combinations
    linear_combinations : Iterable[str]
        List of column names to level (e.g., ["gflc_phase", "gflc_code"])

    Returns
    -------
    pl.DataFrame
        DataFrame with levelled columns (suffix '_levelled') added
    """
    if not isinstance(linear_combinations, Iterable):
        raise TypeError("linear_combinations must be an iterable of column names")

    df_ = df.clone()

    for lc_ in linear_combinations:
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

        # Cumulative sum of cycle slips within each arc
        df_ = df_.with_columns(
            pl.col(f"_cs_delta_{lc_}")
            .cum_sum()
            .over("id_arc_valid")
            .alias(f"_cs_cumsum_{lc_}")
        )

        # Create levelled column
        df_ = df_.with_columns(
            (pl.col(lc_) - pl.col(f"_cs_cumsum_{lc_}")).alias(f"{lc_}_levelled")
        )

    return df_.drop(pl.col("^_.*$"))
