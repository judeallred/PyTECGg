import polars as pl


def add_arc_id(min_arc_length: int = 30) -> list[pl.Expr]:
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

    Returns:
    -------
    list[pl.Expr]
        List of Polars expressions representing:
        - id_arc: Complete arc identifier in format "sv_YYYYMMDD_arcnumber"
        - id_arc_valid: Validated arc identifier (None for arcs shorter than min_arc_length)
    """
    _id_arc = pl.col("is_loss_of_lock").cum_sum().over("sv")
    _arc_length = pl.col("gflc_code").is_not_null().sum().over(["sv", _id_arc])

    id_arc = pl.format(
        "{}_{}_{}",
        pl.col("sv").str.to_lowercase(),
        pl.col("epoch").dt.strftime("%Y%m%d"),
        _id_arc,
    )
    id_arc_valid = pl.when(_arc_length >= min_arc_length).then(id_arc).otherwise(None)

    return [id_arc.alias("id_arc"), id_arc_valid.alias("id_arc_valid")]
