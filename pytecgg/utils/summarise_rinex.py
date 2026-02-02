import polars as pl


def summarise_rinex_data(
    obs: pl.DataFrame,
    nav: dict[str, pl.DataFrame],
) -> None:
    """
    Prints a summary of RINEX GNSS observation and navigation data.
    """
    print("=" * 55)
    print("🛰️  GNSS RINEX DATA SUMMARY")
    print("=" * 55)

    epochs = obs["epoch"].unique().sort()
    start_t = epochs.min()
    end_t = epochs.max()
    duration = end_t - start_t
    sampling = epochs.diff().median()

    print(f"\n📅 Observations:")
    print(f"   - Start:    {start_t}")
    print(f"   - End:      {end_t}")
    print(f"   - Duration: {duration}")
    print(f"   - Sampling: {sampling.total_seconds() if sampling else 'N/A'}s")

    print(f"\n📡 Constellations Breakdown:")
    const_stats = (
        obs.with_columns(pl.col("sv").str.slice(0, 1).alias("const"))
        .group_by("const")
        .agg(
            [
                pl.col("sv").n_unique().alias("unique_svs"),
                pl.col("value").count().alias("total_obs"),
                pl.col("observable").n_unique().alias("unique_signals"),
            ]
        )
        .sort("const")
    )

    header = f"{'Sys':<5} | {'SVs':<5} | {'# Observables':<15} | {'Total Records':<15}"
    print(header)
    print("-" * len(header))

    for row in const_stats.to_dicts():
        print(
            f"{row['const']:<5} | {row['unique_svs']:<5} | {row['unique_signals']:<15} | {row['total_obs']:,}"
        )

    print(f"\n📖 Navigation Coverage (Ephemerides):")
    for const, df_nav in nav.items():
        nav_svs = df_nav["sv"].n_unique()
        print(f"   - {const:<8}: {nav_svs:>3} satellites with valid ephemeris")
