import polars as pl

from .constants import C


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
