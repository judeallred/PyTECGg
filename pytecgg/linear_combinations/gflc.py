import polars as pl

from .constants import C


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
