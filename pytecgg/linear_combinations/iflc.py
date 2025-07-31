import polars as pl


def _calculate_iflc_phase(
    phase1: pl.Expr, phase2: pl.Expr, freq1: pl.Expr, freq2: pl.Expr
) -> pl.Expr:
    """
    Calculate the ionosphere-free linear combination (IFLC) from two phase observations

    Parameters:
        phase1 (pl.Expr): Phase observation for frequency 1
        phase2 (pl.Expr): Phase observation for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz
    Returns:
        pl.Expr: Expression for the calculated GFLC
    """
    return (freq1**2 * phase1 - freq2**2 * phase2) / (freq1**2 - freq2**2)


def _calculate_iflc_code(
    code1: pl.Expr, code2: pl.Expr, freq1: pl.Expr, freq2: pl.Expr
) -> pl.Expr:
    """
    Calculate the ionosphere-free linear combination (IFLC) from two code observations

    Parameters:
        code1 (pl.Expr): Code observation for frequency 1
        code2 (pl.Expr): Code observation for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz
    Returns:
        pl.Expr: Expression for the calculated GFLC
    """
    return (freq1**2 * code1 - freq2**2 * code2) / (freq1**2 - freq2**2)
