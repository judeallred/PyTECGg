import re
from typing import Optional, Tuple
import polars as pl

from .constants import (
    PHASE_FREQ_PRIORITY,
    CODE_FREQ_PRIORITY,
    PHASE_CHAN_PRIORITY,
    CODE_CHAN_PRIORITY,
)

_band_re = re.compile(r"^([A-Za-z]\d+)")


def _extract_band(obs: str) -> Optional[str]:
    m = _band_re.match(obs)
    return m.group(1) if m else None


def _pick_best(
    candidates: list[str], suffix_priority: list[str], df: pl.DataFrame | None = None
):
    if not candidates:
        return None
    if df is None:

        def score(c):
            band = _extract_band(c)
            suf = c[len(band) :]
            return (
                suffix_priority.index(suf[0])
                if suf and suf[0] in suffix_priority
                else 99
            )

        return min(candidates, key=score)
    counts = {c: df.filter(pl.col("observable") == c).height for c in candidates}
    return max(counts, key=counts.get)


def retrieve_observable_pairs(
    obs_data: pl.DataFrame,
    system: str,
    rinex_version: str,
    prefer_by_suffix: bool = True,
    df_for_counts: pl.DataFrame | None = None,
) -> Optional[Tuple[str, str]]:
    """
    Automatically select the best phase and code observable pairs for a given GNSS system.

    Parameters
    ----------
    obs_data : pl.DataFrame
        DataFrame with columns ['epoch','sv','observable','value'].
    system : str
        GNSS system identifier ("G", "R", "E", "C").
    rinex_version : str
        RINEX version ("2" or "3").
    prefer_by_suffix : bool
        If True, selects based on suffix priority; otherwise uses occurrence count.
    df_for_counts : pl.DataFrame | None
        Required if prefer_by_suffix is False to compute coverage-based selection.

    Returns
    -------
    tuple[(phase1, phase2), (code1, code2)] | None
        Best (phase1, phase2) and (code1, code2) pairs for the system, or None if not available.
    """
    if "system" not in obs_data.columns:
        obs_data = obs_data.with_columns([pl.col("sv").str.slice(0, 1).alias("system")])

    df_sys = obs_data.filter(pl.col("system") == system)
    if df_sys.is_empty():
        return None

    phase_avail = (
        df_sys.filter(pl.col("observable").str.starts_with("L"))
        .select("observable")
        .unique()
        .to_series()
        .to_list()
    )
    code_avail = (
        df_sys.filter(pl.col("observable").str.slice(0, 1).is_in(["C", "P"]))
        .select("observable")
        .unique()
        .to_series()
        .to_list()
    )

    if rinex_version.startswith("2"):
        phase1 = "L1" if "L1" in phase_avail else phase_avail[0]
        phase2 = (
            "L2"
            if "L2" in phase_avail
            else phase_avail[1] if len(phase_avail) > 1 else phase_avail[0]
        )
        code1 = "C1" if "C1" in code_avail else code_avail[0]
        code2 = (
            "C2"
            if "C2" in code_avail
            else (
                "P2"
                if "P2" in code_avail
                else code_avail[1] if len(code_avail) > 1 else code_avail[0]
            )
        )
        return (phase1, phase2), (code1, code2)

    def choose_best_pair(
        available: list[str], priority_list: list[tuple[str, str]], is_phase: bool
    ):
        suffix_priority = PHASE_CHAN_PRIORITY if is_phase else CODE_CHAN_PRIORITY
        for spec1, spec2 in priority_list:

            def pick(spec):
                if spec in available:
                    return spec
                band = _extract_band(spec)
                if not band:
                    return None
                candidates = [o for o in available if o.startswith(band)]
                if not candidates:
                    return None
                return _pick_best(
                    candidates,
                    suffix_priority,
                    None if prefer_by_suffix else df_for_counts,
                )

            o1, o2 = pick(spec1), pick(spec2)
            if o1 and o2:
                return o1, o2
        return None

    phase_pair = choose_best_pair(
        phase_avail, PHASE_FREQ_PRIORITY.get(system, []), True
    )
    code_pair = choose_best_pair(code_avail, CODE_FREQ_PRIORITY.get(system, []), False)

    if phase_pair and code_pair:
        return phase_pair, code_pair

    return None
