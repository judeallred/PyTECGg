# TEC arcs identification and correction 🔎

To ensure integrity in GNSS processing, it's essential to identify cycle slip (CS) and loss-of-lock (LoL) events, which indicate disruptions in the carrier-phase signal or receiver-satellite tracking.

The function `extract_arcs` handles this automatically by detecting CS and LoL events, discarding arcs shorter than `min_arc_length` epochs, correcting residual jumps (greater than `threshold_jumps`) in the linear combinations, and producing arc-levelled GFLC.

```python
from pytecgg.tec_calibration import extract_arcs

df_arcs = extract_arcs(
    df=df_lc,
    const_symb="E",
    threshold_abs=10,
    threshold_std=10,
    threshold_jump=5,
    receiver_acronym=rec_name,
)
```

The output is a Polars DataFrame with cycle slip and loss-of-lock flags, unique arc identifiers, CS-corrected linear combinations, and arc-levelled GFLC values.

In particular, CSs are flagged when abrupt changes in the MW combination exceed either a given number of standard deviations (`threshold_std`) or a fixed absolute threshold (`threshold_abs`). Additionally, if the time gap between consecutive epochs becomes too large, a LoL is declared; a `max_gap` argument can be explicitly set or automatically inferred from the data.