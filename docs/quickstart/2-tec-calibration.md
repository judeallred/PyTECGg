# TEC calibration ⚖️

The calibration process is a fundamental step in GNSS-based ionospheric studies. Its goal is to estimate and remove the hardware-related inter-frequency biases (often referred to as Differential Code Biases, or DCBs) from both the transmitting satellites and the receiving hardware.

`PyTECGg` implements a modern version of the calibration algorithm described by [**Ciraolo et al. (2007)**](https://link.springer.com/article/10.1007/s00190-006-0093-1). This approach addresses the systematic errors introduced during the "levelling" process (where carrier-phase observations are aligned with code-delay data) and accounts for the short-term instability of receiver biases.

## How it works 💬

The calibration computes both slant TEC (**sTEC**) and vertical TEC (**vTEC**) by modeling the ionosphere as a thin shell at a fixed altitude. The spatial variation of the ionospheric delay is represented using a polynomial expansion of the shell up to a configurable degree (`max_degree`, default is 3).

To ensure temporal stability and account for intra-day variations of the hardware biases, the calibration is performed in batches of size `batch_size_epochs`. This allows the model to remain responsive to ionospheric variations while mitigating the impact of code-delay multipath.

To calibrate your TEC data, you need the prepared `DataFrame` (containing arc-levelled combinations and IPP coordinates) and the `GNSSContext`:

```python
from pytecgg.tec_calibration import calculate_tec

df_calibrated = calculate_tec(
    df_final, 
    ctx=ctx,
    max_polynomial_degree=3,
    batch_size_epochs=30
)
```

Calibration Parameters:

- `max_polynomial_degree`: maximum degree of the polynomial expansion used to model the ionosphere; default is 3.
- `batch_size_epochs`: number of epochs per batch; smaller batches track bias variations more closely, while larger ones offer more stability; default is 30.

## Wrapping up 🏁

This function estimates instrumental offsets and appends calibrated physical values to the `DataFrame`. You can verify the results by inspecting the estimated biases and the resulting absolute TEC values.

=== "Code"
    ```python
    df_calibrated.filter(
        pl.col("id_arc_valid").is_not_null()
    ).select(
        [
            "epoch",
            "sv",
            "id_arc_valid",
            "gflc_levelled",
            "azi",
            "ele",
            "bias",
            "stec",
            "vtec",
        ]
    ).head()
    ```
=== "Example Output"

    | epoch | sv | id_arc_valid | gflc_levelled | azi | ele | bias | stec | vtec |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | 00:00:00 | E02 | "bneu_e02_001" | 42.39 | 257.73 | 26.55 | 16.47 | 25.93 | 13.74 |
    | 00:00:00 | E03 | "bneu_e03_001" | 42.11 | 314.36 | 47.16 | 22.99 | 19.12 | 14.62 |
    | 00:00:00 | E05 | "bneu_e05_001" | 44.97 | 179.50 | 73.10 | 25.73 | 19.24 | 18.49 |
    | 00:00:00 | E16 | "bneu_e16_001" | 11.03 | 7.57 | 52.26 | -6.88 | 17.91 | 14.59 |
    | 00:00:00 | E25 | "bneu_e25_001" | 34.64 | 321.15 | 59.68 | 19.23 | 15.41 | 13.53 |

The new columns represent:

- `bias`: estimated instrumental bias (in TECu) for the specific arc and batch.
- `stec`: sTEC, corrected for hardware biases.
- `vtec`: final vTEC, obtained by applying the mapping function to the sTEC.

!!! tip "Absolute TEC"
    Unlike the relative geometry-free combination, vTEC provides the absolute electron content, making it directly comparable across different stations and satellites.