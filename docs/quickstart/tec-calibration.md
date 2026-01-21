# TEC calibration ⚖️

The calibration process is a fundamental step in GNSS-based ionospheric studies. Its goal is to estimate and remove the hardware-related inter-frequency biases (often referred to as Differential Code Biases, or DCBs) from the transmitting satellites and the receiving hardware.

PyTECGg implements a modern version of the calibration algorithm described by [Ciraolo et al. (2007)](https://link.springer.com/article/10.1007/s00190-006-0093-1). This approach addresses the systematic errors introduced during the "levelling" process (where carrier-phase observations are aligned with code-delay data) and accounts for the short-term instability of receiver biases.

## How it works

The calibration computes both slant TEC (sTEC) and vertical TEC (vTEC) by modeling the ionosphere as a thin shell at a fixed altitude. The spatial variation of the ionospheric delay is represented using a polynomial expansion of the shell up to a configurable degree (`max_degree`, default is 3).

To ensure temporal stability and account for intra-day variations of the hardware biases, the calibration is performed in batches (`n_epochs`, default is 30). This allows the model to remain responsive to ionospheric variations while mitigating the impact of code-delay multipath.

To calibrate TEC, you simply need the prepared DataFrame (containing arc-levelled combinations and IPP coordinates) and the receiver's ECEF position:

```python
from pytecgg.tec_calibration import calculate_tec

df_calibrated = calculate_tec(
    df_, 
    receiver_position=rec_pos,
    max_degree=3,
    n_epochs=30
)
```

The function estimates biases and adds `bias`, `stec` and `vtec` columns to the DataFrame.