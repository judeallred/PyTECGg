# TEC calibration ⚖️

The TEC calibration is performed by estimating and removing per-arc biases from geometry-free combinations. It computes both slant TEC (sTEC) and vertical TEC (vTEC) using a polynomial expansion of the ionospheric shell up to a configurable degree (`max_degree`, default is 3).

The calibration is performed in batches (`n_epochs`, default is 30) to ensure temporal stability while maintaining responsiveness to ionospheric variations.

```python
from pytecgg.tec_calibration import calculate_tec

calculate_tec(df_, receiver_position=rec_pos)
```