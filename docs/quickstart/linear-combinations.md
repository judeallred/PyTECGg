# Combinations of GNSS measurements 📡

Starting from the basic observables, we can compute the following linear [combinations](https://gssc.esa.int/navipedia/index.php/Combination_of_GNSS_Measurements), useful for removing biases or isolating physical effects:

- [Geometry-Free](https://gssc.esa.int/navipedia/index.php/Detector_based_in_carrier_phase_data:_The_geometry-free_combination) Linear Combination (GFLC), sensitive to ionospheric effects.
- [Ionosphere-Free](https://gssc.esa.int/navipedia/index.php/Ionosphere-free_Combination_for_Dual_Frequency_Receivers) Linear Combination (IFLC), used to eliminate the ionospheric delay.
- [Melbourne-Wübbena](https://gssc.esa.int/navipedia/index.php/Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination) (MW) combination, useful for cycle-slip detection and ambiguity resolution.

The function `calculate_linear_combinations` supports both phase and code versions of GFLC and IFLC. You can choose which `combinations` to compute:

```python
from pytecgg.satellites import prepare_ephemeris
from pytecgg.linear_combinations import calculate_linear_combinations

# Prepare the ephemerides, e.g. for Galileo
ephem_dict = prepare_ephemeris(nav_dict, constellation="Galileo")

df_lc = calculate_linear_combinations(
    df_obs,
    system="E",
    combinations=["gflc_phase", "gflc_code", "mw"],
    rinex_version=rinex_version,    
)
```

Available options for `combinations` are:

- `"gflc_phase"` – GFLC using carrier phase

- `"gflc_code"` – GFLC using code pseudorange

- `"mw"` – MW combination

- `"iflc_phase"` – IFLC using carrier phase

- `"iflc_code"` – IFLC using code pseudorange

If not specified, the default is `["gflc_phase", "gflc_code", "mw"]`.

`ephem_dict` is a dictionary containing ephemeris parameters, keyed by satellite ID.
The resulting `df_lc` is a Polars DataFrame with one row per satellite and epoch, containing the requested combinations.