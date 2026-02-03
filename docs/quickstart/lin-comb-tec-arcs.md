# Linear Combinations & TEC Arcs 📡

## Combinations of GNSS measurements

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

## TEC arcs identification and correction

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