# PyTECGg

[![PyPI version](https://img.shields.io/pypi/v/pytecgg.svg)](https://pypi.org/project/pytecgg/)
![Python version](https://img.shields.io/badge/python-3.11--3.13-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Tests](https://github.com/viventriglia/PyTECGg/actions/workflows/pytest.yml/badge.svg)

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core

## Table of Contents

- [What is it?](#what-is-it)

- [Installation](#installation)

- [Example usage](#example-usage)


## What is it?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the [Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content) (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

This package:
- is open source: read and access all the code!
- supports all modern GNSS constellations, codes and signals:
    - GPS, Galileo, BeiDou, ~~GLONASS~~ and QZSS
- supports RINEX V2-3-4
- provides seamless decompression for RINEX files

| ![Earth's ionosphere and GNSS satellites](images/project_cover.webp) |
|:--:| 
| *Generated image of Earth's ionosphere with GNSS satellites studying TEC* |


👉 [**Contributing to PyTECGg**](./CONTRIBUTING.md)


## Installation

### 📦 From PyPI (recommended)

You can install the package directly from PyPI:

```shell
pip install pytecgg
```

This will also install all required Python dependencies automatically.

### 🛠️ From source distribution

If you prefer to install from the source distribution (e.g. for development or inspection), pip will compile the Rust core locally.

```shell
pip install pytecgg --no-binary :all:
```

> ℹ️ Note: building from source requires a working Rust toolchain (rustc, cargo). You can install it via [rustup](https://rustup.rs/).


## Example usage

### Parse RINEX files — fast ⚡

```python
from pytecgg.parsing import read_rinex_nav, read_rinex_obs

# Load a RINEX navigation file into a dictionary of DataFrames (one per constellation)
nav_dict = read_rinex_nav("./path/to/your/nav_file.rnx")

# Load a RINEX observation file and extract:
# - a DataFrame of observations,
# - the receiver's approximate position in ECEF,
# - the RINEX version string.
df_obs, rec_pos, version = read_rinex_obs("./path/to/your/obs_file.rnx")
```

Timestamps in the epoch column are parsed as strings by default.
To enable time-based filtering and computation, convert them to timezone-aware datetimes using Polars:

```python
import polars as pl

df_obs = df_obs.with_columns(
    pl.col("epoch")
    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S GPST", strict=False)
    .dt.replace_time_zone("UTC")
    .alias("epoch")
)
```

### Combinations of GNSS measurements 📡

Starting from the basic observables, we can compute the following linear [combinations](https://gssc.esa.int/navipedia/index.php/Combination_of_GNSS_Measurements), useful for removing biases or isolating physical effects:
- [Geometry-Free](https://gssc.esa.int/navipedia/index.php/Detector_based_in_carrier_phase_data:_The_geometry-free_combination) Linear Combination (GFLC), sensitive to ionospheric effects.
- [Ionosphere-Free](https://gssc.esa.int/navipedia/index.php/Ionosphere-free_Combination_for_Dual_Frequency_Receivers) Linear Combination (IFLC), used to eliminate the ionospheric delay.
- [Melbourne-Wübbena](https://gssc.esa.int/navipedia/index.php/Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination) (MW) combination, useful for cycle-slip detection and ambiguity resolution.

The function `calculate_linear_combinations` supports both phase and code versions of GFLC and IFLC. You can choose which `combinations` to compute:

```python
from pytecgg.satellites.ephemeris import prepare_ephemeris
from pytecgg.linear_combinations.lc_calculation import calculate_linear_combinations

# Prepare the ephemerides, e.g. for Galileo
ephem_dict = prepare_ephemeris(nav_dict, constellation='Galileo')

df_lc = calculate_linear_combinations(
    df_obs,
    system='E',
    combinations=['gflc_phase', 'mw'],
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

### Cycle slip (CS) and Loss-of-Lock (LoL) detection 🚨

To ensure integrity in GNSS processing, it's essential to identify CS and LoL events, which indicate disruptions in the carrier-phase signal or receiver-satellite tracking.

The function `detect_cs_lol` uses the MW combination to detect anomalies in the observation stream:

```python
from pytecgg.linear_combinations.cs_lol_detection import detect_cs_lol

df_cs_lol = detect_cs_lol(
    df_lc,
    system='E',
    threshold_abs=10,
    threshold_std=5,
)
```

CSs are flagged when abrupt changes in the MW combination exceed either a given number of standard deviations (`threshold_std`) or a fixed absolute threshold (`threshold_abs`). Additionally, if the time gap between consecutive epochs becomes too large, a LoL is declared; the `max_gap` argument can be explicitly set or automatically inferred from the data.

The output is a Polars DataFrame with one row per epoch-satellite, containing boolean flags: `is_cycle_slip` signals the presence of a cycle slip, while `is_loss_of_lock` indicates a discontinuity due to signal loss or satellite setting. When LoL occurs, `is_cycle_slip` is set to `None` to avoid ambiguity.

### Satellite coordinates and Ionospheric Pierce Point (IPP) 🛰️

To get the satellite's position in space, we can compute ECEF coordinates for each satellite–epoch and add them as columns to an existing Polars DataFrame:

```python
from pytecgg.satellites.positions import satellite_coordinates

df_lc_pos = df_lc.with_columns(
    *satellite_coordinates(
        sv_ids=df_lc["sv"],
        epochs=df_lc["epoch"],
        ephem_dict=ephem_dict,
        gnss_system="Galileo",
    )
)
```

We can then compute the IPP — the intersection between the satellite–receiver line of sight and a thin-shell ionosphere at a fixed altitude:

```python
from pytecgg.satellites.ipp import calculate_ipp

# Extract satellite positions as a NumPy array
sat_ecef_array = df_lc_pos.select(["sat_x", "sat_y", "sat_z"]).to_numpy()

# Compute IPP latitude and longitude, azimuth and elevation angle from
# receiver to satellite, assuming a fixed ionospheric shell height of 350 km
lat_ipp, lon_ipp, azi, ele = calculate_ipp(
    rec_pos,
    sat_ecef_array,
    h_ipp=350_000,
)

df_lc_ipp = df_lc_pos.with_columns([
    pl.Series("lat_ipp", lat_ipp),
    pl.Series("lon_ipp", lon_ipp),
    pl.Series("azi", azi),
    pl.Series("ele", ele)
])
```
