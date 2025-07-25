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

- [How can I help?](#how-can-i-help)

## What is it?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the [Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content) (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

This package:
- is open source: read and access all the code!
- supports all modern GNSS constellations, codes and signals:
    - GPS, Galileo, BeiDou, ~~GLONASS~~ and QZSS
- supports RINEX V2-3-4
- provides seamless decompression for RINEX files

| ![Earth's ionosphere and GNSS satellites](images/project_cover.png) | 
|:--:| 
| *Generated image of Earth's ionosphere with GNSS satellites studying TEC* |


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

### Compute Geometry-Free Linear Combinations (GFLC) 📡

To compute GFLC for a given GNSS constellation, first filter and format the ephemerides, then pass the observation data to the GFLC computation function.

```python
from pytecgg.satellites.ephemeris import prepare_ephemeris
from pytecgg.linear_combinations.gflc import calculate_gflc

# Prepare the ephemerides, e.g. for Galileo
ephem_dict = prepare_ephemeris(nav_dict, constellation='Galileo')

# Compute GFLC
df_gflc = calculate_gflc(
    df_obs,
    system='E',
)
```

`ephem_dict` is a dictionary containing ephemeris parameters, keyed by satellite ID.
The resulting `df_gflc` is a Polars DataFrame with one row per satellite and epoch, containing both the carrier-phase and code GFLCs.

### Satellite coordinates and Ionospheric Pierce Point (IPP) 🛰️

To associate GFLCs with the satellite's position in space, we can compute ECEF coordinates for each satellite–epoch and add them as columns to the existing Polars DataFrame:

```python
from pytecgg.satellites.positions import satellite_coordinates

df_gflc_pos = df_gflc.with_columns(
    *satellite_coordinates(
        sv_ids=df_gflc["sv"],
        epochs=df_gflc["epoch"],
        ephem_dict=ephem_dict,
        gnss_system="Galileo",
    )
)
```

We can then compute the IPP — the intersection between the satellite–receiver line of sight and a thin-shell ionosphere at a fixed altitude:

```python
from pytecgg.satellites.ipp import calculate_ipp

# Extract satellite positions as a NumPy array
sat_ecef_array = df_gflc_pos.select(["sat_x", "sat_y", "sat_z"]).to_numpy()

# Compute IPP latitude and longitude, azimuth and elevation angle from
# receiver to satellite, assuming a fixed ionospheric shell height of 350 km
lat_ipp, lon_ipp, azi, ele = calculate_ipp(
    rec_pos,
    sat_ecef_array,
    h_ipp=350_000,
)

df_gflc_ipp = df_gflc_pos.with_columns([
    pl.Series("lat_ipp", lat_ipp),
    pl.Series("lon_ipp", lon_ipp),
    pl.Series("azi", azi),
    pl.Series("ele", ele)
])
```


## How can I help?

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature_amazing_feature`)
3. Commit your Changes (`git commit -m 'Add some amazing stuff'`)
4. Push to the Branch (`git push origin feature_amazing_feature`)
5. Open a Pull Request

