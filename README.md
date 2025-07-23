# PyTECGg

<!-- Add PyPi version when published -->
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

If you prefer to install from the source distribution (e.g. for development or inspection), pip will compile the Rust core locally. In that case, a working Rust toolchain is required:

```shell
# Option 1 – let pip build from source
pip install pytecgg --no-binary :all:

# Option 2 – download the .tar.gz from PyPI and install manually
pip install pytecgg-*.tar.gz
```

> ℹ️ Note: Building from source requires a working Rust toolchain (rustc, cargo). You can install it via [rustup](https://rustup.rs/).


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
obs_df, receiver_pos, version = read_rinex_obs("./path/to/your/obs_file.rnx")
```

### Compute Satellite Coordinates 🛰️

To compute satellite positions, you need to filter and format the ephemerides for a specific GNSS constellation.

```python
from pytecgg.satellites.ephemeris import prepare_ephemeris
from pytecgg.satellites.positions import satellite_coordinates

# Prepare the ephemerides, e.g. for Galileo
ephem_dict = prepare_ephemeris(nav_dict, constellation='Galileo')

# Compute the ECEF coordinates, e.g. for satellite E25
coords = satellite_coordinates(
    ephem_dict=ephem_dict,
    sv_id='E25',
    gnss_system='Galileo'
)
```

`ephem_dict` is a dictionary keyed by satellite ID, containing ephemeris parameters ready for coordinate computation.

Currently supported `gnss_system` are `'Galileo', 'GPS', 'BeiDou', 'QZSS'`.


## How can I help?

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature_amazing_feature`)
3. Commit your Changes (`git commit -m 'Add some amazing stuff'`)
4. Push to the Branch (`git push origin feature_amazing_feature`)
5. Open a Pull Request

