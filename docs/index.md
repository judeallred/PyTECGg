![logo](images/pytecgg_logo.svg)

<p align="center">
  <a href="https://pypi.org/project/pytecgg/">
    <img src="https://img.shields.io/pypi/v/pytecgg.svg" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/badge/python-3.11--3.13-blue.svg" alt="Python version">
  <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License">
  <img src="https://img.shields.io/pypi/dm/PyTECGg" alt="Downloads">
</p>

<p align="center">
  <img src="https://github.com/viventriglia/PyTECGg/actions/workflows/pytest.yml/badge.svg" alt="Tests">
  <img src="https://github.com/viventriglia/pytecgg/actions/workflows/build_publish.yml/badge.svg" alt="CI">
</p>

---

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core.

## What is PyTECGg?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the **[Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content)** (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

## Main features

This package:

- Is open source: read and access [all the code](https://github.com/viventriglia/PyTECGg).
- Supports all modern GNSS constellations:
    - GPS, Galileo, BeiDou, GLONASS.
- Supports RINEX 2, 3 and 4.
- Provides seamless decompression for RINEX files:
    - No need to preprocess your Hatanaka-compressed files.

<!-- TODO -->
<!-- !!! info "Citing PyTECGg"
    If you use PyTECGg for your research, please cite ... -->

## Get started

📥 [**Install**](installation.md)

🚀 [**Quickstart**](quickstart/parsing-and-utils.md)

🌱 [**Contribute**](contributing.md)