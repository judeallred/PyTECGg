![logo](images/pytecgg_logo.svg)

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core.

## What is PyTECGg?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the **[Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content)** (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

## Main features

This package:

- is open source: read and access all the code!
- supports all modern GNSS constellations, codes and signals:
    - GPS, Galileo, BeiDou, GLONASS
- supports RINEX V2-3-4
- provides seamless decompression for RINEX files

## Get started

📥 [**Install**](installation.md)

🚀 [**Quickstart**](quickstart/parsing.md)

🌱 [**Contribute**](contributing.md)