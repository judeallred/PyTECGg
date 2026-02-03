# TEC Calibration

The `tec_calibration` module implements the core scientific logic for estimating and removing instrumental biases from GNSS observations. The calibration process is based on the algorithm described by [**Ciraolo et al. (2007)**](https://link.springer.com/article/10.1007/s00190-006-0093-1). To account for the magnetic control of the ionosphere, the module takes into account Modified Dip latitude (MoDip), which provides a more physically accurate representation of ionospheric structures than standard geographic latitude.

## Key Stages

1.  **Arc identification & levelling**: cycle slips and loss-of-lock events are detected to extract continuous observation "arcs"; phase measurements are then levelled to the (unambiguous) code measurements to reduce noise while maintaining continuity.
2.  **Bias estimation**: a polynomial expansion in a MoDip/Longitude frame is evaluated and the resulting system is solved using via [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) to separate the ionospheric signal from the combined satellite-receiver biases.
3.  **Calibrated output**: estimated biases are removed to provide the calibrated slant (sTEC) and vertical (vTEC) TEC values.

---

## API Reference

::: pytecgg.tec_calibration
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
      docstring_section_style: table
      members:
        - extract_arcs
        - calculate_tec
        - extract_modip