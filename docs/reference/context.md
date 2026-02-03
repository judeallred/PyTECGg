# Context

The `context` module provides a unified state management system for the entire processing pipeline. `GNSSContext` ensures that all modules – from parsing to final calibration – operate with consistent settings, receiver position, ionospheric shell height, and GNSS constellations, thus acting as a single source of truth.

---

## API Reference

::: pytecgg.context
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
      docstring_section_style: table
      members:
        - GNSSContext