# RINEX utilities 🛠️

PyTECGg provides utility functions to automate the retrieval of observation files and navigation messages from reliable servers.

## Observation files from RING (INGV)

The `download_obs_ring` function fetches RINEX observation files from the [INGV RING](https://webring.gm.ingv.it/) server. This network manages continuous GNSS stations across Italy and archives GNSS data from thousands of stations along the Eurasia-Africa plate boundary.

```python
from pathlib import Path
from pytecgg.utils import download_obs_ring

# Define the root directory for data
data_path = Path("./data")

download_obs_ring(
    station_code="GRO200ITA",
    year=2025,
    doys=[1, 55, 252],
    output_path=data_path
)
```

The function creates a subdirectory named after the `station_code` under the provided `output_path`. The above example will produce the following structure:

```bash
data/
└── GRO2/
    ├── GRO200ITA_R_20250010000_01D_30S_MO.crx.gz
    ├── GRO200ITA_R_20250550000_01D_30S_MO.crx.gz
    └── GRO200ITA_R_20252520000_01D_30S_MO.crx.gz
```

The function also handles station naming conventions: if a 4-character code like `GRO2` is provided, it is converted to the RINEX 3 long-name format (e.g., `GRO200ITA`) to match the server structure.

## Navigation files from BKG

Multi-constellation navigation messages can be downloaded from the [BKG](https://igs.bkg.bund.de/) GNSS Data Center.

```python
from pathlib import Path
from pytecgg.utils import download_nav_bkg

download_nav_bkg(
    year=2025,
    doys=[1, 55, 252],
    output_path=data_path
)
```

BRDC files provide aggregated navigation data from the global [IGS network](https://network.igs.org/), ensuring coverage for GPS, GLONASS, Galileo, and BeiDou. The RINEX files are saved under the provided `output_path`.