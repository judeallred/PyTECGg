# Parse RINEX files — fast ⚡

Import the `read_rinex_nav` and `read_rinex_obs` functions from the `pytecgg.parsing` module:

```python
from pytecgg.parsing import read_rinex_nav, read_rinex_obs

NAV_PATH = "./path/to/your/nav_file.rnx"
OBS_PATH = "./path/to/your/obs_file.rnx"
```

Load a RINEX navigation file into a dictionary of DataFrames (one per constellation):
```python
nav_dict = read_rinex_nav(NAV_PATH)
```

Load a RINEX observation file and extract:

- a DataFrame of observations,
- the receiver's (approximate) position in ECEF,
- the RINEX version string.

```python
df_obs, rec_pos, rinex_version = read_rinex_obs(OBS_PATH)
rec_name = OBS_PATH.split("/")[-1][:4].lower()
```