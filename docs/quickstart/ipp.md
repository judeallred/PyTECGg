# Satellite coordinates and Ionospheric Pierce Point (IPP) 🛰️

To get the satellite's position in space, we can compute ECEF coordinates for each satellite–epoch:

```python
from pytecgg.satellites import satellite_coordinates

df_coords = satellite_coordinates(
    sv_ids=df_lc["sv"],
    epochs=df_lc["epoch"],
    ephem_dict=ephem_dict,
    gnss_system="Galileo",
)

df_ = df_arcs.join(df_coords, on=["sv", "epoch"], how="left")
```

We can then compute the IPP — the intersection between the satellite–receiver line of sight and a thin-shell ionosphere at a fixed altitude:

```python
from pytecgg.satellites import calculate_ipp

# Extract satellite positions as a NumPy array
sat_ecef_array = df_.select(["sat_x", "sat_y", "sat_z"]).to_numpy()

# Compute IPP latitude and longitude, azimuth and elevation angle from
# receiver to satellite, assuming a fixed ionospheric shell height of 350 km
lat_ipp, lon_ipp, azi, ele = calculate_ipp(
    rec_pos,
    sat_ecef_array,
    h_ipp=350_000,
)

df_ = df_.with_columns(
    [
        pl.Series("lat_ipp", lat_ipp),
        pl.Series("lon_ipp", lon_ipp),
        pl.Series("azi", azi),
        pl.Series("ele", ele)
    ]
).filter(
    pl.col("ele") >= 20
)
```

Filtering out elevations below 20° is optional, but highly recommended, in order to feed cleaner data to the calibration model. However, the calibration should remain effective even without filtering, provided there are sufficient arcs.