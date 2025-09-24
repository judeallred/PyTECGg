import numpy as np

ALTITUDE_KM = 350  # Altitude in km
RESOLUTION = 1  # Grid resolution in degrees
LONGITUDES = np.arange(-180, 180 + RESOLUTION, RESOLUTION)
LATITUDES = np.arange(-90, 90 + RESOLUTION, RESOLUTION)
