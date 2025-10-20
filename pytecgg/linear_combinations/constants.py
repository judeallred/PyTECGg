# Speed of light in m/s
C: float = 299792458.0

FREQ_BANDS: dict[str, dict] = {
    "G": {"L1": 1575.42e6, "L2": 1227.60e6},
    "R": {
        "L1": lambda n: (1602 + n * 0.5625) * 1e6,
        "L2": lambda n: (1246 + n * 0.4375) * 1e6,
    },
    "E": {"L1": 1575.42e6, "L5": 1176.45e6},
    "C": {"L1": 1561.098e6, "L5": 1207.14e6},
}

# OBS_MAPPING: dict[str, dict] = {
#     "G": {"phase": {"L1": "L1", "L2": "L2"}, "code": {"C1": "C1", "P2": "P2"}},
#     "R": {"phase": {"L1": "L1C", "L2": "L2C"}, "code": {"C1": "C1C", "C2": "C2C"}},
#     "E": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
#     "C": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
# }

# OBS_MAPPING: dict[str, dict] = {
#     "G": {"phase": {"L1": "L1C", "L2": "L2W"}, "code": {"C1": "C1C", "C2": "C2W"}},
#     "R": {"phase": {"L1": "L1C", "L2": "L2C"}, "code": {"C1": "C1C", "C2": "C2C"}},
#     "E": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
#     "C": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
# }

OBS_MAPPING_BY_VERSION = {
    "2": {
        "G": {"phase": {"L1": "L1", "L2": "L2"}, "code": {"C1": "C1", "C2": "P2"}},
        "R": {"phase": {"L1": "L1", "L2": "L2"}, "code": {"C1": "C1", "C2": "C2"}},
        "E": {"phase": {"L1": "L1", "L5": "L5"}, "code": {"C1": "C1", "C5": "C5"}},
        "C": {"phase": {"L1": "L1", "L5": "L5"}, "code": {"C1": "C1", "C5": "C5"}},
    },
    "3": {
        "G": {"phase": {"L1": "L1C", "L2": "L2W"}, "code": {"C1": "C1C", "C2": "C2W"}},
        "R": {"phase": {"L1": "L1C", "L2": "L2C"}, "code": {"C1": "C1C", "C2": "C2C"}},
        "E": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
        "C": {"phase": {"L1": "L1X", "L5": "L5X"}, "code": {"C1": "C1X", "C5": "C5X"}},
    },
}
