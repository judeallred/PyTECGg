# Speed of light in m/s
C: float = 299792458.0

FREQ_BANDS: dict[str, dict] = {
    "G": {"L1": 1575.42e6, "L2": 1227.60e6},
    "R": {
        "L1": lambda n: (1602 + n * 0.5625) * 1e6,
        "L2": lambda n: (1246 + n * 0.4375) * 1e6,
    },
    "E": {"L1": 1575.42e6, "L2": 1176.45e6},
    "C": {"L1": 1561.098e6, "L5": 1207.14e6},
}

# Priorities for frequency pairs
PHASE_FREQ_PRIORITY = {
    "G": [("L2", "L1"), ("L5", "L1")],
    "E": [("L5", "L1"), ("L7", "L1"), ("L8", "L1")],
    "C": [("L6", "L1"), ("L7", "L1"), ("L5", "L1")],
    "R": [("L2", "L1")],
}
CODE_FREQ_PRIORITY = {
    "G": [("C2", "C1"), ("C5", "C1")],
    "E": [("C5", "C1"), ("C7", "C1"), ("C8", "C1")],
    "C": [("C6", "C2"), ("C7", "C2"), ("C5", "C2")],
    "R": [("C2", "C1")],
}

# Priorities for channel suffixes
PHASE_CHAN_PRIORITY = ["C", "W", "X", "P"]
CODE_CHAN_PRIORITY = ["C", "W", "X", "P"]
