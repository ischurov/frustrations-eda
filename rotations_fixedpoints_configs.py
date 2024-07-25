import numpy as np

default_config = {
    "angles": [np.pi / 4],
    "system.lattice": "kagome2x4",
    "system.J2": 1,
    "optimization.lr": 0.1,
    "optimization.max_steps": 10000,
    "test_angles": [],
}

configs = {
    0: {},
    1: {"angles": [np.pi / 10]},
    2: {"angles": [np.pi / 4, np.pi / 8]},
    3: {"angles": [np.pi / 4, np.pi / 8, np.pi / 16]},
    4: {"angles": [1]},
    5: {"angles": [np.pi / 4], "test_angles": [1]},
}
