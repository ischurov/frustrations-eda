import numpy as np

def make_unpacked_configurations(states, number_spins):
    return (
        np.array(states, dtype="uint64").reshape(-1, 1)
        >> np.arange(number_spins, dtype="uint64")
    ) & 1
