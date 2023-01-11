import numpy as np
import numpy.typing as npt


def make_unpacked_configurations(states: npt.ArrayLike, number_spins: int):
    return (
        np.asarray(states, dtype="uint64").reshape(-1, 1)
        >> np.arange(number_spins, dtype="uint64")
    ) & 1
