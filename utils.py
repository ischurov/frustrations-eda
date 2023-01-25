import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import pandas as pd


def make_unpacked_configurations(states: npt.ArrayLike, number_spins: int):
    return (
        np.asarray(states, dtype="uint64").reshape(-1, 1)
        >> np.arange(number_spins, dtype="uint64")
    ) & 1


def batched_state_info_df(basis: ls.SpinBasis, states: npt.NDArray[np.uint64]):
    """
    Parameters
    ----------
    basis : ls.SpinBasis
        Basis to use for the state info
    states : npt.NDArray[np.uint64]
        States to get the info for

    Returns
    -------
    pd.DataFrame

    Returns a DataFrame with index states and the following columns:

    - representative: representative of the group trajectory containing the state
    - character: character of the group element that takes the representative to the state
    - norm: normalizing coefficient
    """
    representative, eigenvalue, norm = basis.state_info(states)

    return pd.DataFrame(
        dict(representative=representative, character=eigenvalue, norm=norm),
        index=states,
    )
