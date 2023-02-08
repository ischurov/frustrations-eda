import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger


def make_unpacked_configurations(states: npt.ArrayLike, number_spins: int):
    initial_shape = np.shape(states)
    return (
        (
            np.asarray(states, dtype="uint64").reshape(-1, 1)
            >> np.arange(number_spins, dtype="uint64")
        )
        & 1
    ).reshape(initial_shape + (number_spins,))


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


### BASED ON: https://github.com/amitport/hadamard-transform
### MIT LICENSE
def hadamard_transform(x: npt.NDArray[np.float64], verbose: bool = False):

    """Fast Walsh–Hadamard transform

    The hadamard transform is not numerically stable by nature (lots of subtractions),
    it is recommended to use with float64 when possible

    :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
              Each vector's length is expected to be a power of 2! (or each row if it is batched)
    :return: The normalized Hadamard transform of each vector in x
    """
    original_shape = x.shape
    assert 1 <= len(original_shape) <= 2, "input's dimension must be either 1 or 2"
    if len(original_shape) == 1:
        # add fake 1 batch dimension
        # for making the code a follow a single (batched) path
        x = x[None, :]
    batch_dim, d = x.shape

    h = 2
    while h <= d:
        logger.debug(f"iteration {np.log2(h)} of {np.log2(d)}")
        if verbose:
            print(f"iteration {np.log2(h)} of {np.log2(d)}")

        hf = h // 2

        x = x.view()
        x.shape = (batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = np.concatenate((half_1 + half_2, half_1 - half_2), axis=-1)

        h *= 2

    return (x / np.sqrt(d)).reshape(*original_shape)


### END BASED
