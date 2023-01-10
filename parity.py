import numpy as np
import numpy.typing as npt

from _parity import ffi, lib


def parity(x: np.ndarray) -> np.ndarray:
    """
    Calculates the parities of number of 1's for all elements of x

    Params
    ------
    x : np.ndarray
        should have dtype == 'uint64'

    Returns
    -------
    parity : np.ndarray
             dtype == 'uint8'
    """

    x = np.ascontiguousarray(x, dtype="uint64")
    out = np.empty_like(x, dtype="uint8")
    x_ptr = ffi.from_buffer("uint64_t[]", x, require_writable=False)
    out_ptr = ffi.from_buffer("uint8_t[]", out, require_writable=True)

    lib.parity(x_ptr, out_ptr, x.size)

    return out


def calculate_fourier_transform_matrix(
    states: np.ndarray, subsets: np.ndarray
) -> npt.NDArray[np.int8]:
    """
    Calculates the fourier transform matrix for a given set of states and subsets

    Params
    ------
    states : np.ndarray
             dtype == 'uint64'
    subsets : np.ndarray
              dtype == 'uint64'
    number_spins : int
                   number of spins in the system

    Returns
    -------
    fourier_transform_matrix : np.ndarray
                               dtype == 'uint8'
    """

    states = np.ascontiguousarray(states, dtype="uint64")
    subsets = np.ascontiguousarray(subsets, dtype="uint64")
    out = np.empty((states.size, subsets.size), dtype="int8")
    states_ptr = ffi.from_buffer("uint64_t[]", states, require_writable=False)
    subsets_ptr = ffi.from_buffer("uint64_t[]", subsets, require_writable=False)
    out_ptr = ffi.from_buffer("int8_t[]", out, require_writable=True)

    lib.calculate_fourier_transform_matrix(
        states_ptr, states.size, subsets_ptr, subsets.size, out_ptr
    )

    return out


def test_parity():
    def parity_of_1s(x: np.ndarray, n: int, show_progress=False):
        parity = np.zeros_like(x, dtype="uint8")
        x = x.copy()
        for i in range(n):
            parity ^= x & 1
            x >>= 1
        return parity

    x = np.random.randint(0, 2**64, size=10000, dtype="uint64").reshape(10, -1).T
    assert np.all(parity(x) == parity_of_1s(x, n=64))


def test_calculate_fourier_transform_matrix():
    def cftm_reference(
        states: np.ndarray, subsets: np.ndarray, number_spins: int, show_progress=False
    ) -> np.ndarray:
        """
        This is a low-level function that calculates the Fourier Transform Matrix.

        Warning! This function returns np.array with dtype int8! This is memory-efficient,
        but can lead to overfulls. When using this matrix, make sure that other operands
        are of larger type (i.e. float64).

        See details in get_fourier_transform_matrix
        """

        masks = subsets.reshape(1, -1)
        masked = states.reshape(-1, 1) & masks
        parities = parity(masked)
        return parities.astype("int8") * 2 - 1

    states = np.random.randint(0, 2**64, size=10000, dtype="uint64")
    subsets = np.random.randint(0, 2**64, size=20000, dtype="uint64")

    assert np.all(
        calculate_fourier_transform_matrix(states, subsets, 64)
        == cftm_reference(states, subsets, 64)
    )


if __name__ == "__main__":
    test_parity()
    test_calculate_fourier_transform_matrix()
