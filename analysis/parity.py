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


def popcount(x: np.ndarray) -> np.ndarray:
    """
    Calculates the number of 1's for all elements of x

    Params
    ------
    x : np.ndarray
        should have dtype == 'uint64'

    Returns
    -------
    popcount : np.ndarray
               dtype == 'uint8'
    """

    x = np.ascontiguousarray(x, dtype="uint64")
    out = np.empty_like(x, dtype="uint8")
    x_ptr = ffi.from_buffer("uint64_t[]", x, require_writable=False)
    out_ptr = ffi.from_buffer("uint8_t[]", out, require_writable=True)

    lib.popcount(x_ptr, out_ptr, x.size)

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
