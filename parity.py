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


def popcount(x: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]:
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
    if x.dtype != np.uint64:
        raise ValueError("x must have dtype == 'uint64'")

    x = np.ascontiguousarray(x, dtype="uint64")
    out = np.empty_like(x, dtype="uint8")
    x_ptr = ffi.from_buffer("uint64_t[]", x, require_writable=False)
    out_ptr = ffi.from_buffer("uint8_t[]", out, require_writable=True)

    lib.popcount(x_ptr, out_ptr, x.size)

    return out


def calculate_fourier_transform_matrix(
    states: np.ndarray, subsets: np.ndarray, out_dtype="int8"
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
                               dtype == out_dtype
    """
    if out_dtype not in ("int8", "float64"):
        raise ValueError("out_dtype must be either 'int8' or 'float64'")

    out_ctype = {"int8": "int8_t", "float64": "double"}[out_dtype]

    states = np.ascontiguousarray(states, dtype="uint64")
    subsets = np.ascontiguousarray(subsets, dtype="uint64")
    out = np.empty((states.size, subsets.size), dtype=out_dtype)
    states_ptr = ffi.from_buffer("uint64_t[]", states, require_writable=False)
    subsets_ptr = ffi.from_buffer("uint64_t[]", subsets, require_writable=False)
    out_ptr = ffi.from_buffer(out_ctype + "[]", out, require_writable=True)

    if out_dtype == "int8":
        lib.calculate_fourier_transform_matrix_int8(
            states_ptr, states.size, subsets_ptr, subsets.size, out_ptr
        )
    elif out_dtype == "float64":
        lib.calculate_fourier_transform_matrix_float64(
            states_ptr, states.size, subsets_ptr, subsets.size, out_ptr
        )
    else:
        raise ValueError(f"out_dtype must be either 'int8' or 'float64'")

    return out
