from _parity import ffi, lib
import numpy as np


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


def test():
    def parity_of_1s(x: np.ndarray, n: int, show_progress=False):

        parity = np.zeros_like(x, dtype="uint8")
        x = x.copy()
        for i in [lambda _: _, tqdm][show_progress](range(n)):
            parity ^= x & 1
            x >>= 1
        return parity

    x = np.random.randint(0, 2**64, size=10000, dtype="uint64").reshape(10, -1).T
    assert np.all(parity(x) == parity_of_1s(x, n=64))


if __name__ == "__main__":
    test()
