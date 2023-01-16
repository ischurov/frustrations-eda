from unittest import TestCase

import numpy as np

from parity import calculate_fourier_transform_matrix, parity, popcount


class TestParity(TestCase):
    def test_parity(self):
        def parity_of_1s(x: np.ndarray, n: int, show_progress=False):
            parity = np.zeros_like(x, dtype="uint8")
            x = x.copy()
            for i in range(n):
                parity ^= x & 1
                x >>= 1
            return parity

        np.random.seed(123)
        x = np.random.randint(0, 2**64, size=10000, dtype="uint64").reshape(10, -1).T
        self.assertTrue(np.all(parity(x) == parity_of_1s(x, n=64)))

    def test_popcount(self):
        def popcount_reference(x: np.ndarray, n: int, show_progress=False):
            popcount = np.zeros_like(x, dtype="uint8")
            x = x.copy()
            for i in range(n):
                popcount += x & 1
                x >>= 1
            return popcount

        np.random.seed(1234)
        x = np.random.randint(0, 2**64, size=10000, dtype="uint64").reshape(10, -1).T
        self.assertTrue(np.all(popcount(x) == popcount_reference(x, n=64)))

    def test_calculate_fourier_transform_matrix(self):
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

        np.random.seed(42)
        states = np.random.randint(0, 2**64, size=10000, dtype="uint64")
        subsets = np.random.randint(0, 2**64, size=20000, dtype="uint64")

        self.assertTrue(
            np.all(
                calculate_fourier_transform_matrix(states, subsets)
                == cftm_reference(states, subsets, 64)
            )
        )
