import unittest

import numpy as np

from fourier_supervised_cleanroom import (
    how_many_terms_to_keep_fourier_weight,
    keep_fourier_weight_inplace,
    keep_largest_n,
    kept_fourier_weight,
)


class TestKeepLargestNInplace(unittest.TestCase):
    def test_normal_case(self):
        signal = np.array([1, -4, 3, 2, -5])
        result = keep_largest_n(signal, 3, inplace=True)
        expected = np.array([0, -4, 3, 0, -5])
        np.testing.assert_array_equal(result, expected)

    def test_all_positive(self):
        signal = np.array([1, 2, 3, 4, 5])
        result = keep_largest_n(signal, 2, inplace=True)
        expected = np.array([0, 0, 0, 4, 5])
        np.testing.assert_array_equal(result, expected)

    def test_all_negative(self):
        signal = np.array([-1, -2, -3, -4, -5])
        result = keep_largest_n(signal, 2, inplace=True)
        expected = np.array([0, 0, 0, -4, -5])
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        signal = np.array([])
        result = keep_largest_n(signal, 2, inplace=True)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_n_zero(self):
        signal = np.array([1, 2, 3])
        result = keep_largest_n(signal, 0, inplace=True)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_n_greater_than_length(self):
        signal = np.array([1, 2, 3])
        result = keep_largest_n(signal, 5, inplace=True)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)


class TestConsistency(unittest.TestCase):
    def test_consistency(self):
        np.random.seed(0)
        coeffs = np.random.rand(100)
        weight = 0.7

        # Test first condition
        coeffs1 = coeffs.copy()
        keep_fourier_weight_inplace(coeffs1, weight)
        terms_to_keep = how_many_terms_to_keep_fourier_weight(coeffs, weight)
        coeffs2 = coeffs.copy()
        keep_largest_n(coeffs2, terms_to_keep, inplace=True)
        self.assertTrue(
            np.all(coeffs1 == coeffs2),
            f"Failed on condition 1. coeffs1: {coeffs1}, coeffs2: {coeffs2}",
        )

        # Test second condition
        terms = how_many_terms_to_keep_fourier_weight(coeffs, weight)
        self.assertTrue(
            (kept_fourier_weight(coeffs, terms) >= weight),
            f"Failed on condition 2, kept weight too low.",
        )
        self.assertTrue(
            (kept_fourier_weight(coeffs, terms - 1) < weight),
            f"Failed on condition 2, kept weight too high.",
        )
