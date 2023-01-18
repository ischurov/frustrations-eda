from unittest import TestCase

import numpy as np

from boolean_fourier_learner import BooleanFourierLearner
from parity import calculate_fourier_transform_matrix, parity, popcount


class TestBooleanFourierLearner(TestCase):
    def test_boolean_fourier_learner(self):
        np.random.seed(123)
        number_spins = 8
        x = np.arange(2**number_spins, dtype="uint64")
        y = np.random.uniform(size=2**number_spins)
        learner = BooleanFourierLearner(number_spins=number_spins)
        learner.fit(x, y, batch_size=15)

        fourier_transform_matrix = calculate_fourier_transform_matrix(x, learner.subsets)
        prediction = fourier_transform_matrix @ learner.coeffs_
        np.testing.assert_allclose(prediction, y)
        