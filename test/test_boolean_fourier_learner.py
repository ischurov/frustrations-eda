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

    def test_hadamard(self):
        np.random.seed(123)
        number_spins = 8

        x = np.random.choice(np.arange(2**number_spins, dtype="uint64"), 100, replace=False)
        # x = np.arange(2**number_spins, dtype="uint64")
        y = np.random.uniform(size=x.shape[0])

        subsets = np.random.choice(
            np.arange(2**number_spins, dtype="uint64"),
            size=2 ** (number_spins - 1) - 1,
            replace=False,
        )
        learner = BooleanFourierLearner(number_spins=number_spins, subsets=subsets, hadamard=True)
        learner.fit(x, y)

        reference_learner = BooleanFourierLearner(number_spins=number_spins, subsets=subsets)
        reference_learner.fit(x, y)
        print(f"{learner.coeffs_=}")
        print(f"{reference_learner.coeffs_=}")

        np.testing.assert_allclose(learner.coeffs_, reference_learner.coeffs_)
        self.assertTrue((learner.x_ == reference_learner.x_).all())
        np.testing.assert_allclose(learner.y_, reference_learner.y_)
        self.assertTrue((learner.subsets == reference_learner.subsets).all())
