from unittest import TestCase

import numpy as np

from utils import hadamard_transform


class TestHadamardTransform(TestCase):
    def test_hadamard_transform(self):
        x = np.array([[1, 9, 3, 7], [5, 6, 7, 10]], dtype="float64")
        np.testing.assert_allclose(
            hadamard_transform(x), np.array([[10.0, -6.0, 0.0, -2.0], [14.0, -2.0, -3.0, 1.0]])
        )
        np.testing.assert_allclose(hadamard_transform(hadamard_transform(x)), x)
