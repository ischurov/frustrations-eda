from unittest import TestCase

import numpy as np

from fast_boolean_analysis import keep_largest_n


class TestTruncateStrategies(TestCase):
    def test_keep_largest_n(self):

        np.random.seed(123)

        x = np.random.uniform(-1, 1, size=1000)
        assert len(set(x)) == 1000

        obtained = keep_largest_n(10)(x)
        x_abs_sorted = np.sort(np.abs(x))
        expected = np.abs(x) >= x_abs_sorted[-10]

        np.testing.assert_equal(obtained, expected)
        self.assertEqual(np.sum(obtained), 10)
        self.assertTrue(max(x[obtained]) > min(x[~obtained]))
