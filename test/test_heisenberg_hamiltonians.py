from unittest import TestCase

import numpy as np

from heisenberg_hamiltonians import pad_right


class TestPadRight(TestCase):
    def test_pad_right(self):
        self.assertTrue(
            (
                pad_right(np.array([1, 2, 3]), 8)
                == np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            ).all()
        )
