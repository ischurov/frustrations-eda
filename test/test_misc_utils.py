import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from unittest import TestCase

import numpy as np
import torch

from misc_utils import (
    hadamard_transform,
    hadamard_transform_pytorch_inplace,
    make_packed_configurations,
    make_unpacked_configurations,
    groupby_shuffle,
)


class TestMakeUnpackedConfigurations(TestCase):
    def test_unpacked_configurations(self):
        x = np.array([1, 2, 3, 5, 7], dtype="uint64")
        number_spins = 3
        np.testing.assert_allclose(
            make_packed_configurations(
                make_unpacked_configurations(x, number_spins), number_spins
            ),
            x,
        )

    def test_unpacked_configurations_torch(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([1, 2, 3, 5, 7], device=device, dtype=torch.int64)
        number_spins = 3
        output = make_unpacked_configurations(x, number_spins)
        self.assertEqual(output.device, device)
        expected = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],
            device=device,
            dtype=torch.int64,
        )

        torch.testing.assert_close(output, expected, rtol=0, atol=0)


class TestHadamardTransform(TestCase):
    def test_hadamard_transform(self):
        x = np.array([[1, 9, 3, 7], [5, 6, 7, 10]], dtype="float64")
        np.testing.assert_allclose(
            hadamard_transform(x),
            np.array([[10.0, -6.0, 0.0, -2.0], [14.0, -2.0, -3.0, 1.0]]),
        )
        np.testing.assert_allclose(hadamard_transform(hadamard_transform(x)), x)

    def test_hadamard_transform_pytorch_inplace(self):
        import torch

        x = torch.tensor([[1, 9, 3, 7], [5, 6, 7, 10]], dtype=torch.float64)
        x_bkp = x.clone()
        np.testing.assert_allclose(
            hadamard_transform_pytorch_inplace(x).numpy(),
            np.array([[10.0, -6.0, 0.0, -2.0], [14.0, -2.0, -3.0, 1.0]]),
        )
        x = x_bkp.clone()
        np.testing.assert_allclose(
            hadamard_transform_pytorch_inplace(
                hadamard_transform_pytorch_inplace(x)
            ).numpy(),
            x_bkp.numpy(),
        )

    def test_hadamard_transform_pytorch_inplace2(self):
        import torch

        torch.manual_seed(0)

        x = torch.randn(100, 2048, dtype=torch.float64)
        x_bkp = x.clone()
        np.testing.assert_allclose(
            hadamard_transform_pytorch_inplace(x).numpy(),
            hadamard_transform(x_bkp.numpy()),
        )


class TestGroupByShuffle(TestCase):
    def test_groupby_shuffle(self):
        # Test 1
        values = np.array([1, 2, 3, 4, 5])
        groups = np.array([1, 2, 1, 3, 2])
        shuffled_values = groupby_shuffle(values, groups)
        # Check that the shuffled array has the same elements as the original one
        self.assertTrue(np.array_equal(np.sort(shuffled_values), np.sort(values)))
        # Check that groups correspondence is preserved
        self.assertTrue(
            np.array_equal(groups, groups[np.argsort(np.argsort(shuffled_values))])
        )

        # Test 2
        values = np.array([10, 20, 30, 40, 50])
        groups = np.array([1, 1, 2, 2, 3])
        shuffled_values = groupby_shuffle(values, groups)
        # Check that the shuffled array has the same elements as the original one
        self.assertTrue(np.array_equal(np.sort(shuffled_values), np.sort(values)))
        # Check that groups correspondence is preserved
        self.assertTrue(
            np.array_equal(groups, groups[np.argsort(np.argsort(shuffled_values))])
        )

        # Test 3
        values = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        groups = np.array([2, 1, 2, 1, 2])
        shuffled_values = groupby_shuffle(values, groups)
        # Check that the shuffled array has the same elements as the original one
        self.assertTrue(np.array_equal(np.sort(shuffled_values), np.sort(values)))
        # Check that groups correspondence is preserved
        self.assertTrue(
            np.array_equal(groups, groups[np.argsort(np.argsort(shuffled_values))])
        )
