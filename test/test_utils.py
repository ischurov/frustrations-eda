import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from unittest import TestCase

import numpy as np

from utils import hadamard_transform, hadamard_transform_pytorch_inplace


class TestHadamardTransform(TestCase):
    def test_hadamard_transform(self):
        x = np.array([[1, 9, 3, 7], [5, 6, 7, 10]], dtype="float64")
        np.testing.assert_allclose(
            hadamard_transform(x), np.array([[10.0, -6.0, 0.0, -2.0], [14.0, -2.0, -3.0, 1.0]])
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
            hadamard_transform_pytorch_inplace(hadamard_transform_pytorch_inplace(x)).numpy(),
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

