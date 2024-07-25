import sys

sys.path.append(".")

from loguru import logger
from scipy.sparse import csr_matrix

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from unittest import TestCase

import numpy as np
import torch

from misc_utils import (
    force_csr_symmetric,
    groupby_shuffle,
    hadamard_transform,
    hadamard_transform_pytorch_inplace,
    make_packed_configurations,
    make_unpacked_configurations,
    kronecker_power,
    kronecker_power_pytorch,
    rotation_matrix,
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


class TestForceCsrSymmetric(TestCase):
    def test_force_csr_symmetric(self):
        dense = np.random.uniform(-1, 1, size=(10, 10))
        mask = np.random.choice([True, False, False], size=(10, 10))
        noise = np.random.uniform(-1e-6, 1e-6, size=(10, 10))
        dense[mask] = 0
        noise[mask] = 0
        dense = dense + dense.T + noise
        sparse = csr_matrix(dense)
        symmetric_sparse = force_csr_symmetric(sparse)
        self.assertTrue((symmetric_sparse != symmetric_sparse.transpose()).nnz == 0)
        self.assertTrue(np.abs(symmetric_sparse - sparse).max() < 1e-5)


class TestKroneckerPower(TestCase):
    def test_hadamard_transform_is_kronecker_power(self):
        x = np.random.rand(2**8)
        matrix = rotation_matrix(np.pi / 4)[[1, 0], :]
        obtained = kronecker_power(x, matrix)
        expected = hadamard_transform(x)
        self.assertTrue(
            np.allclose(
                obtained,
                expected,
            )
        )

    def test_kronecker_power_torch(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Test cases with different input sizes
        input_sizes = [2, 4, 8, 16, 32]

        for size in input_sizes:
            # Generate random input vector
            x_np = np.random.rand(size)
            x_torch = torch.from_numpy(x_np)

            # Generate random 2x2 transform matrix
            transform_np = np.random.rand(2, 2)
            transform_torch = torch.from_numpy(transform_np)

            # Compute results using both functions
            result_np = kronecker_power(x_np, transform_np)
            result_torch = kronecker_power_pytorch(x_torch, transform_torch)

            # Convert PyTorch result to NumPy for comparison
            result_torch_np = result_torch.numpy()

            # Compare results
            self.assertTrue(
                np.allclose(
                    result_np,
                    result_torch_np,
                    rtol=1e-5,
                    atol=1e-8,
                    # err_msg=f"Results don't match for input size {size}",
                )
            )

            print(f"Test passed for input size {size}")
