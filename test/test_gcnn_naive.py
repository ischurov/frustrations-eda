import unittest

from gcnn_naive import GConvG
from sympy.combinatorics import Permutation
from spin_lattices import SquareLattice
import torch


class TestGConvG(unittest.TestCase):
    def test_forward_matmul(self):
        lattice = SquareLattice(4, 4)
        group = [Permutation(g) for g in lattice.get_automorphisms()]
        layer = GConvG(
            group_elements=group,
            filter_idxs=[0, 1, 5, 7],
            in_channels=2,
            out_channels=3,
        )
        x = torch.randn(100, 2, len(group))
        result = layer.forward(x)
        # print(f"{result.shape=}, {result=}")
        result2 = layer.forward_matmul(x)
        # print(f"{result2.shape=}, {result2=}")
        self.assertTrue(torch.allclose(result, result2, atol=1e-7))
