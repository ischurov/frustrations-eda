import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from spin_lattices import SpinLattice
from utils import make_unpacked_configurations


class SpinNN(nn.Module):
    def __init__(self, lattice: SpinLattice):
        super().__init__()
        self.lattice = lattice
        self.number_spins = lattice.number_spins
        self.basis = lattice.get_basis(
            use_symmetries=True, hamming_weight=self.number_spins // 2, spin_inversion=1
        )
        self.canonical_basis = lattice.get_basis(
            use_symmetries=False, hamming_weight=self.number_spins // 2, spin_inversion=None
        )
        self.state_info_df = lattice.get_state_info_df(
            use_symmetries=True, hamming_weight=self.number_spins // 2, spin_inversion=1
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.lattice.get_automorphisms()].swapaxes(0, 1)

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)
