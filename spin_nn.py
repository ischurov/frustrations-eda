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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x_with_translations = x[:, self.lattice.get_automorphisms()].swapaxes(0, 1)
        x_with_sign_inverse = 1 - x_with_translations
        return torch.cat((x_with_translations, x_with_sign_inverse), dim=0)

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)


class FC1SpinNN(SpinNN):
    def __init__(self, lattice: SpinLattice, hidden_size: int):
        super().__init__(lattice)
        input_size = lattice.number_spins
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(inp)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.postprocess(x)
