import itertools
from hashlib import md5
from math import factorial
from pathlib import Path

import lattice_symmetries as ls
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import minimize
from sympy.combinatorics import Permutation

from heisenberg_hamiltonians import HeisenbergJ1J2
from spin_lattices import KagomeLattice, SquareLattice1Diag, TriangleLattice
from utils import make_unpacked_configurations


# FROM: GPT-4
def configurations_to_tensors(configurations: torch.Tensor | np.ndarray, up=1, down=0):
    # Ensure the input is a torch.Tensor
    if isinstance(configurations, np.ndarray):
        configurations = torch.from_numpy(configurations)

    # Get the indices of up and down elements
    up_indices = (configurations == up).nonzero(as_tuple=True)
    down_indices = (configurations == down).nonzero(as_tuple=True)

    # Split the indices into batch-wise arrays and create a 3D tensor
    up_indices_split = up_indices[1].view(-1, configurations.shape[1] // 2)
    down_indices_split = down_indices[1].view(-1, configurations.shape[1] // 2)

    result = torch.stack([up_indices_split, down_indices_split], dim=1)
    return result


# END FROM


class SlaterDeterminant(nn.Module):
    def __init__(
        self,
        basis: ls.Basis,
        factorial_correction=False,
        initialization="orthogonal",
        sign_cache_dir: Path | None = None,
    ):
        super().__init__()
        self.basis = basis
        self.n_sites = basis.number_bits
        unpacked_configurations = make_unpacked_configurations(
            basis.states, number_spins=self.n_sites
        ).astype("int64")

        self.configurations = configurations_to_tensors(unpacked_configurations)
        if initialization == "orthogonal":
            A = torch.randn(self.n_sites, self.n_sites, dtype=torch.float64)
            Q, R = torch.linalg.qr(A)
            half_Q = Q[:, : self.n_sites // 2]
            self.f = nn.Parameter(half_Q @ half_Q.T)
        elif initialization == "randn":
            self.f = nn.Parameter(
                torch.randn(self.n_sites, self.n_sites, dtype=torch.float64)
                / np.sqrt(self.n_sites)
            )
        else:
            raise ValueError(f"Unknown initialization {initialization}")
        signs = None
        cache_path = None
        if sign_cache_dir is not None:
            sign_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_id = md5(unpacked_configurations.tobytes()).hexdigest()

            cache_path = sign_cache_dir / f"{cache_id}.npy"

            if cache_path.exists():
                logger.debug(f"Using cached signs from file {cache_path}")
                signs = np.load(cache_path)

        if signs is None:
            signs = []
            logger.debug("Finding signs...")
            for configuration in self.configurations:
                perm_initial = list(
                    itertools.chain(*zip(configuration[0, :], configuration[1, :]))
                )
                signs.append(Permutation(perm_initial).signature())
            logger.debug("Ready")
            signs = np.array(signs)
            if cache_path is not None:
                logger.debug(f"Saving signs to file {cache_path}")
                np.save(cache_path, signs)

        self.signs = torch.from_numpy(signs)
        self.factorial_correction = factorial_correction

    def forward(self, x: torch.Tensor):
        configs = self.configurations[x.view(-1)]
        signs = self.signs[x.view(-1)]
        # x[..., 0, :] is the spin up part
        # x[..., 1, :] is the spin down part
        # x[..., 0, i] is the i-th spin up site
        # x[..., 1, i] is the i-th spin down site
        # it is expected that x[..., i] < x[..., j] for i < j
        rows = configs[:, 0, :]
        columns = configs[:, 1, :]

        row_indices = rows.unsqueeze(-1).expand(rows.shape[0], rows.shape[1], columns.shape[1])
        col_indices = columns.unsqueeze(1).expand(rows.shape[0], rows.shape[1], columns.shape[1])

        matrices = self.f[row_indices, col_indices]

        return (torch.linalg.det(matrices) * signs).reshape(x.shape) * (
            factorial(self.n_sites // 2) if self.factorial_correction else 1
        )
