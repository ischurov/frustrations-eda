import itertools
import os
import pickle
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from random import shuffle
from typing import Iterable, Literal

import numpy as np
import numpy.typing as npt
import sympy as sp
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import dual_annealing
from sympy.combinatorics import permutations

from spin_systems import HeisenbergJ1J2
from spin_lattices import SquareLattice1Diag, TriangularLattice

self_name = Path(__file__).name
output_dir = Path("experiments") / self_name.removesuffix(".py")
output_dir.mkdir(parents=True, exist_ok=True)
lattice = SquareLattice1Diag(2, 3, boundary_conditions="open")
logger.debug(f"Lattice: {lattice.get_cache_id()}, {lattice.number_spins=}")
system = HeisenbergJ1J2(lattice, J1=1, J2=1, use_symmetries=False, spin_inversion=None)
logger.debug(f"System: {system.get_cache_id()}")


class SlaterDeterminant(nn.Module):
    def __init__(self, n_sites: int):
        super().__init__()
        self.n_sites = n_sites
        self.f = nn.Parameter(torch.randn(n_sites, n_sites))

    def forward(self, x: torch.Tensor):
        assert x.shape[-2:] == (2, self.n_sites // 2)
        # x[..., 0, :] is the spin up part
        # x[..., 1, :] is the spin down part
        # x[..., 0, i] is the i-th spin up site
        # x[..., 1, i] is the i-th spin down site
        # it is expected that x[..., i] < x[..., j] for i < j
        original_shape = x.shape
        x = x.reshape(-1, 2, self.n_sites // 2)

        signs = []
        for i in range(x.shape[0]):
            perm_initial = list(itertools.chain(*zip(x[i, 0, :], x[i, 1, :])))
            signs.append(permutations.Permutation(perm_initial).signature())

        rows = x[..., 0, :]
        columns = x[..., 1, :]

        row_indices = rows.unsqueeze(-1).expand(rows.shape[0], rows.shape[1], columns.shape[1])
        col_indices = columns.unsqueeze(1).expand(rows.shape[0], rows.shape[1], columns.shape[1])

        matrices = self.f[row_indices, col_indices]

        return (torch.linalg.det(matrices) * torch.tensor(signs)).reshape(original_shape[:-2])


def configuration_to_tensor(configuration: Iterable, up=1, down=0):
    return torch.tensor(
        [
            [i for i, c in enumerate(configuration) if c == up],
            [i for i, c in enumerate(configuration) if c == down],
        ]
    )


def sign_overlap(true: npt.NDArray[np.float64], pred: npt.NDArray[np.float64]) -> float:
    probs = np.abs(true) ** 2
    return float(np.sum((np.sign(true) * np.sign(pred)) * probs) / np.sum(probs))


if __name__ == "__main__":
    system.get_eigenstates(1)

    determinant = SlaterDeterminant(system.number_spins)
    ground_state = system.get_df_ground_state(canonical_basis=True, unpack_configurations=True)
    configurations_tensor = torch.stack(
        list(ground_state.configuration.apply(configuration_to_tensor).values)
    )
    ground_state_coeffs = ground_state["eigenstate_coeff"].values.astype("float64")

    @torch.no_grad()
    def loss(f):
        determinant.f = nn.Parameter(
            torch.tensor(f.reshape(system.number_spins, system.number_spins))
        )
        return -sign_overlap(ground_state_coeffs, determinant(configurations_tensor).numpy())

    logger.debug("Starting annealing")
    anneal_res = dual_annealing(loss, bounds=[(-10.0, 10.0)] * system.number_spins**2)
    logger.debug("Annealing done")
    (output_dir / f"anneal_res_{system.get_cache_id()}.pkl").write_bytes(pickle.dumps(anneal_res))
