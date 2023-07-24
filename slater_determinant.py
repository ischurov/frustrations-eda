import itertools
from hashlib import md5
from itertools import product
from math import factorial
from pathlib import Path
from typing import Callable, Literal

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from scipy.linalg import schur
from scipy.optimize import minimize
from sympy.combinatorics import Permutation

from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from spin_lattices import (
    KagomeLattice,
    SpinLattice,
    SquareLattice1Diag,
    TriangleLattice,
)
from misc_utils import make_unpacked_configurations


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

Initializer = Callable[[SpinLattice], npt.NDArray[np.float64]]


def tight_binding_init(
    ts: list[float],
    keep_symmetries: Literal["x", "xy"] | None = "xy",
) -> Initializer:
    def tight_binding(lattice: SpinLattice) -> npt.NDArray[np.float64]:
        adj = np.zeros((lattice.number_spins, lattice.number_spins), dtype=np.int64)
        for edge, kind in lattice.edges_to_kind.items():
            adj[edge[0], edge[1]] = 1
            adj[edge[1], edge[0]] = 1

        tight_binding_hamiltonian = np.zeros(
            (lattice.number_spins, lattice.number_spins), dtype=np.complex128
        )

        nbd = adj
        for t in ts:
            tight_binding_hamiltonian += t * ((tight_binding_hamiltonian == 0) & (nbd > 0))
            nbd = nbd @ adj

        energies, orbitals = np.linalg.eigh(tight_binding_hamiltonian)

        if keep_symmetries is not None and "x" in keep_symmetries:
            tr_x = lattice.x_translation

            ### BASED ON: Nikita Astrakhantsev's code
            e_round = np.around(energies, decimals=7)
            orbitals_tx = orbitals * 0.0 + 0.0j

            tx_momenta = np.zeros(
                orbitals.shape[0], dtype=np.complex128
            )  # this is later needed to select joint (e, t_x) sectors
            for e_sector in np.unique(e_round):
                idxs = np.where(e_round == e_sector)[0]

                tx_matrix = orbitals[:, idxs].conj().T @ orbitals[:, idxs][tr_x]

                momenta, Um = schur(tx_matrix)[:2]
                momenta = np.diag(momenta)
                orbitals_tx[:, idxs] = orbitals[:, idxs] @ Um
                tx_momenta[idxs] = momenta
            orbitals = orbitals_tx
            tx_momenta = np.around(tx_momenta, decimals=5)

            if "y" in keep_symmetries:
                tr_y = lattice.y_translation
                orbitals_ty = orbitals * 0.0 + 0.0j

                for e_sector, kx_sector in product(np.unique(e_round), np.unique(tx_momenta)):
                    idxs = np.where((e_round == e_sector) & (tx_momenta == kx_sector))[0]
                    if len(idxs) == 0:
                        continue
                    ty_matrix = orbitals[:, idxs].conj().T @ orbitals[:, idxs][tr_y]

                    momenta, Um = schur(ty_matrix)[:2]
                    momenta = np.diag(momenta)
                    orbitals_ty[:, idxs] = orbitals[:, idxs].dot(Um)
                orbitals = orbitals_ty

            ### END BASED
        half_orbitals = orbitals[:, : lattice.number_spins // 2]
        f_ij = half_orbitals @ half_orbitals.T.conj()
        return f_ij.astype(np.float64)

    return tight_binding


class SlaterDeterminant(nn.Module):
    def __init__(
        self,
        lattice: SpinLattice,
        basis: ls.Basis,
        factorial_correction=False,
        initialization: str | Initializer = "orthogonal",
        sign_cache_dir: Path | None = None,
    ):
        super().__init__()
        self.lattice = lattice
        self.basis = basis
        self.n_sites = self.basis.number_bits
        unpacked_configurations = make_unpacked_configurations(
            self.basis.states, number_spins=self.n_sites
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
        elif isinstance(initialization, Callable):
            self.f = nn.Parameter(torch.from_numpy(initialization(self.lattice)))
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


class HalfSpaceProjector(nn.Module):
    def forward(self, orth: torch.Tensor):
        n = orth.shape[0]
        half_orth = orth[:, : n // 2]
        return half_orth @ half_orth.T

    def right_inverse(self, f):
        return f


class TwoHalvesOuterProduct(nn.Module):
    def forward(self, orth: torch.Tensor):
        n = orth.shape[0]
        half1_orth = orth[:, : n // 2]
        half2_orth = orth[:, n // 2 :]
        return half1_orth @ half2_orth.T

    def right_inverse(self, f):
        return f
