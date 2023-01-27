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

        orbit_lengths = self.state_info_df.groupby("representative").size()
        self.orbits_lcm = np.lcm.reduce(orbit_lengths)

    def preprocess(self, x: npt.NDArray[np.uint64]) -> torch.Tensor:
        print("Preprocessing...")
        print("Making orbits...")
        df = (
            self.state_info_df.reset_index()
            .rename(columns={"index": "state"})
            .merge(
                self.state_info_df.loc[x]
                .reset_index()
                .rename(columns={"index": "input_state"})[["input_state", "representative"]],
                on="representative",
                how="inner",
            )
        )

        print("Finding extended_states...")

        extended_states = []
        for _, group in df.groupby("input_state"):
            extended_states.append(
                np.tile(group["state"].values, self.orbits_lcm // group.shape[0])
            )

        states = np.array(extended_states).T

        print("Unpacking configurations...")
        unpacked_configurations = make_unpacked_configurations(
            states,
            self.number_spins,
        ).astype(np.float32)

        return torch.tensor(
            unpacked_configurations,
            dtype=torch.float32,
        )

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)
