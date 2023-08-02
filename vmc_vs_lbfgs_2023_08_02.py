from spin_lattices import KagomeLattice, SpinLattice, ChainLattice, SquareLattice, TriangleLattice
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from pathlib import Path
import networkx as nx
import numpy as np
from typing import Callable
import torch
import numpy.typing as npt
import lattice_symmetries as ls
from typing import Any, Optional, Union, Dict, Tuple
from loguru import logger
from collections import namedtuple
from torch import Tensor
import torch.nn as nn
from misc_utils import make_unpacked_configurations
import io
from contextlib import redirect_stderr
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from nqs_playground_helpers import (
    SamplingOptions,
    split_into_batches,
    safe_exp,
    sample_exactly,
    sample_full,
    forward_with_batches,
)
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import connected_components
import sys
from kagome_cnn import KagomeCNNRegression
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import time
from scipy.optimize import minimize
from vmc_amplitude import compute_local_energies, almost_true_relsigns
from vmc_vs_lbfgs import ScipyOptimizer
import fire

logger.remove()
logger.add(sys.stderr, level="INFO")


class LogProbDenseNet(nn.Module):
    def __init__(self, system: SpinSystem, n_hidden: int = 100, hidden_layers=1):
        super().__init__()
        self.system = system
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(system.number_spins, n_hidden), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(
            torch.from_numpy(
                make_unpacked_configurations(x, self.system.number_spins).astype(np.float32)
            )
        )


def main(task_id: int):
    system_specs = [
        (KagomeLattice(2, 3), 1),
        (KagomeLattice(2, 3), 0.5),
        (KagomeLattice(2, 4), 1),
        (KagomeLattice(2, 4), 0.5),
        (SquareLattice(6, 4), 0.5),
        (TriangleLattice(6, 4), 1.2),
    ]

    lattice, J2 = system_specs[task_id]

    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1,
        J2=J2,
        use_symmetries=False,
        spin_inversion=None,
        ground_state_cache_dir=Path("groundstates"),
    )
    print(f"System: {system.get_cache_id()}")

    energy, _ = system.get_eigenstates(1)
    log_prob_fn = LogProbDenseNet(system, n_hidden=512, hidden_layers=1)
    print(f"True energy: {energy[0]}")
    optimizer = ScipyOptimizer(
        system=system,
        log_prob_fn=log_prob_fn,
        relsigns_fn=almost_true_relsigns(system, eps=0.0),
        method="BFGS",
    )
    optimizer.optimize()


if __name__ == "__main__":
    fire.Fire(main)
