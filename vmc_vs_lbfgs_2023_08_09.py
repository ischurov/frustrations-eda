import sys
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import fire
import lattice_symmetries as ls
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from kagome_cnn import KagomeCNNRegression
from misc_utils import make_unpacked_configurations
from my_stopwatch import stopwatch
from nqs_playground_helpers import (
    SamplingOptions,
    forward_with_batches,
    safe_exp,
    sample_exactly,
    sample_full,
    split_into_batches,
)
from slater_determinant import Initializer, SlaterDeterminant
from spin_lattices import (
    ChainLattice,
    KagomeLattice,
    SpinLattice,
    SquareLattice,
    TriangularLattice,
)
from vmc_amplitude import almost_true_relsigns
from vmc_vs_lbfgs import AmplitudeOptimizer
from vmc_vs_lbfgs_2023_08_02 import LogProbDenseNet

# from misc_utils import torch_overlap as overlap

logger.remove()
logger.add(sys.stderr, level="INFO")

# @torch.no_grad()
# def evaluate(system: SpinSystem, model: nn.Module):
#     eval_set = torch.from_numpy(system.canonical_basis.states.astype(np.int64))
#     log_probs = model(eval_set).view(-1)
#     amplitudes = torch.exp(log_probs * 0.5)
#     true_amplitudes = torch.from_numpy(
#         np.abs(system.get_ground_state_coeffs(eval_set.detach().numpy())).astype(
#             np.float32
#         )
#     )
#     return overlap(amplitudes, true_amplitudes)


def main(task_id: int):
    n_hidden = 512
    stopwatch.reset()
    lattice = KagomeLattice(2, 3)
    J2 = 1.0

    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1,
        J2=J2,
        use_symmetries=False,
        spin_inversion=None,
        ground_state_cache_dir=Path("groundstates"),
    )
    logger.info(f"System: {system.get_cache_id()}")

    energy, _ = system.get_eigenstates(1)
    log_prob_fn = LogProbDenseNet(system, n_hidden=n_hidden, hidden_layers=1)

    logger.info(f"True energy: {energy[0]}")

    writer = SummaryWriter(
        log_dir=(
            f"experiments/{datetime.now().strftime('%Y_%m_%d')}_n_hidden/{n_hidden=}_{datetime.now().strftime('%H_%M_%S')}"
        )
    )

    optimizer = AmplitudeOptimizer(
        system=system,
        log_prob_fn=log_prob_fn,
        method="L-BFGS-B",
        maxiter=300,
        batch_size=8096,
        tb_writer=writer,
        full_spin_loss_weight=0.0,
        full_energy_weight=1.0,
        # clip_grad_value=0.02,
        # clip_grad_norm=0.02,
        # clip_grad_norm_type='inf',
        # annealing_steps=0, #240,
        # initial_temp=2,
        # sign_noise_annealing_steps=10,
        # sign_noise_initial_eps=0.2,
    )
    try:
        r = optimizer.optimize_scipy()
        logger.info(str(r))
    finally:
        logger.info(str(stopwatch))


if __name__ == "__main__":
    fire.Fire(main)
