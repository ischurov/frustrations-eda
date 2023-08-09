from spin_lattices import (
    KagomeLattice,
    SpinLattice,
    ChainLattice,
    SquareLattice,
    TriangleLattice,
)
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
from vmc_amplitude import almost_true_relsigns
from vmc_vs_lbfgs import AmplitudeOptimizer
import fire
from my_stopwatch import stopwatch
from vmc_vs_lbfgs_2023_08_02 import LogProbDenseNet
from slater_determinant import SlaterDeterminant, Initializer
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from vmc_amplitude import LogProbDenseNetPairwiseXor
import itertools

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


def main():
    n_hidden = 512
    stopwatch.reset()
    lattice = KagomeLattice(2, 4)
    J2 = 1.0
    maxiter = 10000

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
    pairs = tuple(
        map(np.array, zip(*itertools.combinations(range(system.number_spins), 2)))
    )
    log_prob_fn = LogProbDenseNetPairwiseXor(
        system, n_hidden=n_hidden, hidden_layers=1, xor_pairs=pairs
    )

    logger.info(f"True energy: {energy[0]}")

    writer = SummaryWriter(
        log_dir=(
            f"experiments/{datetime.now().strftime('%Y_%m_%d')}_kagome24/{datetime.now().strftime('%H_%M_%S')}"
        )
    )

    optimizer = AmplitudeOptimizer(
        system=system,
        log_prob_fn=log_prob_fn,
        method="L-BFGS-B",
        maxiter=maxiter,
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
