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
from vmc_vs_lbfgs import AmplitudeOptimizer
import fire
from my_stopwatch import stopwatch
from vmc_amplitude import LogProbDenseNet

logger.remove()
logger.add(sys.stderr, level="INFO")


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
    logger.info(f"System: {system.get_cache_id()}")

    energy, _ = system.get_eigenstates(1)
    log_prob_fn = LogProbDenseNet(system, n_hidden=512, hidden_layers=1)
    logger.info(f"True energy: {energy[0]}")
    optimizer = AmplitudeOptimizer(
        system=system,
        log_prob_fn=log_prob_fn,
        method="L-BFGS-B",
        maxiter=100000,
        batch_size=8096,
    )
    try:
        r = optimizer.optimize()
        print(r)
    finally:
        logger.info(str(stopwatch))


if __name__ == "__main__":
    fire.Fire(main)
