import io
import sys
import time
from collections import namedtuple
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import fire
import lattice_symmetries as ls
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
from spin_lattices import (
    ChainLattice,
    KagomeLattice,
    SpinLattice,
    SquareLattice,
    TriangularLattice,
)
from vmc_amplitude import LogProbDenseNet
from vmc_vs_lbfgs import AmplitudeOptimizer

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(task_id: int):
    system_specs = [
        (KagomeLattice(2, 3), 1),
        (KagomeLattice(2, 3), 0.5),
        (KagomeLattice(2, 4), 1),
        (KagomeLattice(2, 4), 0.5),
        (SquareLattice(6, 4), 0.5),
        (TriangularLattice(6, 4), 1.2),
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
