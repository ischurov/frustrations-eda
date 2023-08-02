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


def set_params(net: nn.Module, params: npt.NDArray):
    vector_to_parameters(torch.from_numpy(params.astype(np.float32)), net.parameters())


def get_params(net: nn.Module) -> npt.NDArray:
    return parameters_to_vector(net.parameters()).detach().numpy()


class ScipyOptimizer:
    def __init__(
        self,
        system: SpinSystem,
        log_prob_fn: torch.nn.Module,
        relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
        batch_size=64,
        method="BFGS",
    ):
        self.system = system
        self.log_prob_fn = log_prob_fn
        self.relsigns_fn = relsigns_fn
        self.batch_size = batch_size
        self.init_params = get_params(log_prob_fn)
        self.method = method

    def _compute_local_energies(
        self, flat_params: npt.NDArray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        set_params(self.log_prob_fn, flat_params)
        states, log_probs, _extra = sample_full(
            self.log_prob_fn,
            self.system.basis,
            SamplingOptions(
                number_samples=1,
                number_chains=1,
                mode="full",
                sweep_size=1,
                number_discarded=0,
            ),
        )
        states = states.view(-1)

        all_probs: torch.Tensor = _extra["weights"].view(-1)

        E = compute_local_energies(
            self.system.hamiltonian,
            states.detach().numpy(),
            relsigns_fn=self.relsigns_fn,
            log_prob_fn=lambda s: self.log_prob_fn(torch.from_numpy(s)).view(-1).detach().numpy(),
        )
        E = torch.from_numpy(E).to(torch.float32)

        return E, states, all_probs

    def objective(self, flat_params: npt.NDArray) -> float:
        E, _, all_probs = self._compute_local_energies(flat_params)
        full_energy = (E @ all_probs).item().real
        print(full_energy)
        return full_energy

    def gradient(self, flat_params) -> npt.NDArray:
        E, states, all_probs = self._compute_local_energies(flat_params)
        weights = all_probs
        with torch.no_grad():
            grad = 4 * (E - E @ weights) * weights
            # coeff 4 is due to: 2 from formula, 2 due to we are working with log probs
            # instead of log amplitudes

            grad = grad.view(-1, 1)
            grad_norm = torch.linalg.norm(grad)
            #    logger.info("‖∇E‖₂ = {}", grad_norm)
            # writer.add_scalar("loss/‖∇E‖₂", grad_norm, step)
            # writer.add_scalar("loss/E_variance", grad_norm / n_samples, step)

            # # Calculate full energy
            # if sampling_mode == "exact":
            #     E_full = E @ safe_exp(log_prob_fn(states).view(-1), normalise=True)
            #     writer.add_scalar("loss/E_full", E_full - torch.tensor(true_energy), step)

        self.log_prob_fn.zero_grad(set_to_none=True)

        forward_fn = self.log_prob_fn
        for states_chunk, grad_chunk in split_into_batches(
            (states.view(-1, 1), grad), self.batch_size
        ):
            output = forward_fn(states_chunk.view(-1))
            output.backward(grad_chunk)

        # extract gradient from output
        grad = torch.cat([p.grad.view(-1) for p in self.log_prob_fn.parameters()])
        return grad.detach().numpy()

    def optimize(self):
        result = minimize(
            self.objective,
            self.init_params,
            jac=self.gradient,
            method=self.method,
            options={"maxiter": 1000},
        )
        return result
