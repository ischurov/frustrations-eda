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
from vmc_amplitude import compute_log_local_energies, almost_true_relsigns, true_relsigns
from collections import defaultdict
from my_stopwatch import stopwatch


def set_params(net: nn.Module, params: npt.NDArray):
    vector_to_parameters(torch.from_numpy(params.astype(np.float32)), net.parameters())


def get_params(net: nn.Module) -> npt.NDArray:
    return parameters_to_vector(net.parameters()).detach().numpy()


class ScipyOptimizer:
    def __init__(
        self,
        system: SpinSystem,
        log_prob_fn: torch.nn.Module,
        batch_size=8096,
        method="BFGS",
        maxiter=100,
        tb_writer: SummaryWriter | None = None,
    ):
        self.system = system
        energy, _ = system.get_eigenstates(1)
        self.true_energy = energy[0]

        self.log_prob_fn = log_prob_fn
        self.relsigns_fn = true_relsigns(system)
        self.batch_size = batch_size
        self.init_params = get_params(log_prob_fn)
        self.method = method
        self.maxiter = maxiter
        self.nbd_matrix_w_signs = None
        self.nbd_states = None
        self.state_indices = None
        self.all_probs = None
        self.true_amplitudes = np.abs(
            system.get_ground_state_coeffs(self.system.canonical_basis.states)
        )
        assert np.allclose((self.true_amplitudes**2).sum(), 1)
        self.true_ipr = (self.true_amplitudes**4).sum()

        self.full_energy = None
        self.tb_writer = tb_writer
        self.iteration = 0

    def _compute_log_local_energies(
        self, flat_params: npt.NDArray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_E_loc: log of the local energies
            states: the states that were sampled
            log_probs: RESCALED (!!!) log of the probabilities of the sampled states
            probs: the probabilities of the sampled states
        """

        set_params(self.log_prob_fn, flat_params)
        with stopwatch("vmc_vs_lbfgs/_compute_local_energies/sample_full"):
            states, log_probs, _extra = sample_full(
                self.log_prob_fn,
                self.system.canonical_basis,
                SamplingOptions(
                    number_samples=1,
                    number_chains=1,
                    mode="full",
                    sweep_size=1,
                    number_discarded=0,
                ),
            )
        log_probs = log_probs.view(-1)
        states = states.view(-1)
        probs: torch.Tensor = _extra["weights"].view(-1)

        with torch.no_grad():
            (
                log_E_loc,
                self.nbd_states,
                self.state_indices,
                _,
                self.nbd_matrix_w_signs,
            ) = compute_log_local_energies(
                self.system.hamiltonian,
                states.detach().numpy(),
                relsigns_fn=self.relsigns_fn,
                log_prob_fn=lambda s: self.log_prob_fn(torch.from_numpy(s.astype(np.int64)))
                .view(-1)
                .detach()
                .numpy(),
                override_nbd_states=self.nbd_states,
                override_nbd_matrix_w_signs=self.nbd_matrix_w_signs,
                override_state_indices=self.state_indices,
            )
            log_E_loc = torch.from_numpy(log_E_loc).to(torch.complex64)
        return log_E_loc, states, log_probs, probs

    def objective(self, flat_params) -> tuple[float, npt.NDArray]:
        log_E_loc, states, _, all_probs = self._compute_log_local_energies(
            flat_params.astype(np.float32)
        )
        self.all_probs = all_probs.detach().numpy()

        full_energy = (torch.exp(log_E_loc).to(torch.float32) @ all_probs).item().real
        logger.info(full_energy)
        self.full_energy = full_energy

        weights = all_probs
        with stopwatch("vmc_vs_lbfgs/gradient/grad"):
            with torch.no_grad():
                weighted_E_loc = torch.exp(log_E_loc + np.log(weights)).real
                grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

                # grad_norm = torch.linalg.norm(grad)
        self.log_prob_fn.zero_grad(set_to_none=True)
        forward_fn = self.log_prob_fn
        for states_chunk, grad_chunk in split_into_batches(
            (states.view(-1, 1), grad.view(-1, 1)), self.batch_size
        ):
            with stopwatch("vmc_vs_lbfgs/gradient/forward"):
                output = forward_fn(states_chunk.view(-1))
            with stopwatch("vmc_vs_lbfgs/gradient/backward"):
                output.backward(grad_chunk)
        with stopwatch("vmc_vs_lbfgs/gradient/extract_grad"):
            # extract gradient from output
            grad = torch.cat([p.grad.view(-1) for p in self.log_prob_fn.parameters()])
        return full_energy, grad.detach().numpy().astype(np.float64)

    def extract_overlap(self) -> float:
        if self.all_probs is None:
            raise RuntimeError("Must call objective first")
        assert np.allclose(self.all_probs.sum(), 1)

        predicted_amplitudes = np.sqrt(self.all_probs)
        return predicted_amplitudes @ self.true_amplitudes

    def write_tb(self):
        if self.tb_writer is None or self.full_energy is None or self.all_probs is None:
            return

        energy_delta = self.full_energy - self.true_energy
        self.tb_writer.add_scalar("optimize/energy_delta", energy_delta, self.iteration)
        self.tb_writer.add_scalar(
            "optimize/rel_energy_delta", energy_delta / self.true_energy, self.iteration
        )
        ipr_delta = (self.all_probs**2).sum() - self.true_ipr
        self.tb_writer.add_scalar("optimize/ipr_delta", ipr_delta, self.iteration)
        self.tb_writer.add_scalar(
            "optimize/rel_ipr_delta",
            ipr_delta / self.true_ipr,
            self.iteration,
        )

        self.tb_writer.add_scalar("optimize/overlap", self.extract_overlap(), self.iteration)
        self.iteration += 1

    def optimize(self):
        self.iteration = 0
        result = minimize(
            self.objective,
            self.init_params,
            jac=True,
            method=self.method,
            callback=lambda x: self.write_tb(),
            options={"maxiter": self.maxiter},
        )
        return result
