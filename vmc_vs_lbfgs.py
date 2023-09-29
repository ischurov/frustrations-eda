import io
import sys
import time
from collections import defaultdict, namedtuple
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from IPython.display import display
from loguru import logger
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
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
    AllToAllLattice,
    ChainLattice,
    KagomeLattice,
    SpinLattice,
    SquareLattice,
    TriangularLattice,
)
from vmc_amplitude import (
    almost_true_relsigns,
    compute_log_local_energies,
    true_relsigns,
)


def set_params(net: nn.Module, params: npt.NDArray):
    vector_to_parameters(torch.from_numpy(params.astype(np.float32)), net.parameters())


def get_params(net: nn.Module) -> npt.NDArray:
    return parameters_to_vector(net.parameters()).detach().numpy()


def differentiable_safe_exp(x: Tensor, normalise: bool = True) -> Tensor:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - torch.max(x)
    x = torch.exp(x)
    if normalise:
        x = x / torch.sum(x)
    return x


class AmplitudeOptimizer:
    def __init__(
        self,
        system: SpinSystem,
        log_prob_fn: torch.nn.Module,
        batch_size=8096,
        method="L-BFGS-B",
        maxiter=100,
        annealing_steps=0,
        initial_temp=1.0,
        tb_writer: SummaryWriter | None = None,
        plot_each=None,
        clip_grad_value=None,
        sign_noise_annealing_steps: int = 0,
        sign_noise_initial_eps: float = 0.2,
        full_spin_loss_weight: float = 0.0,
        full_energy_weight: float = 1.0,
    ):
        self.system = system
        energy, _ = system.get_eigenstates(1)
        self.true_energy = energy[0]
        self.annealing_steps = annealing_steps
        self.initial_temp = initial_temp
        self.plot_each = plot_each
        self.clip_grad_value = clip_grad_value
        self.sign_noise_annealing_steps = sign_noise_annealing_steps
        self.sign_noise_initial_eps = sign_noise_initial_eps
        self.full_spin_loss_weight = full_spin_loss_weight
        self.full_energy_weight = full_energy_weight

        self.full_spin_operator = HeisenbergJ1J2(
            AllToAllLattice(system.lattice),
            J1=1,
            use_symmetries=system.use_symmetries,
            spin_inversion=system.spin_inversion,
        ).hamiltonian

        self.log_prob_fn = log_prob_fn

        self.batch_size = batch_size
        self.init_params = get_params(log_prob_fn)
        self.method = method
        self.maxiter = maxiter
        self.nbd_matrix_w_signs = None
        self.nbd_states = None
        self.state_indices = None
        self.nbd_matrix_w_signs_fullspin = None
        self.nbd_states_fullspin = None
        self.state_indices_fullspin = None

        self.all_probs = None
        self.log_amplitudes = None
        self.true_amplitudes = np.abs(
            system.get_ground_state_coeffs(self.system.canonical_basis.states)
        )
        self.true_log_amplitudes = np.log(self.true_amplitudes)
        self.log_sorted_true_amplitudes = np.sort(self.true_log_amplitudes)[::-1]
        assert np.allclose((self.true_amplitudes**2).sum(), 1)
        self.true_ipr = (self.true_amplitudes**4).sum()

        self.full_energy = None
        self.tb_writer = tb_writer
        self.iteration = 0

    def relsigns_fn(self, states: np.ndarray) -> np.ndarray:
        if self.iteration < self.sign_noise_annealing_steps:
            eps = self.sign_noise_initial_eps * (
                1 - self.iteration / self.sign_noise_annealing_steps
            )
            return almost_true_relsigns(self.system, eps=eps)(states)
        else:
            return true_relsigns(self.system)(states)

    def _compute_log_local_energies(
        self,
        flat_params: npt.NDArray | None,
        states: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_E_loc: log of the local energies
            states: the states that were sampled
            log_probs: RESCALED (!!!) log of the probabilities of the sampled states
            probs: the probabilities of the sampled states
        """
        if states is None:
            assert log_probs is None and probs is None

        if flat_params is not None:
            set_params(self.log_prob_fn, flat_params)
        if states is None or log_probs is None or probs is None:
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
            probs = _extra["weights"].view(-1)

        assert probs is not None
        # assert states is not None and log_probs is not None and probs is not None
        our_hamiltonian = self.system.hamiltonian

        with torch.no_grad():
            (
                log_E_loc,
                extras,
            ) = compute_log_local_energies(
                our_hamiltonian,
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
            self.nbd_states = extras.nbd_states
            self.state_indices = extras.state_indices
            self.nbd_matrix_w_signs = extras.nbd_matrix_w_signs

            log_E_loc = torch.from_numpy(log_E_loc).to(torch.complex64)

        return log_E_loc, states, log_probs, probs

    def _compute_local_fullspin(
        self,
        flat_params: npt.NDArray | None,
        states: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_fullspin_loc: log of the local full spins
            states: the states that were sampled
            log_probs: RESCALED (!!!) log of the probabilities of the sampled states
            probs: the probabilities of the sampled states
        """
        if states is None:
            assert log_probs is None and probs is None

        if flat_params is not None:
            set_params(self.log_prob_fn, flat_params)
        if states is None or log_probs is None or probs is None:
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
            probs = _extra["weights"].view(-1)

        assert probs is not None
        # assert states is not None and log_probs is not None and probs is not None
        our_hamiltonian = self.full_spin_operator

        with torch.no_grad():
            (
                log_fullspin_loc,
                self.nbd_states_fullspin,
                _,
                self.state_indices_fullspin,
                _,
                self.nbd_matrix_w_signs_fullspin,
            ) = compute_log_local_energies(
                our_hamiltonian,
                states.detach().numpy(),
                relsigns_fn=self.relsigns_fn,
                log_prob_fn=lambda s: self.log_prob_fn(torch.from_numpy(s.astype(np.int64)))
                .view(-1)
                .detach()
                .numpy(),
                override_nbd_states=self.nbd_states_fullspin,
                override_nbd_matrix_w_signs=self.nbd_matrix_w_signs_fullspin,
                override_state_indices=self.state_indices_fullspin,
            )
            log_fullspin_loc = torch.from_numpy(log_fullspin_loc).to(torch.complex64)

        return log_fullspin_loc, states, log_probs, probs

    def objective(self, flat_params: np.ndarray | None = None) -> tuple[float, npt.NDArray]:
        log_E_loc, states, log_probs, all_probs = self._compute_log_local_energies(
            flat_params.astype(np.float32) if flat_params is not None else None
        )
        self.all_probs = all_probs.detach().numpy()
        self.log_amplitudes = log_probs * 0.5
        logger.info(f"{log_E_loc.real.min()=}, {log_E_loc.real.max()=}")
        full_energy = (torch.exp(log_E_loc).to(torch.float32) @ all_probs).item().real
        logger.info(f"{full_energy=}")

        self.full_energy = full_energy

        predicted_wavefunction = np.sign(
            self.system.get_ground_state_coeffs(self.system.canonical_basis.states)
        ) * np.sqrt(all_probs.detach().numpy())
        self.full_spin = predicted_wavefunction @ (
            self.full_spin_operator @ predicted_wavefunction
        )

        weights = all_probs
        if self.full_energy_weight > 0:
            with stopwatch("vmc_vs_lbfgs/gradient/grad"):
                with torch.no_grad():
                    weighted_E_loc = torch.exp(log_E_loc + np.log(weights)).real
                    grad = (
                        4
                        * self.full_energy_weight
                        * (weighted_E_loc - weighted_E_loc.sum() * weights)
                    )

                    self.grad_norm = torch.linalg.norm(grad)
                    logger.info(f"{grad[:10]=}")
                    logger.info(f"{log_E_loc[:10]=}")
        else:
            grad = torch.zeros_like(log_E_loc, dtype=torch.float32)
            self.grad_norm = 0

        if self.full_spin_loss_weight > 0:
            with torch.no_grad():
                log_spin_loc, _, _, _ = self._compute_local_fullspin(
                    flat_params.astype(np.float32) if flat_params is not None else None,
                    states=states,
                    log_probs=log_probs,
                    probs=all_probs,
                )
                logger.info(f"{log_spin_loc[:10]=}")
                weighted_spin_loc = torch.exp(log_spin_loc + np.log(weights)).real
                full_spin_grad = (
                    self.full_spin_loss_weight
                    * 4
                    * (weighted_spin_loc - weighted_spin_loc.sum() * weights)
                )
                if self.full_energy_weight > 0:
                    self.energy_full_spin_grad_overlap = (
                        (full_spin_grad @ grad)
                        / self.grad_norm
                        / torch.linalg.norm(full_spin_grad)
                    )
                logger.info(f"{full_spin_grad[:10]=}")
                grad += full_spin_grad

        self.log_prob_fn.zero_grad(set_to_none=True)
        forward_fn = self.log_prob_fn
        for states_chunk, grad_chunk in split_into_batches(
            (states.view(-1, 1), grad.view(-1, 1)), self.batch_size
        ):
            with stopwatch("vmc_vs_lbfgs/gradient/forward"):
                output = forward_fn(states_chunk.view(-1))
            with stopwatch("vmc_vs_lbfgs/gradient/backward"):
                output.backward(grad_chunk, retain_graph=True)

            if self.iteration < self.annealing_steps:
                temp = self.initial_temp * (1 - self.iteration / self.annealing_steps)
                probs = differentiable_safe_exp(output, normalise=True)
                entropy = -torch.sum(probs * torch.log(probs))
                entropy_loss = -temp * entropy
                entropy_loss.backward()
        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.log_prob_fn.parameters(),
                self.clip_grad_value,
            )
        with stopwatch("vmc_vs_lbfgs/gradient/extract_grad"):
            # extract gradient from output
            full_grad = torch.cat([p.grad.view(-1) for p in self.log_prob_fn.parameters()])
            self.full_grad_L2_norm = torch.linalg.norm(full_grad)
            self.full_grad_Linf_norm = torch.max(torch.abs(full_grad))

        return (
            self.full_energy_weight * full_energy + self.full_spin_loss_weight * self.full_spin,
            full_grad.detach().numpy().astype(np.float64),
        )

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
        self.tb_writer.add_scalar("optimize/grad_norm", self.grad_norm, self.iteration)
        self.tb_writer.add_scalar(
            "optimize/full_grad_L2_norm", self.full_grad_L2_norm, self.iteration
        )
        self.tb_writer.add_scalar(
            "optimize/full_grad_Linf_norm", self.full_grad_Linf_norm, self.iteration
        )
        self.tb_writer.add_scalar("optimize/full_spin", self.full_spin, self.iteration)
        if self.full_energy_weight > 0 and self.full_spin_loss_weight > 0:
            self.tb_writer.add_scalar(
                "optimize/energy_full_spin_grad_overlap",
                self.energy_full_spin_grad_overlap,
                self.iteration,
            )

        self.iteration += 1

        if self.plot_each is not None and self.iteration % self.plot_each == 0:
            fig = plt.figure()
            sorted_predictions = self.log_amplitudes[(-self.log_amplitudes).argsort()]
            plt.plot(
                sorted_predictions - sorted_predictions[0] + self.log_sorted_true_amplitudes[0],
                label="predicted log amplitudes",
            )
            plt.plot(
                self.log_sorted_true_amplitudes,
                label="true log ampitudes",
            )
            plt.title(f"Iteration {self.iteration}")
            plt.legend()
            display(fig)
            plt.close(fig)

    def optimize_scipy(self):
        self.iteration = 0
        result = minimize(
            self.objective,
            self.init_params,
            jac=True,
            method=self.method,
            callback=lambda _: self.write_tb(),
            options={"maxiter": self.maxiter},
        )
        return result
