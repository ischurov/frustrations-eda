from enum import unique
import itertools
from datetime import datetime
import os
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from spin_systems import (
    ground_state_basis,
    heisenberg,
    SpinSystem,
    spin_system,
    no_symmetries_basis,
    zero_sector_basis,
    LatticeExpr,
)
from typing import Callable
from misc_utils import differentiable_safe_exp
from misc_utils import torch_overlap as find_overlap, get_git_revision_hash
from my_stopwatch import Stopwatch, stopwatch
from nqs_playground_helpers import (
    SamplingOptions,
    forward_with_batches,
    safe_exp,
    sample_exactly,
    sample_full,
    split_into_batches,
)
from spin_lattices import KagomeLattice, ParallelogramSpinLattice
from vmc_amplitude import (
    LogProbDenseNetPairwiseXor,
    almost_true_relsigns,
    compute_log_local_energies,
    get_csr_hamiltonian,
)
from dilated_nns_xors import resolve_config_inheritance
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from typing import Any
import torch
from torch import nn
import jsonlines
from conv2d_circular import InvariantSpinCNNRegression
from vmc_2024_02_28 import get_network, get_device, get_eval_set
from ising_sign_reconstruction import find_sign_overlap, reconstruct_signs, custom_signs
from nqs_playground_helpers import forward_with_batches
import lattice_symmetries as ls
from spin_lattices import AllToAllLattice
from vmc_importance_config import get_config
from vmc_ising import get_basis, get_system

self_name = Path(__file__).stem
git_hash = get_git_revision_hash()

output_dir = Path("experiments") / self_name


def main(task_id: int):
    config = get_config(task_id)
    output_dir_task = output_dir / str(task_id)
    output_dir_task.mkdir(parents=True, exist_ok=True)

    logger.add(output_dir_task / "log.log", backtrace=True, diagnose=True)
    device = get_device(config)
    logger.debug(f"Torch will use device: {device}")
    # lattice = TriangleLattice(6, 4)
    lattice = get_lattice(config["lattice"])
    system = get_system(config)
    true_energy = system.ground_energy
    for run in range(config["runs"]):
        do_run(
            config=config,
            device=device,
            output_dir_task=output_dir_task,
            run=run,
            system=system,
            true_energy=true_energy,
        )


def do_run(config, device, output_dir_task, run, system, true_energy):
    eval_set = get_eval_set(system, config["eval_set_max_size"], canonical_basis=False)

    log_prob_fn = get_network(config, system)
    log_prob_fn.to(device)

    optimizer = torch.optim.Adam(
        log_prob_fn.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    true_amplitudes = torch.from_numpy(
        np.abs(system.get_ground_state_coeffs(eval_set, apply_symmetries=False)).astype(
            np.float32
        )
    ).to(device)

    start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    relsigns_fn = almost_true_relsigns(
        system, eps=config["sign_noise"], apply_symmetries=False
    )
    for outer_step in range(config["outer_sample_size"]):
        do_outer_step(
            config=config,
            device=device,
            eval_set=eval_set,
            log_prob_fn=log_prob_fn,
            optimizer=optimizer,
            output_dir_task=output_dir_task,
            relsigns_fn=relsigns_fn,
            run=run,
            step=outer_step,
            system=system,
            true_amplitudes=true_amplitudes,
            true_energy=true_energy,
        )


### BASED ON: https://stackoverflow.com/a/72005790/3025981
def unique_with_first_indices(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_values, indices, counts = torch.unique(
        values, return_inverse=True, return_counts=True
    )

    _, ind_sorted = torch.sort(indices, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return unique_values, indices, counts, first_indicies


### END BASED


def do_outer_step(
    config,
    device,
    eval_set,
    log_prob_fn,
    optimizer,
    output_dir_task,
    relsigns_fn,
    run,
    step,
    system,
    true_amplitudes,
    true_energy,
):

    initial_states, log_probs, all_probs = sample_exactly(
        log_prob_fn,
        system.basis,
        SamplingOptions(
            number_samples=config["outer_sample_size"],
            number_chains=1,
            mode="exact",
            sweep_size=1,
            number_discarded=0,
            device=device,
        ),
        return_all_probs=True,
    )
    initial_states = initial_states.to(device)
    # states, weights = torch.unique(states.view(-1), return_counts=True)
    # weights = weights.float() / torch.sum(weights)

    initial_log_probs = log_probs.to(device)

    if config["outer_sample_size"] % config["inner_sample_size"] != 0:
        raise ValueError("outer_sample_size must be divisible by inner_sample_size")

    inner_samples = config["outer_sample_size"] // config["inner_sample_size"]

    for epoch in range(config["inner_epochs"]):
        shuffle_order = torch.randperm(config["outer_sample_size"])
        for i_sample in range(inner_samples):
            do_inner_step(
                epoch=epoch,
                inner_samples=inner_samples,
                i_sample=i_sample,
                shuffle_order=shuffle_order,
                config=config,
                device=device,
                initial_states=initial_states,
                initial_log_probs=initial_log_probs,
                log_prob_fn=log_prob_fn,
                optimizer=optimizer,
                relsigns_fn=relsigns_fn,
                system=system,
            )


def do_inner_step(
    epoch,
    inner_samples,
    i_sample,
    shuffle_order,
    config,
    device,
    initial_states,
    initial_log_probs,
    log_prob_fn,
    optimizer,
    relsigns_fn,
    system,
):
    inner_step = epoch * inner_samples + i_sample
    inner_start = i_sample * config["inner_sample_size"]
    inner_end = (i_sample + 1) * config["inner_sample_size"]
    inner_states_all = initial_states[shuffle_order[inner_start:inner_end]]
    inner_log_probs_all = initial_log_probs[shuffle_order[inner_start:inner_end]]

    inner_states, indices, counts, first_indicies = unique_with_first_indices(
        inner_states_all.view(-1)
    )
    inner_weights = counts.float() / torch.sum(counts)
    inner_log_probs = inner_log_probs_all[first_indicies]

    log_E_loc, *_ = compute_log_local_energies(
        system.hamiltonian,
        inner_states.detach().numpy(),
        relsigns_fn=relsigns_fn,
        log_prob_fn=lambda s: forward_with_batches(
            log_prob_fn,
            torch.from_numpy(s.astype(np.int64)).to(device),
            batch_size=config["batch_size"],
        )
        .view(-1)
        .detach()
        .numpy(),
    )
    with torch.no_grad():
        log_E_loc = torch.from_numpy(log_E_loc).to(device).to(torch.complex64)
        new_log_probs = log_prob_fn(inner_states).view(-1).to(torch.float64)
        weights = safe_exp(
            torch.log(inner_weights) + new_log_probs - inner_log_probs
        ).to(torch.float32)

        weighted_E_loc = torch.exp(log_E_loc + torch.log(weights)).real
        grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

        grad = grad.view(-1, 1)
        grad_norm = torch.linalg.norm(grad)

        E = torch.exp(log_E_loc).real
        E_full = E @ safe_exp(new_log_probs.to(torch.float32).view(-1), normalise=True)

    optimizer.zero_grad()

    for states_chunk, grad_chunk in split_into_batches(
        (inner_states.view(-1, 1), grad), config["batch_size"]
    ):
        output = log_prob_fn(states_chunk.view(-1))
        output.backward(grad_chunk, retain_graph=False)

        # if step < annealing_steps:
        #     temp = initial_temp * (1 - step / annealing_steps)
        #     probs = differentiable_safe_exp(output, normalise=True)
        #     entropy = -torch.sum(probs * torch.log(probs))
        #     entropy_loss = -temp * entropy
        #     entropy_loss.backward()

    # full_gradient_norm = get_gradient_norm(forward_fn.parameters())
    # writer.add_scalar("loss/full_gradient_norm", full_gradient_norm, step)

    optimizer.step()

    # TODO: evaluation and reporting
