import itertools
import time
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from spin_systems import HeisenbergJ1J2
from misc_utils import differentiable_safe_exp
from misc_utils import torch_overlap as find_overlap
from my_stopwatch import stopwatch
from nqs_playground_helpers import (
    SamplingOptions,
    safe_exp,
    sample_exactly,
    sample_full,
    split_into_batches,
)
from spin_lattices import KagomeLattice
from vmc_amplitude import (
    LogProbDenseNetPairwiseXor,
    almost_true_relsigns,
    compute_log_local_energies,
)


def main(task_id: int):
    stopwatch.reset()

    # n_samples = 48620
    n_samples = [100, 1000, 10000, 100_000][(task_id // 2) % 4]
    lr = [1e-2, 1e-3][task_id % 2]
    momentum = 0.0
    batch_size = 8092
    sign_noise = 0.0
    weight_decay = 0
    annealing_steps = 0  # 400
    initial_temp = 3
    max_iter = 10000

    sampling_mode = "exact"

    # lattice = TriangleLattice(6, 4)
    lattice = KagomeLattice(2, 3)
    # lattice = ChainLattice(10)
    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1,
        J2=1,
        ground_state_cache_dir=Path("groundstates"),
        use_symmetries=False,
        spin_inversion=None,
    )
    true_energy, _ = system.get_eigenstates(1)
    true_energy = true_energy[0]

    if len(system.canonical_basis.states) > 50000:
        eval_set = np.random.choice(system.canonical_basis.states, 50000, replace=False)
    else:
        eval_set = system.canonical_basis.states

    pairs = tuple(map(np.array, zip(*itertools.combinations(range(system.number_spins), 2))))
    # pairs = tuple(map(np.array, zip(*system.lattice.edges_to_kind.keys())))

    log_prob_fn = LogProbDenseNetPairwiseXor(
        system, n_hidden=512, hidden_layers=1, xor_pairs=pairs
    )
    # log_prob_fn = SlaterAbs(
    #     system.lattice,
    #     system.canonical_basis,
    #     sign_cache_dir=Path("signs_cache"),
    #     initialization="randn",
    # )

    # log_prob_fn = KagomeCNNRegression(system.lattice, hidden_channels1=32, hidden_channels2=64)
    # optimizer = torch.optim.SGD(log_prob_fn.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr, weight_decay=weight_decay)

    true_amplitudes = torch.from_numpy(
        np.abs(system.get_ground_state_coeffs(eval_set)).astype(np.float32)
    )
    relsigns_fn = almost_true_relsigns(system, eps=sign_noise)

    writer = SummaryWriter(
        log_dir=(
            f"experiments/{datetime.now().strftime('%Y_%m_%d')}_vmc_xor/"
            f"{n_samples=}_{lr=}_{datetime.now().strftime('%H_%M_%S')}"
        )
    )

    sampling_time = 0
    local_energies_time = 0
    optimization_time = 0
    evaluation_time = 0

    for step in range(max_iter):
        sampling_time_tick = time.time()
        if sampling_mode == "exact":
            states, log_probs, all_probs = sample_exactly(
                log_prob_fn,
                system.basis,
                SamplingOptions(
                    number_samples=n_samples,
                    number_chains=1,
                    mode="exact",
                    sweep_size=1,
                    number_discarded=0,
                ),
                return_all_probs=True,
            )
            states, weights = torch.unique(states.view(-1), return_counts=True)
            weights = weights.float() / torch.sum(weights)

        elif sampling_mode == "full":
            states, log_probs, _extra = sample_full(
                log_prob_fn,
                system.basis,
                SamplingOptions(
                    number_samples=1,
                    number_chains=1,
                    mode="full",
                    sweep_size=1,
                    number_discarded=0,
                ),
            )
            states = states.view(-1)
            weights = _extra["weights"].view(-1)
            all_probs = weights
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        sampling_time += time.time() - sampling_time_tick

        ipr = torch.sum(all_probs**2)
        writer.add_scalar("loss/ipr", ipr, step)

        local_energies_time_tick = time.time()
        log_E_loc, *_ = compute_log_local_energies(
            system.hamiltonian,
            states.detach().numpy(),
            relsigns_fn=relsigns_fn,
            log_prob_fn=lambda s: log_prob_fn(torch.from_numpy(s.astype(np.int64)))
            .view(-1)
            .detach()
            .numpy(),
        )
        log_E_loc = torch.from_numpy(log_E_loc).to(torch.complex64)
        local_energies_time += time.time() - local_energies_time_tick

        # states = states.view(-1, states.size(-1))
        # log_probs = log_probs.view(-1)
        # weights = weights.view(-1)

        # Compute output gradient
        sampling_time_tick = time.time()

        with torch.no_grad():
            weighted_E_loc = torch.exp(log_E_loc + np.log(weights)).real
            grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

            # coeff 4 is due to: 2 from formula, 2 due to we are working with log probs
            # instead of log amplitudes

            # grad = 4 * (E - E @ weights) * weights

            grad = grad.view(-1, 1)
            grad_norm = torch.linalg.norm(grad)
            #    logger.info("‖∇E‖₂ = {}", grad_norm)
            writer.add_scalar("loss/‖∇E‖₂", grad_norm, step)
            writer.add_scalar("loss/E_variance", grad_norm / n_samples, step)

            # Calculate full energy
            # if sampling_mode == "exact":
            E = torch.exp(log_E_loc).real
            E_full = E @ safe_exp(log_prob_fn(states).view(-1), normalise=True)
            writer.add_scalar("loss/E_full_delta", E_full - torch.tensor(true_energy), step)
            logger.info("E_full_delta = {}", E_full)

        optimizer.zero_grad()
        # batch_size = self.config.inference_batch_size

        # Computing gradients for the amplitude network
        # logger.info("Computing gradients...")
        # if _should_optimize(self.config.amplitude):
        #     self.config.amplitude.train()
        forward_fn = log_prob_fn
        for states_chunk, grad_chunk in split_into_batches((states.view(-1, 1), grad), batch_size):
            output = forward_fn(states_chunk.view(-1))
            output.backward(grad_chunk, retain_graph=True)

            if step < annealing_steps:
                temp = initial_temp * (1 - step / annealing_steps)
                probs = differentiable_safe_exp(output, normalise=True)
                entropy = -torch.sum(probs * torch.log(probs))
                entropy_loss = -temp * entropy
                entropy_loss.backward()

        # full_gradient_norm = get_gradient_norm(forward_fn.parameters())
        # writer.add_scalar("loss/full_gradient_norm", full_gradient_norm, step)

        optimizer.step()
        optimization_time += time.time() - sampling_time_tick

        evaluation_time_tick = time.time()
        predicted_amplitudes = safe_exp(log_prob_fn(eval_set) * 0.5)

        overlap = find_overlap(true_amplitudes, predicted_amplitudes.view(-1))
        writer.add_scalar("overlap", overlap, step)
        logger.info(f"{step}: overlap = {overlap:.3f}, ‖∇E‖₂ = {grad_norm:.3f}")
        evaluation_time += time.time() - evaluation_time_tick


if __name__ == "__main__":
    fire.Fire(main)
