import itertools
from datetime import datetime
import os
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from misc_utils import differentiable_safe_exp
from misc_utils import torch_overlap as find_overlap, get_git_revision_hash
from my_stopwatch import Stopwatch, stopwatch
from nqs_playground_helpers import (
    SamplingOptions,
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

self_name = Path(__file__).stem
git_hash = get_git_revision_hash()

output_dir = Path("experiments") / self_name
default_config = {
    "n_samples": 10000,
    "lr": 1e-3,
    "batch_size": 10000,
    "weight_decay": 0,
    "max_iter": 15000,
    "lattice": "kagome2x3",
    "J2": 1,
    "use_symmetries": False,
    "spin_inversion": None,
    "eval_set_max_size": 50000,
    "device": "auto",
    "random_seed": None,
    "dilations": None,
    "hidden_channels": None,
    "warm_up_overlap": 0.7,
    "sign_update_period": 100,
    "sign_reconstruction.method": "annealing",
    "sign_reconstruction.number_sweeps": 100,
    "sign_reconstruction.repetitions": 18,
    "warm_up.sign_noise": 0.0,
    "checkpoint_log_prob_fn_each": None,
    "checkpoint_log_prob_fn_on_sign_update": False,
    "checkpoint_signs": False,
    "checkpoint_signs_greedy": False,
}

configs = {
    0: {
        "log_prob_fn": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 2,
        "warm_up_overlap": 0.4,
        "sign_update_period": 5,
    },
    1: {"_inherit": 0, "warm_up_overlap": 0.95, "sign_update_period": 10},
    2: {"_inherit": 1, "sign_update_period": 100},
    3: {"_inherit": 1, "sign_update_period": 500},
    4: {"_inherit": 0, "warm_up_overlap": 0.9, "sign_update_period": 10},
    5: {"_inherit": 4, "sign_update_period": 100},
    6: {"_inherit": 4, "sign_update_period": 500},
    7: {"_inherit": 0, "warm_up_overlap": 0.8, "sign_update_period": 10},
    8: {"_inherit": 7, "sign_update_period": 100},
    9: {"_inherit": 7, "sign_update_period": 500},
    10: {"_inherit": 0, "warm_up_overlap": 0.7, "sign_update_period": 10},
    11: {"_inherit": 10, "sign_update_period": 100},
    12: {"_inherit": 10, "sign_update_period": 500},
    13: {"_inherit": 0, "warm_up_overlap": 0.6, "sign_update_period": 10},
    14: {"_inherit": 13, "sign_update_period": 100},
    15: {"_inherit": 13, "sign_update_period": 500},
    16: {
        "_inherit": 0,
        "warm_up_overlap": 0.7,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 1000,
    },
    17: {
        "_inherit": 0,
        "warm_up_overlap": 0.7,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 5000,
    },
    18: {
        "_inherit": 0,
        "warm_up_overlap": 0.7,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 10000,
    },
    19: {
        "_inherit": 0,
        "warm_up_overlap": 0.8,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 1000,
    },
    20: {
        "_inherit": 0,
        "warm_up_overlap": 0.8,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 5000,
    },
    21: {
        "_inherit": 0,
        "warm_up_overlap": 0.8,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 10000,
    },
    22: {
        "_inherit": 0,
        "warm_up_overlap": 0.9,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 1000,
    },
    23: {
        "_inherit": 0,
        "warm_up_overlap": 0.9,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 5000,
    },
    24: {
        "_inherit": 0,
        "warm_up_overlap": 0.9,
        "sign_update_period": 100,
        "sign_reconstruction.repetitions": 128,
        "sign_reconstruction.number_sweeps": 10000,
    },
    25: {
        "_inherit": 0,
        "warm_up_overlap": 0.7,
        "sign_update_period": 100,
        "checkpoint_log_prob_fn_each": 100,
    },
    26: {
        "_inherit": 0,
        "warm_up_overlap": 0.3,
        "sign_update_period": 10000,
        "max_iter": 200000,
    },
    27: {
        "_inherit": 26,
        "warm_up_overlap": 0.6,
    },
    28: {"_inherit": 15, "sign_reconstruction.repetitions": 16},
    29: {"_inherit": 27, "sign_update_period": 5000},
    30: {"_inherit": 27, "sign_update_period": 2000},
    31: {"_inherit": 27, "sign_update_period": 1000},
    32: {"_inherit": 27, "sign_update_period": 500},
    33: {"_inherit": 26, "warm_up_overlap": 0.7},
    34: {"_inherit": 33, "sign_update_period": 5000},
    35: {"_inherit": 33, "sign_update_period": 2000},
    36: {"_inherit": 33, "sign_update_period": 1000},
    37: {"_inherit": 33, "sign_update_period": 500},
    38: {
        "_inherit": 27,
        "checkpoint_log_prob_fn_on_sign_update": True,
        "checkpoint_signs": True,
        "sign_reconstruction.method": "greedy_solve",
    },
    39: {
        "_inherit": 27,
        "checkpoint_log_prob_fn_on_sign_update": True,
        "checkpoint_signs": True,
        "checkpoint_signs_greedy": True,
    },
    40: {
        "_inherit": 38,
        "warm_up_overlap": 0.7,
    },
    41: {
        "_inherit": 38,
        "warm_up_overlap": 0.8,
    },
    42: {
        "_inherit": 38,
        "warm_up_overlap": 0.9,
    },
    43: {
        "_inherit": 38,
        "warm_up_overlap": 0.7,
        "sign_update_period": 5000,
    },
    44: {
        "_inherit": 43,
        "sign_update_period": 2000,
    },
    45: {
        "_inherit": 43,
        "sign_update_period": 1000,
    },
    46: {
        "_inherit": 43,
        "sign_update_period": 500,
    },
    47: {
        "_inherit": 43,
        "sign_update_period": 100,
    },
    48: {
        "_inherit": 38,
        "warm_up_overlap": 0.7,
    },
    49: {
        "_inherit": 38,
        "warm_up_overlap": 0.8,
    },
    50: {
        "_inherit": 0,
        "lattice": "kagome2x4",
        "sign_reconstruction.method": "greedy_solve",
        "warm_up_overlap": 0.1,
    },
    51: {
        "_inherit": 0,
        "lattice": "kagome2x4",
        "sign_reconstruction.method": "greedy_solve",
        "warm_up_overlap": 0.7,
        "sign_update_period": 10000,
        "checkpoint_log_prob_fn_on_sign_update": True,
        "max_iter": 1000,
        "checkpoint_signs": True,
    },
    52: {
        "_inherit": 51,
        "warm_up_overlap": 0.8,
        "max_iter": 200000,
    },
    53: {
        "_inherit": 52,
        "warm_up_overlap": 0.9,
    },
    53: {
        "_inherit": 52,
        "warm_up_overlap": 0.95,
    },
}


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)


def get_system(config):
    lattice = get_lattice(config["lattice"])
    return HeisenbergJ1J2(
        lattice=lattice,
        J1=1,
        J2=config["J2"],
        ground_state_cache_dir=Path("groundstates"),
        use_symmetries=config["use_symmetries"],
        spin_inversion=config["spin_inversion"],
    )


def main(task_id: int):
    stopwatch.reset()
    local_sw = Stopwatch()
    config = get_config(task_id)

    n_samples = config["n_samples"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    sign_noise = config["warm_up.sign_noise"]
    weight_decay = config["weight_decay"]
    max_iter = config["max_iter"]
    warm_up_overlap = config["warm_up_overlap"]

    output_dir_task = output_dir / str(task_id)

    if config["random_seed"] is not None:
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.use_deterministic_algorithms(True)

    output_dir_task.mkdir(parents=True, exist_ok=True)

    device = get_device(config)

    logger.debug(f"Torch will use device: {device}")
    # lattice = TriangleLattice(6, 4)
    lattice = get_lattice(config["lattice"])
    # lattice = ChainLattice(10)
    system = get_system(config)
    true_energy = system.get_eigenstates(1)[0][0]

    eval_set = get_eval_set(system, config["eval_set_max_size"])

    log_prob_fn = get_network(config, system)
    log_prob_fn.to(device)

    optimizer = torch.optim.Adam(
        log_prob_fn.parameters(), lr=lr, weight_decay=weight_decay
    )

    true_amplitudes = torch.from_numpy(
        np.abs(system.get_ground_state_coeffs(eval_set)).astype(np.float32)
    ).to(device)

    start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=(f"{output_dir}/{task_id}/logs/{start_timestamp}"))
    relsigns_fn = almost_true_relsigns(system, eps=sign_noise)
    warm_up = True
    warm_up_finished_at = 0
    signs_updated_at = 0
    for step in range(max_iter):
        if (
            not warm_up
            and (step - warm_up_finished_at) % config["sign_update_period"] == 0
        ):
            logger.debug("Updating signs")
            reconstructed_signs = reconstruct_signs(
                system,
                log_prob_fn,
                how=config["sign_reconstruction.method"],
                number_sweeps=config["sign_reconstruction.number_sweeps"],
                repetitions=config["sign_reconstruction.repetitions"],
                device=device,
            )
            relsigns_fn = custom_signs(
                system,
                reconstructed_signs,
            )
            if config["checkpoint_log_prob_fn_on_sign_update"]:
                torch.save(
                    log_prob_fn.state_dict(),
                    output_dir_task / f"log_prob_fn_{step-1}.pt",
                )
            if config["checkpoint_signs_greedy"]:
                reconstructed_signs_greedy = reconstruct_signs(
                    system, log_prob_fn, how="greedy_solve", device=device
                )
                torch.save(
                    reconstructed_signs_greedy,
                    output_dir_task / f"reconstructed_signs_greedy_{step-1}.pt",
                )
            if config["checkpoint_signs"]:
                torch.save(
                    reconstructed_signs,
                    output_dir_task / f"reconstructed_signs_{step-1}.pt",
                )
            signs_updated = True
            signs_updated_at = step
        else:
            signs_updated = False

        with local_sw("sampling"):
            other_options = {}
            if config["random_seed"]:
                other_options["force_numpy_sampling"] = True

            other_options["prob_to_float64"] = True

            states, log_probs, all_probs = sample_exactly(
                log_prob_fn,
                system.basis,
                SamplingOptions(
                    number_samples=n_samples,
                    number_chains=1,
                    mode="exact",
                    sweep_size=1,
                    number_discarded=0,
                    device=device,
                    other=other_options,
                ),
                return_all_probs=True,
            )
            states, weights = torch.unique(states.view(-1), return_counts=True)
            weights: torch.Tensor = weights.float() / torch.sum(weights)

        ipr = torch.sum(all_probs**2)
        writer.add_scalar("loss/ipr", ipr, step)

        with local_sw("local energies"):
            log_E_loc, *_ = compute_log_local_energies(
                system.hamiltonian,
                states.cpu().detach().numpy(),
                relsigns_fn=relsigns_fn,
                log_prob_fn=lambda s: log_prob_fn(
                    torch.from_numpy(s.astype(np.int64)).to(device)
                )
                .view(-1)
                .cpu()
                .detach()
                .numpy(),
            )

            log_E_loc = torch.from_numpy(log_E_loc).to(
                device=device, dtype=torch.complex64
            )

        with local_sw("energy gradient"):
            with torch.no_grad():
                weighted_E_loc = torch.exp(log_E_loc + torch.log(weights)).real
                grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

                # coeff 4 is due to: 2 from formula, 2 due to we are working with log probs
                # instead of log amplitudes

                # grad = 4 * (E - E @ weights) * weights

                grad = grad.view(-1, 1)
                grad_norm: torch.Tensor = torch.linalg.norm(grad)
                #    logger.info("‖∇E‖₂ = {}", grad_norm)
                writer.add_scalar("loss/‖∇E‖₂", grad_norm, step)
                E_variance = grad_norm / n_samples
                writer.add_scalar("loss/E_variance", E_variance, step)

                # Calculate full energy
                # if sampling_mode == "exact":
                E = torch.exp(log_E_loc).real
                E_full = E @ safe_exp(log_prob_fn(states).view(-1), normalise=True)
                E_full_delta = E_full - torch.tensor(true_energy)
                writer.add_scalar("loss/E_full_delta", E_full_delta, step)
                logger.info("E_full_delta = {}", E_full_delta)

        with local_sw("forward_and_backward"):
            optimizer.zero_grad()

            for states_chunk, grad_chunk in split_into_batches(
                (states.view(-1, 1), grad), batch_size
            ):
                output = log_prob_fn(states_chunk.view(-1))
                output.backward(grad_chunk, retain_graph=False)

            # full_gradient_norm = get_gradient_norm(forward_fn.parameters())
            # writer.add_scalar("loss/full_gradient_norm", full_gradient_norm, step)

            optimizer.step()

        with local_sw("evaluation"):
            predictions = log_prob_fn(
                torch.from_numpy(eval_set.astype(np.float32)).to(device)
            )
            predicted_amplitudes = safe_exp(predictions * 0.5)

            amplitude_overlap = find_overlap(
                true_amplitudes, predicted_amplitudes.view(-1)
            )
            sign_overlap = find_sign_overlap(system, relsigns_fn(system.basis.states))

            writer.add_scalar("overlap", amplitude_overlap, step)
            logger.info(
                f"{step}: amplitude_overlap = {amplitude_overlap:.3f}, sign_overlap = {sign_overlap:.3f},  ‖∇E‖₂ = {grad_norm:.3f}"
            )

        if (
            config["checkpoint_log_prob_fn_each"]
            and step % config["checkpoint_log_prob_fn_each"] == 0
        ):
            torch.save(
                log_prob_fn.state_dict(), output_dir_task / f"log_prob_fn_{step}.pt"
            )

        with jsonlines.open(
            output_dir_task / f"results.jsonl", mode="a"
        ) as json_writer:
            json_writer.write(
                {"config_" + key: value for key, value in config.items()}
                | {
                    "amplitude_overlap": amplitude_overlap.item(),
                    "‖∇E‖₂": grad_norm.item(),
                    "E_full_delta": E_full_delta.item(),
                    "ipr": ipr.item(),
                    "step": step,
                    "E_variance": E_variance.item(),
                    "start_timestamp": start_timestamp,
                    "task_id": task_id,
                    "device": str(device),
                    "warm_up": warm_up,
                    "sign_overlap": sign_overlap,
                    "step_since_warm_up": step - warm_up_finished_at
                    if not warm_up
                    else np.nan,
                    "signs_updated": signs_updated,
                    "signs_updated_at": signs_updated_at,
                    "step_since_signs_updated": step - signs_updated_at,
                    "git_hash": git_hash,
                }
            )

        if warm_up and amplitude_overlap > warm_up_overlap:
            warm_up = False
            warm_up_finished_at = step + 1

        if step % 20 == 0:
            logger.info(str(local_sw))


if __name__ == "__main__":
    fire.Fire(main)
