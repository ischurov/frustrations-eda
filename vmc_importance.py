import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import fire
import jsonlines
import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn

from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from misc_utils import get_git_revision_hash
from misc_utils import torch_overlap as find_overlap
from nqs_playground_helpers import (
    SamplingOptions,
    forward_with_batches,
    safe_exp,
    sample_exactly,
    split_into_batches,
)
from spin_systems import SpinSystem
from vmc_2024_02_28 import get_device, get_eval_set, get_network
from vmc_amplitude import almost_true_relsigns, compute_log_local_energies
from vmc_importance_config import get_config
from vmc_ising import get_system

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
            task_id=task_id,
        )


def do_run(
    config: dict[str, Any],
    device: torch.device,
    output_dir_task: Path,
    run: int,
    system: SpinSystem,
    true_energy: float,
    task_id: int,
) -> None:
    eval_set = torch.from_numpy(
        get_eval_set(system, config["eval_set_max_size"], canonical_basis=False).astype(
            np.int64
        )
    ).to(device)

    log_prob_fn = get_network(config, system)
    log_prob_fn.to(device)

    optimizer = torch.optim.Adam(
        log_prob_fn.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    true_amplitudes = torch.from_numpy(
        np.abs(
            system.get_ground_state_coeffs(
                eval_set.cpu().numpy(), apply_symmetries=False
            )
        ).astype(np.float32)
    ).to(device)

    start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    relsigns_fn = almost_true_relsigns(
        system, eps=config["sign_noise"], apply_symmetries=False
    )
    for outer_step in range(config["max_iter"]):
        do_outer_step(
            config=config,
            device=device,
            eval_set=eval_set,
            log_prob_fn=log_prob_fn,
            optimizer=optimizer,
            output_dir_task=output_dir_task,
            relsigns_fn=relsigns_fn,
            run=run,
            outer_step=outer_step,
            system=system,
            true_amplitudes=true_amplitudes,
            true_energy=true_energy,
            start_timestamp=start_timestamp,
            task_id=task_id,
        )


def do_outer_step(
    config: dict,
    device: torch.device,
    eval_set: torch.Tensor,
    log_prob_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir_task: Path,
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    run: int,
    outer_step: int,
    system: SpinSystem,
    true_amplitudes: torch.Tensor,
    true_energy: float,
    start_timestamp: str,
    task_id: int,
) -> None:
    other_options = {}
    other_options["prob_to_float64"] = True
    other_options["batch_size"] = config["batch_size"]

    outer_states, log_probs = sample_exactly(
        log_prob_fn,
        system.basis,
        SamplingOptions(
            number_samples=config["outer_sample_size"],
            number_chains=1,
            mode="exact",
            sweep_size=1,
            number_discarded=0,
            device=device,
            other=other_options,
        ),
        return_all_probs=False,
    )
    outer_states = outer_states.to(device)
    outer_log_prob_fn = get_network(config, system)
    outer_log_prob_fn.load_state_dict(log_prob_fn.state_dict())
    outer_log_prob_fn.to(device)

    if config["outer_sample_size"] % config["inner_sample_size"] != 0:
        raise ValueError("outer_sample_size must be divisible by inner_sample_size")

    n_inner_samples = config["outer_sample_size"] // config["inner_sample_size"]

    for epoch in range(config["inner_epochs"]):
        shuffle_order = torch.randperm(config["outer_sample_size"])
        for i_sample in range(n_inner_samples):
            inner_step = epoch * n_inner_samples + i_sample
            step = outer_step * config["inner_epochs"] * n_inner_samples + inner_step
            if step > config["max_iter"]:
                sys.exit(0)

            inner_start = i_sample * config["inner_sample_size"]
            inner_end = (i_sample + 1) * config["inner_sample_size"]
            inner_states_all = outer_states[shuffle_order[inner_start:inner_end]]

            inner_states, grad, E_full_est = do_inner_step(
                inner_states_all=inner_states_all,
                config=config,
                device=device,
                log_prob_fn=log_prob_fn,
                outer_log_prob_fn=outer_log_prob_fn,
                optimizer=optimizer,
                relsigns_fn=relsigns_fn,
                hamiltonian=system.hamiltonian,
            )

            with torch.no_grad():
                predictions = log_prob_fn(eval_set)
                predicted_amplitudes = safe_exp(predictions * 0.5)

                amplitude_overlap = find_overlap(
                    true_amplitudes, predicted_amplitudes.view(-1)
                )
                logger.info(
                    f"Step: {step}, amplitude overlap: {amplitude_overlap.item()}"
                )

            current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with jsonlines.open(
                output_dir_task / f"results.jsonl", mode="a"
            ) as json_writer:
                json_writer.write(
                    config
                    | {
                        "run": run,
                        "outer_step": outer_step,
                        "inner_step": inner_step,
                        "step": step,
                        "epoch": epoch,
                        "amplitude_overlap": amplitude_overlap.item(),
                        "start_timestamp": start_timestamp,
                        "current_timestamp": current_timestamp,
                        "n_inner_samples": n_inner_samples,
                        "task_id": task_id,
                        "git_hash": git_hash,
                        "energy_delta": E_full_est.item() - true_energy,
                        "true_energy": true_energy,
                        "estimated_energy": E_full_est.item(),
                        "grad_norm": torch.norm(grad).item(),
                    }
                )


@torch.no_grad()
def get_grad(
    inner_states_all: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    log_prob_fn: nn.Module,
    outer_log_prob_fn: nn.Module,
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    hamiltonian: ls.Operator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        inner_states,
        counts,
    ) = torch.unique(
        inner_states_all.view(-1),
        return_counts=True,
    )
    inner_weights = counts.float() / torch.sum(counts)
    inner_log_probs = forward_with_batches(
        outer_log_prob_fn, inner_states, batch_size=config["batch_size"]
    ).view(-1)

    log_E_loc, *_ = compute_log_local_energies(
        hamiltonian,
        inner_states.cpu().detach().numpy(),
        relsigns_fn=relsigns_fn,
        log_prob_fn=lambda s: forward_with_batches(
            log_prob_fn,
            torch.from_numpy(s.astype(np.int64)).to(device),
            batch_size=config["batch_size"],
        )
        .view(-1)
        .cpu()
        .detach()
        .numpy(),
    )

    log_E_loc = torch.from_numpy(log_E_loc).to(device).to(torch.complex64)
    new_log_probs = (
        forward_with_batches(log_prob_fn, inner_states, batch_size=config["batch_size"])
        .view(-1)
        .to(torch.float64)
    )
    weights = safe_exp(torch.log(inner_weights) + new_log_probs - inner_log_probs).to(
        torch.float32
    )

    weighted_E_loc = torch.exp(log_E_loc + torch.log(weights)).real
    grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

    grad = grad.view(-1, 1)

    E = torch.exp(log_E_loc).real
    E_full_est = E @ safe_exp(new_log_probs.to(torch.float32).view(-1), normalise=True)
    return inner_states, grad, E_full_est


def do_inner_step(
    inner_states_all: torch.Tensor,
    config: dict,
    device: torch.device,
    log_prob_fn: torch.nn.Module,
    outer_log_prob_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    hamiltonian: ls.Operator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner_states, grad, E_full_est = get_grad(
        inner_states_all=inner_states_all,
        config=config,
        device=device,
        log_prob_fn=log_prob_fn,
        outer_log_prob_fn=outer_log_prob_fn,
        relsigns_fn=relsigns_fn,
        hamiltonian=hamiltonian,
    )

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
    return inner_states, grad, E_full_est


if __name__ == "__main__":
    fire.Fire(main)
