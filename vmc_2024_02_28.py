import itertools
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from misc_utils import differentiable_safe_exp
from misc_utils import torch_overlap as find_overlap
from my_stopwatch import Stopwatch, stopwatch
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
from dilated_nns_xors import resolve_config_inheritance
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from typing import Any
import torch
from torch import nn
import jsonlines

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
default_config = {
    "n_samples": 10000,
    "lr": 1e-2,
    "momentum": 0.0,
    "batch_size": 10000,
    "sign_noise": 0.0,
    "weight_decay": 0,
    "annealing_steps": 0,
    "initial_temp": 3,
    "max_iter": 10000,
    "sampling_mode": "exact",
    "lattice": "kagome2x4",
    "J2": 1,
    "use_symmetries": False,
    "spin_inversion": None,
    "eval_set_max_size": 50000,
    "save_predictions": False,
    "device": "auto",
    "random_seed": None,
    "force_numpy_sampling": False,  # If random_seed is not None, numpy_sampling is forced automatically
    "prob_to_float64": False,
}

configs = {
    0: {
        "log_prob_fn": "dense_pairwise_xor",
        "n_hidden": 512,
        "hidden_layers": 1,
    },
    1: {"_inherit": 0, "lr": 1e-3},
    2: {
        "_inherit": 0,
        "max_iter": 1000,
        "save_predictions": True,
        "device": "cpu",
        "random_seed": 42,
    },
    3: {"_inherit": 2, "device": "cuda:0"},
    4: {"_inherit": 3, "force_numpy_sampling": True, "random_seed": None},
    5: {
        "_inherit": 3,
        "force_numpy_sampling": False,
        "random_seed": None,
    },
    6: {"_inherit": 5},  # run with CUBLAS_WORKSPACE_CONFIG=:4096:8
    7: {"_inherit": 3, "random_seed": 43},
    8: {"_inherit": 7},  # run with CUBLAS_WORKSPACE_CONFIG=:16:8
    9: {"_inherit": 5, "prob_to_float64": True},
}


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)


def get_network(config: dict[str, Any], system: SpinSystem) -> nn.Module:
    if config["log_prob_fn"] == "dense_pairwise_xor":
        pairs = tuple(
            map(np.array, zip(*itertools.combinations(range(system.number_spins), 2)))
        )
        return LogProbDenseNetPairwiseXor(
            system,
            n_hidden=config["n_hidden"],
            hidden_layers=config["hidden_layers"],
            xor_pairs=pairs,
        )
    else:
        raise ValueError(f"Unknown architecture {config['architecture']}")


def main(task_id: int):
    stopwatch.reset()
    local_sw = Stopwatch()
    config = get_config(task_id)
    n_samples = config["n_samples"]
    lr = config["lr"]
    momentum = config["momentum"]
    batch_size = config["batch_size"]
    sign_noise = config["sign_noise"]
    weight_decay = config["weight_decay"]
    annealing_steps = config["annealing_steps"]
    initial_temp = config["initial_temp"]
    max_iter = config["max_iter"]
    sampling_mode = config["sampling_mode"]

    output_dir_task = output_dir / str(task_id)

    if config["random_seed"] is not None:
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.use_deterministic_algorithms(True)

    temp = None

    output_dir_task.mkdir(parents=True, exist_ok=True)

    if config["device"] == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

    logger.debug(f"Torch will use device: {device}")
    # lattice = TriangleLattice(6, 4)
    lattice = get_lattice(config["lattice"])
    # lattice = ChainLattice(10)
    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1,
        J2=config["J2"],
        ground_state_cache_dir=Path("groundstates"),
        use_symmetries=config["use_symmetries"],
        spin_inversion=config["spin_inversion"],
    )
    true_energy, _ = system.get_eigenstates(1)
    true_energy = true_energy[0]

    if len(system.canonical_basis.states) > config["eval_set_max_size"]:
        eval_set = np.random.choice(
            system.canonical_basis.states, config["eval_set_max_size"], replace=False
        )
    else:
        eval_set = system.canonical_basis.states

    if config["save_predictions"]:
        np.save(output_dir_task / "eval_set.npy", eval_set)

    log_prob_fn = get_network(config, system)
    log_prob_fn.to(device)

    # log_prob_fn = SlaterAbs(
    #     system.lattice,
    #     system.canonical_basis,
    #     sign_cache_dir=Path("signs_cache"),
    #     initialization="randn",
    # )

    # log_prob_fn = KagomeCNNRegression(system.lattice, hidden_channels1=32, hidden_channels2=64)
    # optimizer = torch.optim.SGD(log_prob_fn.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.Adam(
        log_prob_fn.parameters(), lr=lr, weight_decay=weight_decay
    )

    true_amplitudes = torch.from_numpy(
        np.abs(system.get_ground_state_coeffs(eval_set)).astype(np.float32)
    ).to(device)

    relsigns_fn = almost_true_relsigns(system, eps=sign_noise)
    start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=(f"{output_dir}/{task_id}/logs/{start_timestamp}"))

    for step in range(max_iter):
        with local_sw("sampling"):
            if sampling_mode == "exact":
                other_options = {}
                if config["random_seed"] is not None or config["force_numpy_sampling"]:
                    other_options["force_numpy_sampling"] = True
                if config["prob_to_float64"]:
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
            # elif sampling_mode == "full":
            # This is commented out, as it is should be rewritten to be GPU compatible and reproducible
            #     states, log_probs, _extra = sample_full(
            #         log_prob_fn,
            #         system.basis,
            #         SamplingOptions(
            #             number_samples=1,
            #             number_chains=1,
            #             mode="full",
            #             sweep_size=1,
            #             number_discarded=0,
            #         ),
            #     )
            #     states = states.view(-1)

            #     weights: torch.Tensor = _extra["weights"].view(-1)
            #     all_probs = weights
            else:
                raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        if config["save_predictions"]:
            np.save(
                output_dir_task / f"states_{step}.npy",
                states.cpu().detach().numpy(),
            )
            np.save(
                output_dir_task / f"log_probs_{step}.npy",
                log_probs.cpu().detach().numpy(),
            )
            np.save(
                output_dir_task / f"weights_{step}.npy",
                weights.cpu().detach().numpy(),
            )
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
            if config["save_predictions"]:
                np.save(
                    output_dir_task / f"log_E_loc_{step}.npy",
                    log_E_loc,
                )
            log_E_loc = torch.from_numpy(log_E_loc).to(
                device=device, dtype=torch.complex64
            )

        # states = states.view(-1, states.size(-1))
        # log_probs = log_probs.view(-1)
        # weights = weights.view(-1)

        # Compute output gradient

        with local_sw("energy gradient"):
            with torch.no_grad():
                weighted_E_loc = torch.exp(log_E_loc + torch.log(weights)).real
                grad = 4 * (weighted_E_loc - weighted_E_loc.sum() * weights)

                if config["save_predictions"]:
                    np.save(
                        output_dir_task / f"weighted_E_loc_{step}.npy",
                        weighted_E_loc.cpu().detach().numpy(),
                    )
                    np.save(
                        output_dir_task / f"grad_{step}.npy",
                        grad.cpu().detach().numpy(),
                    )

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

            forward_fn = log_prob_fn
            for states_chunk, grad_chunk in split_into_batches(
                (states.view(-1, 1), grad), batch_size
            ):
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

        with local_sw("evaluation"):
            predictions = log_prob_fn(
                torch.from_numpy(eval_set.astype(np.float32)).to(device)
            )
            predicted_amplitudes = safe_exp(predictions * 0.5)

            overlap = find_overlap(true_amplitudes, predicted_amplitudes.view(-1))
            writer.add_scalar("overlap", overlap, step)
            logger.info(f"{step}: overlap = {overlap:.3f}, ‖∇E‖₂ = {grad_norm:.3f}")

        with jsonlines.open(
            output_dir_task / f"results.jsonl", mode="a"
        ) as json_writer:
            json_writer.write(
                config
                | {
                    "overlap": overlap.item(),
                    "‖∇E‖₂": grad_norm.item(),
                    "E_full_delta": E_full_delta.item(),
                    "ipr": ipr.item(),
                    "step": step,
                    "E_variance": E_variance.item(),
                    "temp": temp,
                    "start_timestamp": start_timestamp,
                    "task_id": task_id,
                    "device": str(device),
                }
            )
        if config["save_predictions"]:
            np.save(
                output_dir_task / f"predictions_{step}.npy",
                predictions.cpu().detach().numpy(),
            )
            np.save(
                output_dir_task / f"amplitudes_{step}.npy",
                predicted_amplitudes.cpu().detach().numpy(),
            )

        if step % 20 == 0:
            logger.info(str(local_sw))


if __name__ == "__main__":
    fire.Fire(main)
