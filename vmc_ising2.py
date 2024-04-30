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
    true_relsigns,
)
from dilated_nns_xors import resolve_config_inheritance
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from typing import Any
import torch
from torch import nn
import jsonlines
from conv2d_circular import InvariantSpinCNNRegression
from vmc_2024_02_28 import get_eval_set
from ising_sign_reconstruction import find_sign_overlap, reconstruct_signs, custom_signs
from nqs_playground_helpers import forward_with_batches
import lattice_symmetries as ls
from spin_lattices import AllToAllLattice
from vmc_ising2_configs import default_config, configs
from scipy.sparse import csr_matrix

self_name = Path(__file__).stem
git_hash = get_git_revision_hash()

output_dir = Path("experiments") / self_name


def get_device(config: dict[str, Any]):
    if config["vmc.device"] == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["vmc.device"])
    return device


def get_log_prob_fn(config: dict[str, Any], system: SpinSystem) -> nn.Module:
    if config["log_prob_fn"] == "invariant_cnn":
        assert isinstance(system.lattice, ParallelogramSpinLattice)
        return InvariantSpinCNNRegression(
            lattice=system.lattice,
            hidden_channels=config["log_prob_fn.invariant_cnn.hidden_channels"],
            dilations=config["log_prob_fn.invariant_cnn.dilations"],
            kernel_size=config["log_prob_fn.invariant_cnn.kernel_size"],
        )
    else:
        raise ValueError(f"Unknown architecture {config['architecture']}")


def get_optimizer(config: dict[str, Any], log_prob_fn: nn.Module):
    return torch.optim.Adam(
        log_prob_fn.parameters(),
        lr=config["vmc.lr"],
        weight_decay=config["vmc.weight_decay"],
    )


def get_energy_full(
    signs,
    basis: ls.Basis,
    hamiltonian: ls.Operator,
    log_prob_fn,
    device=torch.device("cpu"),
):
    probs = (
        safe_exp(
            (log_prob_fn(torch.from_numpy(basis.states.astype(np.int64)).to(device))),
            normalise=True,
        )
        .cpu()
        .view(-1)
        .detach()
        .numpy()
    )
    amplitudes = np.sqrt(probs)

    wavefunction = signs * amplitudes
    return (hamiltonian @ wavefunction) @ wavefunction


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)


def get_basis(config) -> Callable[[LatticeExpr], ls.Basis]:
    if config["system.symmetry_basis"] is None:
        return no_symmetries_basis(spin_inversion=config["system.spin_inversion"])
    if config["system.symmetry_basis"] == "zero_sector":
        return zero_sector_basis(spin_inversion=config["system.spin_inversion"])
    if config["system.symmetry_basis"] == "ground_state":
        return ground_state_basis(spin_inversion=config["system.spin_inversion"])
    raise ValueError(f"Unknown basis: {config['use_symmetries.basis']}")


def get_system(
    config: dict[str, Any], basis: Callable[[LatticeExpr], ls.Basis] | None = None
):
    if basis is None:
        basis = get_basis(config)
    lattice = get_lattice(config["system.lattice"])
    return spin_system(
        heisenberg(lattice=lattice, J1=1, J2=config["system.J2"]), basis=basis
    )


def get_predicted_amplitudes(
    log_prob_fn: nn.Module, eval_set: torch.Tensor
) -> torch.Tensor:
    predictions = log_prob_fn(eval_set)
    return safe_exp(predictions * 0.5).view(-1)


def evaluate_amplitude_overlap(
    log_prob_fn: nn.Module, eval_set: torch.Tensor, true_amplitudes: torch.Tensor
) -> float:
    predicted_amplitudes = get_predicted_amplitudes(log_prob_fn, eval_set)
    return find_overlap(true_amplitudes, predicted_amplitudes).item()


def do_warm_up(
    config: dict[str, Any],
    system: SpinSystem,
    device: torch.device,
    eval_set: torch.Tensor,
    true_amplitudes: torch.Tensor,
) -> nn.Module:
    if config["warm_up"] != "vmc_true_signs":
        raise ValueError(f"Unknown warm_up: {config['warm_up']}")

    log_prob_fn = get_log_prob_fn(config, system).to(device)
    optimizer = get_optimizer(config, log_prob_fn)
    relsigns_fn = true_relsigns(system, apply_symmetries=False)

    for step in range(config["max_iter"]):
        vmc_step(log_prob_fn, optimizer, relsigns_fn)
        if (
            evaluate_amplitude_overlap(log_prob_fn, eval_set, true_amplitudes)
            > config["warm_up.vmc_true_signs.overlap"]
        ):
            return log_prob_fn

    raise ValueError("Warm-up did not converge")


def get_relsigns_fn(
    config: dict[str, Any],
    system: SpinSystem,
    log_prob_fn: nn.Module,
    device: torch.device,
    full_spin_matrix,
):
    if config["sign_reconstruction.full_spin_regularization"] is not None:
        logger.debug("Using full spin regularization")
        matrix = (
            get_csr_hamiltonian(system)
            + config["sign_reconstruction.full_spin_regularization"] * full_spin_matrix
        )
        assert (matrix != matrix.transpose()).nnz == 0
    else:
        matrix = get_csr_hamiltonian(system)

    reconstructed_signs = reconstruct_signs(
        system.basis,
        matrix,
        log_prob_fn,
        how=config["sign_reconstruction.method"],
        number_sweeps=config["sign_reconstruction.number_sweeps"],
        repetitions=config["sign_reconstruction.repetitions"],
        device=device,
        force_symmetry=True,
    )

    relsigns_fn = custom_signs(
        system,
        reconstructed_signs,
    )
    return relsigns_fn


def main(task_id: int):
    config = get_config(task_id)

    output_dir_task = output_dir / str(task_id)
    output_dir_task.mkdir(parents=True, exist_ok=True)

    logger.add(output_dir_task / "log.log", backtrace=True, diagnose=True)

    if config["random_seed"] is not None:
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.use_deterministic_algorithms(True)

    device = get_device(config)

    logger.debug(f"Torch will use device: {device}")
    lattice = get_lattice(config["lattice"])
    all_to_all_lattice = AllToAllLattice(lattice)

    system = get_system(config)

    if config["sign_reconstruction.full_spin_regularization"] is not None:
        full_spin_system = spin_system(
            heisenberg(lattice=all_to_all_lattice),
            basis=get_basis(config),
        )
        full_spin_matrix = get_csr_hamiltonian(full_spin_system)

    for run in range(config["runs"]):
        do_run(
            task_id=task_id,
            output_dir_task=output_dir_task,
            run=run,
            config=config,
            system=system,
            device=device,
            full_spin_matrix=full_spin_matrix,
        )


def do_run(
    task_id: int,
    output_dir_task: Path,
    run: int,
    config: dict[str, Any],
    system: SpinSystem,
    device: torch.device,
    full_spin_matrix: csr_matrix,
):
    eval_set_numpy = get_eval_set(
        system, config["eval_set_max_size"], canonical_basis=False
    )

    eval_set_torch = torch.from_numpy(eval_set_numpy.astype(np.float32)).to(device)

    true_amplitudes = torch.from_numpy(
        np.abs(
            system.get_ground_state_coeffs(eval_set_numpy, apply_symmetries=False)
        ).astype(np.float32)
    ).to(device)

    log_prob_fn = do_warm_up(config, system, device, eval_set_torch, true_amplitudes)

    optimizer = get_optimizer(config, log_prob_fn)

    start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=(f"{output_dir_task}/logs/{start_timestamp}"))

    outer_steps = config["max_iter"] // config["importance_sampling_iterations"]

    for outer_step in range(outer_steps):
        if outer_step % config["sign_update_period"] == 0:

            if config["checkpoint_log_prob_fn_on_sign_update"]:
                torch.save(
                    log_prob_fn.state_dict(),
                    output_dir_task / f"log_prob_fn_{outer_step-1}.pt",
                )
            if config["checkpoint_signs_greedy"]:
                reconstructed_signs_greedy = reconstruct_signs(
                    system.basis,
                    get_csr_hamiltonian(system),
                    log_prob_fn,
                    how="greedy_solve",
                    device=device,
                    force_symmetry=True,
                )
                torch.save(
                    reconstructed_signs_greedy,
                    output_dir_task / f"reconstructed_signs_greedy_{outer_step-1}.pt",
                )
            if config["checkpoint_signs"]:
                torch.save(
                    reconstructed_signs,
                    output_dir_task / f"reconstructed_signs_{outer_step-1}.pt",
                )
            signs_updated = True
            signs_updated_at = outer_step
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
            # writer.add_scalar("loss/ipr", ipr, step)

            with torch.no_grad():
                initial_log_probs = log_prob_fn(states).to(torch.float64).view(-1)
                initial_weights = weights.to(torch.float64).view(-1)
                initial_log_weights = torch.log(initial_weights)

        for inner_step in range(config["importance_sampling_iterations"]):
            step = outer_step * config["importance_sampling_iterations"] + inner_step

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
                    if config["use_correct_E_full"]:
                        E_full = E.mean()
                    else:
                        E_full = E @ safe_exp(
                            log_prob_fn(states).view(-1), normalise=True
                        )
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
                sign_overlap = find_sign_overlap(
                    system, relsigns_fn(system.basis.states)
                )

                writer.add_scalar("overlap", amplitude_overlap, step)
                logger.info(
                    f"{step}: amplitude_overlap = {amplitude_overlap:.3f}, sign_overlap = {sign_overlap:.3f},  ‖∇E‖₂ = {grad_norm:.3f}"
                )

            if (
                config["checkpoint_log_prob_fn_each"]
                and step % config["checkpoint_log_prob_fn_each"] == 0
            ):
                torch.save(
                    log_prob_fn.state_dict(),
                    output_dir_task / f"log_prob_fn_{step}.pt",
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
                        "step_since_warm_up": (
                            step - warm_up_finished_at if not warm_up else np.nan
                        ),
                        "signs_updated": signs_updated,
                        "signs_updated_at": signs_updated_at,
                        "step_since_signs_updated": step - signs_updated_at,
                        "git_hash": git_hash,
                        "using_true_signs": using_true_signs,
                        "energy_with_true_signs": energy_with_true_signs,
                        "energy_with_reconstructed_signs": energy_with_reconstructed_signs,
                        "E_full": E_full.item(),
                        "true_energy": true_energy,
                        "run": run,
                    }
                )

            if warm_up and amplitude_overlap > warm_up_overlap:
                warm_up = False
                warm_up_finished_at = step + 1

            if step % 20 == 0:
                logger.info(str(local_sw))


if __name__ == "__main__":
    fire.Fire(main)
