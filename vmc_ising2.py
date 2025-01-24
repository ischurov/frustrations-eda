import itertools
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import Value, log_, nn
from torch.utils.tensorboard import SummaryWriter

from conv2d_circular import InvariantSpinCNNRegression
from dilated_nns_xors import resolve_config_inheritance
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from ising_sign_reconstruction import (
    find_sign_overlap,
    partial_custom_signs,
    custom_signs_hadamard_spread,
    reconstruct_signs,
)
from misc_utils import differentiable_safe_exp, get_git_revision_hash
from misc_utils import torch_overlap as find_overlap
from nqs_playground_helpers import (
    SamplingOptions,
    SpinDataset,
    forward_with_batches,
    safe_exp,
    sample_exactly,
    sample_full,
    split_into_batches,
)
from spin_lattices import AllToAllLattice, KagomeLattice, ParallelogramSpinLattice
from spin_systems import (
    LatticeExpr,
    SpinSystem,
    ground_state_basis,
    heisenberg,
    no_symmetries_basis,
    spin_system,
    zero_sector_basis,
)
from vmc_2024_02_28 import get_eval_set, get_network
from vmc_amplitude import (
    LogProbDenseNetPairwiseXor,
    almost_true_relsigns,
    compute_log_local_energies,
    get_csr_hamiltonian,
    safe_exp_numpy,
    true_relsigns,
)
from vmc_ising2_configs import configs, default_config
from scipy.sparse import diags
from parity import calculate_fourier_transform_matrix

self_name = Path(__file__).stem
git_hash = get_git_revision_hash()

output_dir = Path("experiments") / self_name


@dataclass
class LearnableNetwork:
    network: nn.Module
    optimizer: torch.optim.Optimizer | None
    device: torch.device


class SpinOnlyDataset(torch.utils.data.IterableDataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    :param spins: either a ``numpy.ndarray`` of ``uint64`` or a
        ``torch.Tensor`` of ``int64`` containing compact spin configurations.
    :param values: a ``torch.Tensor``.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle the samples.
    :param device: device where the batches will be used.
    """

    def __init__(self, spins, batch_size, shuffle=False, device=None):
        if isinstance(spins, np.ndarray):
            if spins.dtype != np.uint64:
                raise TypeError(
                    "spins must be a numpy.ndarray of uint64; got numpy.ndarray "
                    "of {}".format(spins.dtype.name)
                )
            # Use int64 because PyTorch doesn't support uint64
            self.spins = torch.from_numpy(spins.view(np.int64))
        elif isinstance(spins, torch.Tensor):
            if spins.dtype != torch.int64:
                raise TypeError(
                    "spins must be a torch.Tensor of int64; got torch.Tensor "
                    "of {}".format(spins.dtype)
                )
            self.spins = spins
        else:
            raise TypeError(
                "spins must be either a numpy.ndarray of uint64 or a "
                "torch.Tensor of int64; got {}".format(type(spins))
            )

        if batch_size <= 0:
            raise ValueError(
                "invalid batch_size: {}; expected a positive integer"
                "".format(batch_size)
            )

        self.batch_size = batch_size

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.spins = self.spins.to(self.device)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.spins.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.spins.size(0), device=self.device)
            spins = self.spins[indices]
        else:
            spins = self.spins
        return iter(torch.split(spins, self.batch_size))


def predict_amplitudes_numpy(
    log_prob_network: LearnableNetwork,
    states: npt.NDArray[np.uint64],
    batch_size: int,
    normalize=False,
):
    predicted_amplitudes = safe_exp_numpy(
        forward_with_batches(
            f=lambda batch: log_prob_network.network(
                torch.from_numpy(batch).to(log_prob_network.device)
            )
            .detach()
            .cpu()
            .numpy(),
            xs=states.astype(np.int64),
            batch_size=batch_size,
        ).reshape(-1)
        * 0.5,
        normalise=False,
    )
    if normalize:
        predicted_amplitudes /= np.linalg.norm(predicted_amplitudes)
    return predicted_amplitudes


def get_device(config: dict[str, Any]):
    if config["vmc.device"] == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["vmc.device"])
    return device


def get_optimizer(config: dict[str, Any], log_prob_fn: nn.Module):
    return torch.optim.Adam(
        log_prob_fn.parameters(),
        lr=config["vmc.lr"],
        weight_decay=config["vmc.weight_decay"],
    )


# def get_energy_full(
#     signs,
#     basis: ls.Basis,
#     hamiltonian: ls.Operator,
#     log_prob_fn,
#     device=torch.device("cpu"),
# ):
#     probs = (
#         safe_exp(
#             (log_prob_fn(torch.from_numpy(basis.states.astype(np.int64)).to(device))),
#             normalise=True,
#         )
#         .cpu()
#         .view(-1)
#         .detach()
#         .numpy()
#     )
#     amplitudes = np.sqrt(probs)

#     wavefunction = signs * amplitudes
#     return (hamiltonian @ wavefunction) @ wavefunction


def get_config(task_id: int):
    current_config = resolve_config_inheritance(task_id, configs=configs)
    if not set(current_config).issubset(set(default_config)):
        raise ValueError(
            f"Unknown keys in config: {set(current_config) - set(default_config)}"
        )
    return default_config | current_config


def get_basis(config) -> Callable[[LatticeExpr], ls.Basis]:
    if config["system.symmetry_basis"] is None:
        return no_symmetries_basis(spin_inversion=config["system.spin_inversion"])
    if config["system.symmetry_basis"] == "zero_sector":
        return zero_sector_basis(spin_inversion=config["system.spin_inversion"])
    if config["system.symmetry_basis"] == "ground_state":
        return ground_state_basis(spin_inversion=config["system.spin_inversion"])
    raise ValueError(f"Unknown basis: {config['system.symmetry_basis']}")


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


@dataclass
class RunnerEnvironment:
    run: int
    start_timestamp: str
    task_id: int
    output_dir: Path
    eval_set: torch.Tensor
    true_amplitudes: torch.Tensor
    system: SpinSystem

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        system: SpinSystem,
        run: int,
        task_id: int,
        output_dir_task: Path,
    ):
        self.eval_set = torch.from_numpy(
            get_eval_set(
                system, config["vmc.eval_set_max_size"], canonical_basis=False
            ).astype(np.int64)
        ).to(device)

        self.true_amplitudes = torch.from_numpy(
            np.abs(
                system.get_ground_state_coeffs(
                    self.eval_set.cpu().numpy(), apply_symmetries=False
                )
            ).astype(np.float32)
        ).to(device)

        self.start_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.system = system
        self.output_dir = output_dir_task
        self.task_id = task_id
        self.run = run


def do_warm_up(
    config: dict[str, Any],
    device: torch.device,
    env: RunnerEnvironment,
) -> LearnableNetwork:
    system = env.system
    if config["warm_up"] != "vmc_true_signs":
        raise ValueError(f"Unknown warm_up: {config['warm_up']}")

    log_prob_fn = get_network(config, system).to(device)
    optimizer = get_optimizer(config, log_prob_fn)

    log_prob_network = LearnableNetwork(log_prob_fn, optimizer, device)

    for log_prob_fn, inner_states, grad, E_full_est, vmc_step_extra in vmc_step(
        log_prob_network=log_prob_network,
        get_relsigns=lambda states, log_prob_fn: (
            true_relsigns(system, apply_symmetries=False),
            {},
        ),
        system=system,
        sign_reconstruction_update_each_outer_steps=1,
        outer_sample_size=config["warm_up.vmc_true_signs.outer_sample_size"]
        or config["vmc.outer_sample_size"],
        inner_sample_size=config["warm_up.vmc_true_signs.inner_sample_size"]
        or config["vmc.inner_sample_size"],
        inner_epochs=config["warm_up.vmc_true_signs.inner_epochs"]
        or config["vmc.inner_epochs"],
        batch_size=config["warm_up.vmc_true_signs.batch_size"]
        or config["vmc.batch_size"],
        outer_log_prob_fn_factory=lambda: get_network(config, system).to(device),
    ):
        amplitude_overlap = evaluate_and_write(
            log_prob_fn=log_prob_fn,
            grad=grad,
            E_full_est=E_full_est,
            vmc_step_extra=vmc_step_extra,
            config=config,
            env=env,
            additional_info={"warm_up": True},
        )
        if amplitude_overlap > config["warm_up.vmc_true_signs.overlap"]:
            return log_prob_network

        if config["warm_up.vmc_true_signs.max_steps"] is not None:
            if vmc_step_extra["step"] > config["warm_up.vmc_true_signs.max_steps"]:
                raise ValueError("Warm-up did not converge")

    raise ValueError("vmc_step stopped prematurely")


def get_relsigns_fn(
    states: torch.Tensor,
    hamiltonian: ls.Operator,
    log_prob_network: LearnableNetwork,
    config: dict[str, Any],
):
    nbd_states: npt.NDArray[np.uint64]
    matrix, nbd_states = hamiltonian.to_partial_csr(states.view(-1).cpu().numpy())  # type: ignore
    extension_states = states.view(-1).cpu().numpy().sort()
    extension_states_nbd: npt.NDArray[np.uint64] = nbd_states
    if config["sign_reconstruction.extension_steps"] is None:
        raise ValueError("Sign reconstruction of full basis not implemented")
    logger.debug(f"{states.shape=}, {nbd_states.shape=}")
    for _ in range(config["sign_reconstruction.extension_steps"]):
        extension_states: npt.NDArray[np.uint64] = extension_states_nbd
        matrix, extension_states_nbd = hamiltonian.to_partial_csr(
            extension_states
        )  # type: ignore
        logger.debug(f"{extension_states.shape=}, {extension_states_nbd.shape=}")

    matrix = matrix[:, np.searchsorted(extension_states_nbd, extension_states)]
    assert np.abs(matrix - matrix.transpose()).max() < 1e-10

    graph_matrix = (matrix != 0).astype(np.int8)
    _, labels = connected_components(graph_matrix, directed=False)

    relsigns = np.empty(len(extension_states), dtype=np.int8)
    unique_labels = np.unique(labels)
    logger.debug(f"Found {len(unique_labels)} connected components")
    for component in unique_labels:
        logger.debug(f"Processing component {component}")
        component_mask = labels == component
        cluster = extension_states[component_mask]
        logger.debug(
            f"Component size: {len(cluster)} ({len(cluster) / len(hamiltonian.basis.states)} of full basis)"
        )
        matrix_block = matrix[np.ix_(component_mask, component_mask)]
        assert np.abs(matrix_block - matrix_block.transpose()).max() < 1e-10
        cluster_amplitudes = predict_amplitudes_numpy(
            log_prob_network=log_prob_network,
            states=cluster,
            batch_size=config["vmc.batch_size"],
        )

        if config["sign_reconstruction.hadamard_external_field.iterations"] > 1:
            cluster_nbd: npt.NDArray[np.uint64]
            extended_matrix, cluster_nbd = hamiltonian.to_partial_csr(cluster)  # type: ignore
            cluster_border = np.setdiff1d(cluster_nbd, cluster, assume_unique=True)
            cluster_border_amplitudes = predict_amplitudes_numpy(
                log_prob_network=log_prob_network,
                states=cluster_border,
                batch_size=config["vmc.batch_size"],
            )
            external_field_matrix = (
                diags(cluster_amplitudes)
                @ extended_matrix[:, np.searchsorted(cluster_nbd, cluster_border)]
                @ diags(cluster_border_amplitudes)
            )

        external_field = np.zeros(cluster.shape[0])
        for hadamard_external_field_iteration in range(
            config["sign_reconstruction.hadamard_external_field.iterations"]
        ):
            relsigns[component_mask] = reconstruct_signs(
                predicted_amplitudes=cluster_amplitudes,
                hamiltonian_matrix=matrix_block,
                how=config["sign_reconstruction"],
                number_sweeps=config["sign_reconstruction.annealing.number_sweeps"],
                repetitions=config["sign_reconstruction.annealing.repetitions"],
                field=external_field
                * config["sign_reconstruction.hadamard_external_field.coeff"],
            )
            if (
                hadamard_external_field_iteration
                != config["sign_reconstruction.hadamard_external_field.iterations"] - 1
            ):
                transform_matrix = calculate_fourier_transform_matrix(
                    cluster_border, cluster, out_dtype="float64"
                )
                sign_predictions = np.sign(
                    transform_matrix
                    @ (
                        relsigns[component_mask]
                        * (
                            cluster_amplitudes
                            ** config[
                                "sign_reconstruction.hadamard_external_field.amplitude_power"
                            ]
                        )
                    )
                )
                external_field = external_field_matrix @ sign_predictions

    nbd_states_indices = np.searchsorted(extension_states, nbd_states)
    if config["sign_reconstruction.hadamard_spread"]:
        custom_signs = custom_signs_hadamard_spread
    else:
        custom_signs = partial_custom_signs

    relsigns_fn = custom_signs(
        signs=relsigns[nbd_states_indices],
        states=nbd_states,
    )

    return (
        relsigns_fn,
        {
            "extension_size": len(extension_states),
            "relative_extension_size": len(extension_states)
            / len(hamiltonian.basis.states),
            "n_connected_components": len(unique_labels),
            "largest_connected_component_size": max(
                len(extension_states[labels == component])
                for component in unique_labels
            ),
        },
    )


def prepare_main(task_id: int):
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
    lattice = get_lattice(config["system.lattice"])
    all_to_all_lattice = AllToAllLattice(lattice)

    system = get_system(config)

    if config["sign_reconstruction.full_spin_regularization"] is not None:
        full_spin_system = spin_system(
            heisenberg(lattice=all_to_all_lattice),
            basis=get_basis(config),
        )
        full_spin_matrix = get_csr_hamiltonian(full_spin_system)
    else:
        full_spin_matrix = None

    return (
        config,
        device,
        system,
        full_spin_matrix,
        output_dir_task,
    )


def repeat_epochs(it: Iterable, epochs: int):
    for epoch in range(epochs):
        for batch_index, x in enumerate(it):
            yield (epoch, batch_index, x)


def vmc_step(
    log_prob_network: LearnableNetwork,
    get_relsigns: Callable[
        [torch.Tensor, torch.nn.Module],
        tuple[Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]], dict[str, Any]],
    ],
    system: SpinSystem,
    sign_reconstruction_update_each_outer_steps: int,
    outer_sample_size: int,
    inner_sample_size: int,
    inner_epochs: int,
    batch_size: int,
    outer_log_prob_fn_factory: Callable[[], nn.Module],
):
    for outer_step in itertools.count():
        outer_states, outer_log_prob_fn = make_outer_sample(
            log_prob_network=log_prob_network,
            system=system,
            batch_size=batch_size,
            outer_sample_size=outer_sample_size,
            log_prob_fn_factory=outer_log_prob_fn_factory,
        )

        signs_updated = False

        if outer_step % sign_reconstruction_update_each_outer_steps == 0:
            relsigns_fn, relsigns_extra = get_relsigns(
                outer_states,
                outer_log_prob_fn,
            )
            signs_updated = True

        sign_overlap = abs(
            find_sign_overlap(
                system,
                relsigns_fn(outer_states.view(-1).cpu().numpy()),
                states=outer_states.view(-1).cpu().numpy(),
            )
        )
        logger.info(f"{sign_overlap=}")

        spin_dataset = SpinOnlyDataset(
            spins=outer_states,
            batch_size=inner_sample_size,
            device=log_prob_network.device,
            shuffle=True,
        )

        for inner_step, (epoch, batch_index, inner_states) in enumerate(
            repeat_epochs(spin_dataset, inner_epochs)
        ):
            step: int = outer_step * inner_epochs * len(spin_dataset) + inner_step
            inner_step_results = do_inner_step(
                batch_size=batch_size,
                log_prob_network=log_prob_network,
                inner_states=inner_states,
                outer_log_prob_fn=outer_log_prob_fn,
                relsigns_fn=relsigns_fn,
                hamiltonian=system.hamiltonian,
            )

            yield (
                *inner_step_results,
                {
                    "outer_step": outer_step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "step": step,
                    "inner_step": inner_step,
                    "sign_overlap": sign_overlap,
                    "signs_updated": signs_updated,
                }
                | relsigns_extra,
            )


def main(task_id: int):
    (
        config,
        device,
        system,
        full_spin_matrix,
        output_dir_task,
    ) = prepare_main(task_id)

    for run in range(config["runs"]):
        runner_env = RunnerEnvironment(
            config=config,
            device=device,
            system=system,
            output_dir_task=output_dir_task,
            run=run,
            task_id=task_id,
        )
        try:
            log_prob_network = do_warm_up(
                config=config,
                device=device,
                env=runner_env,
            )
        except ValueError as e:
            logger.error(f"Error during warm-up: {e}")
            continue

        for log_prob_fn, inner_states, grad, E_full_est, vmc_step_extra in vmc_step(
            log_prob_network=log_prob_network,
            get_relsigns=lambda states, log_prob_fn: get_relsigns_fn(
                config=config,
                hamiltonian=system.hamiltonian,
                log_prob_network=LearnableNetwork(
                    network=log_prob_fn, optimizer=None, device=device
                ),
                states=states,
            ),
            system=system,
            sign_reconstruction_update_each_outer_steps=config[
                "sign_reconstruction.update_each_outer_steps"
            ],
            outer_sample_size=config["vmc.outer_sample_size"],
            inner_sample_size=config["vmc.inner_sample_size"],
            inner_epochs=config["vmc.inner_epochs"],
            batch_size=config["vmc.batch_size"],
            outer_log_prob_fn_factory=lambda: get_network(config, system).to(device),
        ):
            evaluate_and_write(
                log_prob_fn=log_prob_fn,
                grad=grad,
                E_full_est=E_full_est,
                vmc_step_extra=vmc_step_extra,
                config=config,
                env=runner_env,
                additional_info={"warm_up": False},
            )
            if vmc_step_extra["step"] > config["vmc.max_steps"]:
                break


def evaluate_and_write(
    log_prob_fn: torch.nn.Module,
    grad: torch.Tensor,
    E_full_est: float,
    vmc_step_extra: dict[str, Any],
    config: dict[str, Any],
    env: RunnerEnvironment,
    additional_info: dict[str, Any] = {},
):
    amplitude_overlap = evaluate_log_prob_fn(
        log_prob_fn, env.eval_set, env.true_amplitudes
    )
    logger.info(f"{amplitude_overlap=}")

    current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    with jsonlines.open(env.output_dir / f"results.jsonl", mode="a") as json_writer:
        json_writer.write(
            config
            | {
                "run": env.run,
                "amplitude_overlap": amplitude_overlap,
                "start_timestamp": env.start_timestamp,
                "current_timestamp": current_timestamp,
                "task_id": env.task_id,
                "git_hash": git_hash,
                "energy_delta": E_full_est - env.system.ground_energy,
                "true_energy": env.system.ground_energy,
                "estimated_energy": E_full_est,
                "grad_norm": torch.norm(grad).item(),
            }
            | vmc_step_extra
            | additional_info
        )
    return amplitude_overlap


@torch.no_grad()
def evaluate_log_prob_fn(
    log_prob_fn: torch.nn.Module, eval_set: torch.Tensor, true_amplitudes: torch.Tensor
):
    predictions = log_prob_fn(eval_set)
    predicted_amplitudes = safe_exp(predictions * 0.5)

    amplitude_overlap = find_overlap(true_amplitudes, predicted_amplitudes.view(-1))
    return amplitude_overlap.item()


def make_outer_sample(
    log_prob_network: LearnableNetwork,
    system: SpinSystem,
    batch_size: int,
    outer_sample_size: int,
    log_prob_fn_factory: Callable[[], nn.Module],
):
    log_prob_fn = log_prob_network.network
    device = log_prob_network.device

    other_options = {}
    other_options["prob_to_float64"] = True
    other_options["batch_size"] = batch_size

    outer_states, log_probs = sample_exactly(
        log_prob_fn,
        system.basis,
        SamplingOptions(
            number_samples=outer_sample_size,
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
    outer_log_prob_fn = log_prob_fn_factory()
    outer_log_prob_fn.load_state_dict(log_prob_fn.state_dict())
    outer_log_prob_fn.to(device)

    return outer_states, outer_log_prob_fn


@torch.no_grad()
def get_grad(
    inner_states_with_repetitions: torch.Tensor,
    batch_size: int,
    device: torch.device,
    log_prob_fn: nn.Module,
    outer_log_prob_fn: nn.Module,
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    hamiltonian: ls.Operator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    (
        inner_states,
        counts,
    ) = torch.unique(
        inner_states_with_repetitions.view(-1),
        return_counts=True,
    )
    inner_weights = counts.float() / torch.sum(counts)
    inner_log_probs = forward_with_batches(
        outer_log_prob_fn, inner_states, batch_size=batch_size
    ).view(-1)

    log_E_loc, *_ = compute_log_local_energies(
        hamiltonian,
        inner_states.cpu().detach().numpy(),
        relsigns_fn=relsigns_fn,
        log_prob_fn=lambda s: forward_with_batches(
            log_prob_fn,
            torch.from_numpy(s.astype(np.int64)).to(device),
            batch_size=batch_size,
        )
        .view(-1)
        .cpu()
        .detach()
        .numpy(),
    )

    log_E_loc = torch.from_numpy(log_E_loc).to(device).to(torch.complex64)
    new_log_probs = (
        forward_with_batches(log_prob_fn, inner_states, batch_size=batch_size)
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
    return inner_states, grad, E_full_est.item()


def do_inner_step(
    log_prob_network: LearnableNetwork,
    inner_states: torch.Tensor,
    outer_log_prob_fn: nn.Module,
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    hamiltonian: ls.Operator,
    batch_size: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, float]:

    log_prob_fn = log_prob_network.network
    device = log_prob_network.device
    optimizer = log_prob_network.optimizer

    inner_states, grad, E_full_est = get_grad(
        inner_states_with_repetitions=inner_states,
        batch_size=batch_size,
        device=device,
        log_prob_fn=log_prob_fn,
        outer_log_prob_fn=outer_log_prob_fn,
        relsigns_fn=relsigns_fn,
        hamiltonian=hamiltonian,
    )

    optimizer.zero_grad()

    for states_chunk, grad_chunk in split_into_batches(
        (inner_states.view(-1, 1), grad), batch_size
    ):

        output = log_prob_fn(states_chunk.view(-1))  # type: ignore
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
    return log_prob_fn, inner_states, grad, E_full_est


if __name__ == "__main__":
    fire.Fire(main)
