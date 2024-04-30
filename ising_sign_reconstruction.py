from spin_systems import SpinSystem
import numpy as np
from vmc_amplitude import get_csr_hamiltonian, safe_exp_numpy
from scipy.sparse import diags
import ising_glass_annealer as ising
from collections.abc import Callable
import os
import numpy.typing as npt
import torch
import lattice_symmetries as ls
from scipy.sparse import csr_matrix
from loguru import logger


def reconstruct_signs(
    basis: ls.Basis,
    hamiltonian_matrix: csr_matrix,
    log_prob_fn: Callable,
    how="annealing",
    number_sweeps: int | None = None,
    repetitions: int | None = None,
    device=torch.device("cpu"),
    force_symmetry=False,
):
    predicted_amplitudes = safe_exp_numpy(
        log_prob_fn(torch.from_numpy(basis.states.astype(np.int64)).to(device))
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        * 0.5,
        normalise=False,
    )
    predicted_amplitudes /= np.linalg.norm(predicted_amplitudes)

    ising_hamiltonian_matrix = (
        diags(predicted_amplitudes) @ hamiltonian_matrix @ diags(predicted_amplitudes)
    )
    if force_symmetry:
        logger.debug("Forcing symmetry")
        ising_hamiltonian_matrix = (
            ising_hamiltonian_matrix + ising_hamiltonian_matrix.transpose()
        ) / 2
        mask = np.abs(ising_hamiltonian_matrix.data) <= 1e-14
        ising_hamiltonian_matrix.data[mask] = 0
        ising_hamiltonian_matrix.eliminate_zeros()
        logger.debug("Forcing symmetry done")

    logger.debug("Creating ising Hamiltonian")
    ising_hamiltonian = ising.Hamiltonian(
        exchange=ising_hamiltonian_matrix,
        field=np.zeros(ising_hamiltonian_matrix.shape[0]),
    )
    if how == "annealing":
        logger.debug("Using annealing")
        if repetitions is None:
            raise ValueError(
                "If how='annealing', repetitions must be specified, but it was None"
            )
        if number_sweeps is None:
            raise ValueError(
                "If how='annealing', number_sweeps must be specified, but it was None"
            )
        reconstructed_bits = ising.anneal(
            ising_hamiltonian, repetitions=repetitions, number_sweeps=number_sweeps
        )[0]
    elif how == "greedy_solve":
        logger.debug("Using greedy solve")
        reconstructed_bits = ising.greedy_solve(ising_hamiltonian)[0]
    else:
        raise ValueError(f"how={how} not implemented")

    reconstructed_signs = ising.bits_to_signs(
        reconstructed_bits,
        ising_hamiltonian.exchange.shape[0],
    )
    return reconstructed_signs


def custom_signs(system: SpinSystem, signs: npt.NDArray):
    def get_signs(s: npt.NDArray[np.uint64]) -> npt.NDArray:
        return signs[system.basis.index(s)]

    return get_signs


def find_sign_overlap(
    system: SpinSystem, reconstructed_signs: npt.NDArray
) -> float | np.floating:
    ground_state = system.ground_state
    reconstructed_sign_overlap = (
        (reconstructed_signs * (np.sign(ground_state))) * ground_state**2
    ).sum()

    return np.abs(reconstructed_sign_overlap)
