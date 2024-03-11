from heisenberg_hamiltonians import SpinSystem
import numpy as np
from vmc_amplitude import get_csr_hamiltonian, safe_exp_numpy
from scipy.sparse import diags
import ising_glass_annealer as ising
from collections.abc import Callable
import os
import numpy.typing as npt
import torch


def reconstruct_signs(
    system: SpinSystem,
    log_prob_fn: Callable,
    how="annealing",
    number_sweeps: int | None = None,
    repetitions: int | None = None,
    device=torch.device("cpu"),
):
    predicted_amplitudes = safe_exp_numpy(
        log_prob_fn(torch.from_numpy(system.basis.states.astype(np.int64)).to(device))
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        * 0.5,
        normalise=False,
    )
    predicted_amplitudes /= np.linalg.norm(predicted_amplitudes)

    ising_hamiltonian_matrix = (
        diags(predicted_amplitudes)
        @ get_csr_hamiltonian(system)
        @ diags(predicted_amplitudes)
    )
    ising_hamiltonian = ising.Hamiltonian(
        exchange=ising_hamiltonian_matrix,
        field=np.zeros(ising_hamiltonian_matrix.shape[0]),
    )
    if how == "annealing":
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
        reconstructed_bits = ising.greedy_solve(ising_hamiltonian)[0]
    else:
        raise ValueError(f"how={how} not implemented")

    reconstructed_signs = ising.bits_to_signs(
        reconstructed_bits,
        ising_hamiltonian.exchange.shape[0],
    )
    return reconstructed_signs


def custom_signs(system: SpinSystem, signs: npt.NDArray):
    def get_signs(s):
        return signs[system.basis.index(s)]

    return get_signs


def find_sign_overlap(
    system: SpinSystem, reconstructed_signs: npt.NDArray
) -> float | np.floating:
    ground_state = system.get_ground_state()
    reconstructed_sign_overlap = (
        (reconstructed_signs * (np.sign(ground_state))) * ground_state**2
    ).sum()

    return np.abs(reconstructed_sign_overlap)
