from typing import Callable

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.sparse import diags

from heisenberg_hamiltonians import HeisenbergJ1J2
from nqs_playground_helpers import forward_with_batches
from spin_lattices import KagomeLattice
from vmc_amplitude import find_nbd, safe_exp_numpy, transfer_signs_to_H, true_relsigns


def generate_training_set_lanczos(
    hamiltonian: ls.Operator,
    states: npt.NDArray[np.uint64],
    log_prob_fn: Callable[[npt.NDArray], npt.NDArray],
    relsigns_fn: Callable[[npt.NDArray], npt.NDArray],
    batch_size=8192,
):
    """
    This function follows [1] in implementing two-step Lanczos.

    [1] Hongwei Chen, Douglas Hendry, Phillip Weinberg, Adrian E. Feiguin,
        Systematic improvement of neural network quantum states using Lanczos
        https://proceedings.neurips.cc/paper_files/paper/2022/file/3173c427cb4ed2d5eaab029c17f221ae-Paper-Conference.pdf
    """
    _, nbd_states = find_nbd(hamiltonian, states)
    nbd_matrix2, nbd_states2 = find_nbd(hamiltonian, nbd_states)

    nbd_matrix2_w_signs = transfer_signs_to_H(nbd_states, nbd_matrix2, nbd_states2, relsigns_fn)

    states_indices_in_nbd_states = np.searchsorted(nbd_states, states)
    nbd_states_indices_in_nbd_states2 = np.searchsorted(nbd_states2, nbd_states)

    # fmt: off
    nbd_matrix_w_signs = (nbd_matrix2_w_signs
                          [states_indices_in_nbd_states, :]
                          [:, nbd_states_indices_in_nbd_states2])
    # fmt: on

    psi_nbd2 = safe_exp_numpy(
        forward_with_batches(log_prob_fn, nbd_states2, batch_size=batch_size) * 0.5
    )
    psi_nbd = psi_nbd2[nbd_states_indices_in_nbd_states2]
    psi_states = psi_nbd[states_indices_in_nbd_states]

    H_psi_nbd = nbd_matrix2_w_signs @ psi_nbd2
    H_psi_states = H_psi_nbd[states_indices_in_nbd_states]

    local_energies = H_psi_states / psi_states
    local_energies_H2 = (nbd_matrix_w_signs @ H_psi_nbd) / psi_states

    H_expected = local_energies.mean()
    H2_expected = (local_energies**2).mean()
    H3_expected = (local_energies * local_energies_H2).mean()

    v = np.sqrt(H2_expected - H_expected**2)
    r = (H3_expected - 3 * H_expected * H2_expected + 2 * H_expected**3) / (2 * v**3)
    alpha = r - np.sqrt(r**2 + 1)
    logger.info(f"alpha = {alpha}")

    psi_1 = (H_psi_states - H_expected * psi_states) / v

    new_psi = psi_states / np.sqrt(1 + alpha**2) + psi_1 * alpha / np.sqrt(1 + alpha**2)
    new_logprobs = np.log(new_psi**2)

    return new_logprobs, alpha


def generate_training_set(
    hamiltonian: ls.Operator,
    states: npt.NDArray[np.uint64],
    log_prob_fn: Callable,
    relsigns_fn: Callable,
    energy_baseline: float,
    batch_size: int = 8192,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate a training set for the amplitude optimization problem.

    Args:
        hamiltonian: The Hamiltonian of the system.
        states: The states to sample from.
        log_prob_fn: The log-probability function.

    Returns:
        log_probs: The log-probabilities of the states.
    """
    M, nbd_states = find_nbd(hamiltonian, states, energy_baseline=energy_baseline)
    M_with_signs = -transfer_signs_to_H(states, M, nbd_states, relsigns_fn)
    current_log_probs = forward_with_batches(log_prob_fn, nbd_states, batch_size=batch_size)
    current_amplitudes = safe_exp_numpy(current_log_probs * 0.5)

    new_psi = (M_with_signs @ current_amplitudes).real

    local_energies = new_psi / current_amplitudes[np.searchsorted(nbd_states, states)]

    new_logprobs = np.log(np.abs(new_psi)) * 2

    return new_logprobs, local_energies, new_psi


def test_generate_training_set_lanczos():
    lattice = KagomeLattice(2, 3)
    system = HeisenbergJ1J2(lattice, J2=1, use_symmetries=False, spin_inversion=None)
    system.get_eigenstates(1)
    n_samples = 100000
    test_wavefunction = np.random.uniform(0, 1, size=system.basis.states.shape[0])
    test_probs = test_wavefunction**2
    test_probs /= test_probs.sum()

    def test_log_prob_fn(states):
        return np.log(np.abs(test_wavefunction[system.basis.index(states)])) * 2

    sampled_states = np.random.choice(
        system.basis.states, size=n_samples, replace=True, p=test_probs
    )

    new_logprobs, _ = generate_training_set_lanczos(
        hamiltonian=system.hamiltonian,
        states=sampled_states,
        log_prob_fn=test_log_prob_fn,
        relsigns_fn=true_relsigns(system),
    )

    mc_new_psi_0 = np.exp(new_logprobs / 2)

    true_signs = np.sign(system.get_ground_state_coeffs(system.basis.states))

    H_matrix, nbd_states = find_nbd(system.hamiltonian, system.basis.states)
    assert np.all(nbd_states == system.basis.states)

    H = diags(true_signs) @ H_matrix @ diags(true_signs)

    psi_0 = test_wavefunction
    psi_0 /= np.linalg.norm(psi_0)

    H_psi_0 = H @ psi_0
    psi_1 = H_psi_0
    psi_1 = psi_1 - psi_1 @ psi_0 * psi_0
    psi_1 = psi_1 / np.linalg.norm(psi_1)

    H_psi_1 = H @ psi_1
    a11 = psi_0 @ H_psi_0
    a12 = psi_0 @ H_psi_1
    a22 = psi_1 @ H_psi_1

    H_restricted = np.array([[a11, a12], [a12, a22]])
    eigenvalues, eigenvectors = np.linalg.eigh(H_restricted)
    # print(eigenvalues, eigenvectors)

    exact_new_psi_0 = eigenvectors[0, 0] * psi_0 + eigenvectors[1, 0] * psi_1
    exact_new_psi_0 = exact_new_psi_0 / np.linalg.norm(exact_new_psi_0)
    exact_new_psi_0 = np.abs(exact_new_psi_0)

    exact_new_psi_0_part = exact_new_psi_0[system.basis.index(sampled_states)]
    assert (
        1
        - (exact_new_psi_0_part @ mc_new_psi_0)
        / (np.linalg.norm(exact_new_psi_0_part) * np.linalg.norm(mc_new_psi_0))
        < 1e-3
    )

    assert (
        (
            exact_new_psi_0_part / np.linalg.norm(exact_new_psi_0_part)
            - mc_new_psi_0 / np.linalg.norm(mc_new_psi_0)
        )
        ** 2
    ).sum() < 1e-3


if __name__ == "__main__":
    test_generate_training_set_lanczos()
