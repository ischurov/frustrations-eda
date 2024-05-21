import ising_glass_annealer as ising
import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.sparse import csr_matrix, diags

from misc_utils import force_csr_symmetric
from spin_systems import SpinSystem


def reconstruct_signs(
    predicted_amplitudes: npt.NDArray[np.floating],
    hamiltonian_matrix: csr_matrix,
    how="annealing",
    number_sweeps: int | None = None,
    repetitions: int | None = None,
):
    logger.debug("Creating Ising Hamiltonian matrix")
    ising_hamiltonian_matrix = (
        diags(predicted_amplitudes) @ hamiltonian_matrix @ diags(predicted_amplitudes)
    )

    if (ising_hamiltonian_matrix != ising_hamiltonian_matrix.transpose()).nnz != 0:
        logger.debug("ising_hamiltonian_matrix is not symmetric, trying to fix it")
        discrepancy = np.abs(
            ising_hamiltonian_matrix - ising_hamiltonian_matrix.transpose()
        ).max()
        logger.debug(f"Discrepancy: {discrepancy}")
        if discrepancy > 1e-10:
            raise ValueError(
                f"Non-symmetric discrepancy is too large ({discrepancy=}). Looks more like a bug rather than round-off error"
            )
        ising_hamiltonian_matrix = force_csr_symmetric(ising_hamiltonian_matrix)

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


def custom_signs(basis: ls.SpinBasis, signs: npt.NDArray):
    def get_signs(s: npt.NDArray[np.uint64]) -> npt.NDArray:
        return signs[basis.index(s)]

    return get_signs


def partial_custom_signs(signs: npt.NDArray, states: npt.NDArray[np.uint64]):

    def get_signs(s: npt.NDArray[np.uint64]) -> npt.NDArray:
        idxs = np.searchsorted(states, s)
        assert np.all(states[idxs] == s)
        return signs[idxs]

    return get_signs


def find_sign_overlap(
    system: SpinSystem, reconstructed_signs: npt.NDArray, states=None
) -> float | np.floating:
    if states is None:
        states = system.basis.states
    ground_state_coeffs = system.get_ground_state_coeffs(states, apply_symmetries=False)
    reconstructed_sign_overlap = (
        (reconstructed_signs * (np.sign(ground_state_coeffs))) * ground_state_coeffs**2
    ).sum() / (ground_state_coeffs**2).sum()

    return np.abs(reconstructed_sign_overlap)
