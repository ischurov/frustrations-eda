from heisenberg_hamiltonians import SpinSystem

import numpy as np
from typing import Callable
import numpy.typing as npt
import lattice_symmetries as ls

from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import connected_components


def nbd_matrix_to_graph(
    states: np.ndarray, nbd_matrix: csr_matrix, nbd_states: np.ndarray
) -> csr_matrix:
    """
    Constructs a graph from a neighborhood matrix (see ``find_nbd``):

    - Expands matrix to make it square. Rows are rearranged to align them
        with columns, indexed by ``nbd_states``. I.e. row ``i`` corresponds to
        ``nbd_states[i]``.

    - Symmetrizes the matrix.

    - Converts to a graph by thresholding at 0.
    """
    # Get the non-zero elements of the nbd_matrix.
    nonzero_row, nonzero_col = nbd_matrix.nonzero()

    # Map the original indices of the states to the corresponding indices in nbd_states.
    state_indices = np.searchsorted(nbd_states, states)
    mapped_row_indices = state_indices[nonzero_row]

    # Create two COO matrices: one for the original non-zero elements, and one for the transposed elements.
    data = np.ones_like(nonzero_row, dtype=np.uint8)
    coo_mat = coo_matrix(
        (data, (mapped_row_indices, nonzero_col)), shape=(len(nbd_states), nbd_matrix.shape[1])
    )
    coo_mat_transpose = coo_matrix(
        (data, (nonzero_col, mapped_row_indices)), shape=(nbd_matrix.shape[1], len(nbd_states))
    )

    # Add the two COO matrices and convert to a CSR matrix for efficient arithmetic operations.
    symmetric_matrix = (coo_mat + coo_mat_transpose).tocsr()

    # Threshold at 0.
    return (symmetric_matrix != 0).astype(np.uint8)


def find_nbd(
    hamiltonian: ls.Operator, states: npt.NDArray[np.uint64]
) -> tuple[csr_matrix, npt.NDArray[np.uint64]]:
    """
    Constructs a sparse matrix that is a slice of the Hamiltonian matrix.
    Included rows are determined by ``states``, included columns

    Parameters
    ----------
    system : SpinSystem
        The system to construct the matrix for.

    states : npt.NDArray[np.uint64]
        The states whose neighbors to include in the matrix.

    Returns
    -------
    M : csr_matrix
        The sparse matrix.

    nbd_states : npt.NDArray[np.uint64]
        The sorted array of neighbors.

        The following holds:

        ``M[i, j] = <states[i] | H | nbd_states[j]>``
    """
    coeff_rows = []
    nbd_states_rows = []
    row_indices = [0]
    for state in states:
        # process neighbors
        cur_coeffs, cur_nbd_states = map(
            list, zip(*hamiltonian.apply_off_diag_to_basis_state(state))
        )

        # process self
        cur_coeffs.append(hamiltonian.apply_diag_to_basis_state(state))
        cur_nbd_states.append(state)

        # make rows
        coeff_rows.append(cur_coeffs)
        nbd_states_rows.append(cur_nbd_states)
        row_indices.append(row_indices[-1] + len(cur_coeffs))

    coeffs_data = np.concatenate(coeff_rows).astype(np.float64)
    nbd_states_data = np.concatenate(nbd_states_rows)
    nbd_states = np.unique(nbd_states_data)
    nbd_indices = np.searchsorted(nbd_states, nbd_states_data)
    row_indices = np.array(row_indices)

    return (
        csr_matrix((coeffs_data, nbd_indices, row_indices), shape=(len(states), len(nbd_states))),
        nbd_states,
    )


def true_relsigns(system: SpinSystem) -> Callable[[npt.NDArray], npt.NDArray]:
    def relings(cluster: npt.NDArray) -> npt.NDArray:
        return np.sign(system.get_ground_state_coeffs(cluster)) * np.random.choice([-1, 1])

    return relings


def almost_true_relsigns(system: SpinSystem, eps: float) -> Callable[[npt.NDArray], npt.NDArray]:
    def relings(cluster: npt.NDArray) -> npt.NDArray:
        return np.sign(system.get_ground_state_coeffs(cluster)) * np.random.choice(
            [1, -1], p=[1 - eps, eps], size=len(cluster)
        )

    return relings


def transfer_signs_to_H(
    states: npt.NDArray,
    M: csr_matrix,
    nbd_states: npt.NDArray,
    relsign_fn: Callable[[npt.NDArray], npt.NDArray],
):
    """
    Moves the signs from the relative signs to the Hamiltonian matrix.
    """
    graph = nbd_matrix_to_graph(states, M, nbd_states)
    _, labels = connected_components(graph, directed=False)
    relsigns = np.empty(len(nbd_states), dtype=np.int8)
    for component in np.unique(labels):
        component_mask = labels == component
        cluster = nbd_states[component_mask]
        relsigns[component_mask] = relsign_fn(cluster)

    state_indices = np.searchsorted(nbd_states, states)

    return diags(relsigns[state_indices], format="csr") @ M @ diags(relsigns, format="csr")


def safe_exp_numpy(x: npt.NDArray, normalise: bool = True) -> npt.NDArray:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - x.max()
    np.exp(x, out=x)
    if normalise:
        x /= x.sum()
    return x


def compute_local_energies(
    hamiltonian: ls.Operator,
    states: npt.NDArray[np.uint64],
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    log_prob_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    nbd_matrix, nbd_states = find_nbd(hamiltonian, states)
    nbd_matrix_w_signs = transfer_signs_to_H(states, nbd_matrix, nbd_states, relsigns_fn)
    abs_psi_nbd = safe_exp_numpy(log_prob_fn(nbd_states) * 0.5)
    states_indices = np.searchsorted(nbd_states, states)
    abs_psi_states = abs_psi_nbd[states_indices]
    return nbd_matrix_w_signs @ abs_psi_nbd / abs_psi_states
