from heisenberg_hamiltonians import SpinSystem

import numpy as np
from typing import Callable
import numpy.typing as npt
import lattice_symmetries as ls

from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from my_stopwatch import stopwatch
from scipy.special import logsumexp


def apply_off_diag_to_basis_states(
    op: ls.Operator, alphas: npt.NDArray[np.uint64]
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.complex128], npt.NDArray[np.int64]]:
    alphas = np.asarray(alphas, order="C", dtype=np.uint64)
    alphas_ptr = ls.ffi.from_buffer("uint64_t[]", alphas, require_writable=False)

    betas = ls.ffi.new("chpl_external_array *")
    coeffs = ls.ffi.new("chpl_external_array *")
    offsets = ls.ffi.new("chpl_external_array *")
    kernels = ls.lib.ls_hs_internal_get_chpl_kernels()
    kernels.operator_apply_off_diag(
        op._payload, alphas.size, alphas_ptr, betas, coeffs, offsets, 0
    )

    offsets_arr = ls._chpl_external_array_as_ndarray(offsets, np.int64)
    betas_arr = ls._chpl_external_array_as_ndarray(betas, np.uint64)[: offsets_arr[-1]]
    coeffs_arr = ls._chpl_external_array_as_ndarray(coeffs, np.complex128)[: offsets_arr[-1]]
    return betas_arr, coeffs_arr, offsets_arr


def apply_diag_to_basis_states(
    op: ls.Operator, alphas: npt.NDArray[np.uint64]
) -> npt.NDArray[np.float64]:
    alphas = np.asarray(alphas, order="C", dtype=np.uint64)
    alphas_ptr = ls.ffi.from_buffer("uint64_t[]", alphas, require_writable=False)

    coeffs = ls.ffi.new("chpl_external_array *")
    kernels = ls.lib.ls_hs_internal_get_chpl_kernels()
    kernels.operator_apply_diag(op._payload, alphas.size, alphas_ptr, coeffs, 0)

    coeffs_arr = ls._chpl_external_array_as_ndarray(coeffs, np.float64)
    return coeffs_arr


def find_nbd_reference(
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
    nbd_states = np.unique(nbd_states_data).astype(np.uint64)
    nbd_indices = np.searchsorted(nbd_states, nbd_states_data)
    row_indices = np.array(row_indices)

    return (
        csr_matrix((coeffs_data, nbd_indices, row_indices), shape=(len(states), len(nbd_states))),
        nbd_states,
    )


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
    with stopwatch("vmc_amplitude/find_nbd/apply_off_diag"):
        nbd_states_data, coeffs_data, offsets_data = apply_off_diag_to_basis_states(
            hamiltonian, states
        )
    with stopwatch("vmc_amplitude/find_nbd/nbd_states"):
        nbd_states = np.unique(np.concatenate([nbd_states_data, states])).astype(np.uint64)
        # TODO: optimize this
    with stopwatch("vmc_amplitude/find_nbd/nbd_indices"):
        nbd_indices = np.searchsorted(nbd_states, nbd_states_data)
        # TODO: optimize this

    with stopwatch("vmc_amplitude/find_nbd/matrix_without_diag"):
        matrix_without_diag = csr_matrix(
            (coeffs_data, nbd_indices, offsets_data), shape=(len(states), len(nbd_states))
        )

    with stopwatch("vmc_amplitude/find_nbd/apply_diag"):
        # Add diagonal elements
        diag_coeffs = apply_diag_to_basis_states(hamiltonian, states)
    with stopwatch("vmc_amplitude/find_nbd/diag_indices"):
        diag_indices = np.searchsorted(nbd_states, states)
    with stopwatch("vmc_amplitude/find_nbd/matrix_with_diag"):
        matrix_with_diag = matrix_without_diag + csr_matrix(
            (diag_coeffs, diag_indices, np.arange(len(states) + 1)),
            shape=(len(states), len(nbd_states)),
        )

    return matrix_with_diag, nbd_states


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
    with stopwatch("vmc_amplitude/nbd_matrix_to_graph/nonzero"):
        nonzero_row, nonzero_col = nbd_matrix.nonzero()
        n_nonzero = len(nonzero_row)
        assert len(nonzero_col) == n_nonzero

    # Map the original indices of the states to the corresponding indices in nbd_states.
    with stopwatch("vmc_amplitude/nbd_matrix_to_graph/state_indices"):
        state_indices = np.searchsorted(nbd_states, states)
        mapped_row_indices = state_indices[nonzero_row]

    # Create two COO matrices: one for the original non-zero elements, and one for the transposed elements.

    with stopwatch("vmc_amplitude/nbd_matrix_to_graph/prepare_coo_data"):
        data = np.ones(shape=(2 * n_nonzero,), dtype=np.uint8)
        full_row_indices = np.concatenate([mapped_row_indices, nonzero_col])
        full_col_indices = np.concatenate([nonzero_col, mapped_row_indices])

    # TODO: sort ?

    # Add the two COO matrices and convert to a CSR matrix for efficient arithmetic operations.
    with stopwatch("vmc_amplitude/nbd_matrix_to_graph/symmetric_matrix"):
        symmetric_matrix = csr_matrix(
            (data, (full_row_indices, full_col_indices)), shape=(len(nbd_states), len(nbd_states))
        )

    # with stopwatch("vmc_amplitude/DEBUG/nbd_matrix_to_graph/half-symmetric-matrix1"):
    #     half_symmetric_matrix = csr_matrix(
    #         (np.ones_like(mapped_row_indices), (mapped_row_indices, nonzero_col)),
    #         shape=(len(nbd_states), len(nbd_states)),
    #     )

    # with stopwatch("vmc_amplitude/DEBUG/nbd_matrix_to_graph/half-symmetric-matrix2"):
    #     half_symmetric_matrix = csr_matrix(
    #         (np.ones_like(mapped_row_indices), (nonzero_col, mapped_row_indices)),
    #         shape=(len(nbd_states), len(nbd_states)),
    #     )

    # Threshold at 0.
    with stopwatch("vmc_amplitude/nbd_matrix_to_graph/graph_matrix"):
        graph_matrix = (symmetric_matrix != 0).astype(np.uint8)

    return graph_matrix


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
    with stopwatch("vmc_amplitude/transfer_signs_to_H/connected_components"):
        _, labels = connected_components(graph, directed=False)

    relsigns = np.empty(len(nbd_states), dtype=np.int8)
    with stopwatch("vmc_amplitude/transfer_signs_to_H/relsigns"):
        for component in np.unique(labels):
            component_mask = labels == component
            cluster = nbd_states[component_mask]
            relsigns[component_mask] = relsign_fn(cluster)

    with stopwatch("vmc_amplitude/transfer_signs_to_H/state_indices"):
        state_indices = np.searchsorted(nbd_states, states)

    with stopwatch("vmc_amplitude/transfer_signs_to_H/M_with_signs"):
        M_with_signs = diags(relsigns[state_indices]) @ M @ diags(relsigns)
        # TODO: optimize

    return M_with_signs


def safe_exp_numpy(x: npt.NDArray, normalise: bool = False) -> npt.NDArray:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - x.max()
    np.exp(x, out=x)
    if normalise:
        x /= x.sum()
    return x


def compute_log_local_energies(
    hamiltonian: ls.Operator,
    states: npt.NDArray[np.uint64],
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    log_prob_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    override_nbd_states: npt.NDArray[np.uint64] | None = None,
    override_state_indices: npt.NDArray[np.int64] | None = None,
    override_nbd_matrix: csr_matrix | None = None,
    override_nbd_matrix_w_signs: csr_matrix | None = None,
) -> tuple[
    npt.NDArray[np.complex128],  # logarithms local_energies
    npt.NDArray[np.uint64],  # nbd_states
    npt.NDArray[np.int64],  # state_indices
    csr_matrix | None,  # nbd_matrix
    csr_matrix,  # nbd_matrix_w_signs
]:
    """
    Computes the logarithms of local energies of the given states.

    Returns:
        log_local_energies: The logarithms (possibly complex) of local energies of the states.
        nbd_states: The states that are used to compute the local energies.
        state_indices: The indices of the states in nbd_states.
        nbd_matrix: The matrix that is used to compute the local energies.
        nbd_matrix_w_signs: The matrix that is used to compute the local energies,
            with the signs of the states transferred to the matrix.
    """
    if override_nbd_states is None:
        nbd_matrix, nbd_states = find_nbd(hamiltonian, states)
    else:
        nbd_states = override_nbd_states
        nbd_matrix = None

    if override_nbd_matrix is not None:
        nbd_matrix = override_nbd_matrix
    if override_state_indices is None:
        state_indices = np.searchsorted(nbd_states, states)
    else:
        state_indices = override_state_indices

    if override_nbd_matrix_w_signs is None:
        if nbd_matrix is None:
            raise ValueError(
                "If override_nbd_states is not None, at least one "
                "override_nbd_matrix or override_nbd_matrix_w_signs "
                "must be provided."
            )
        nbd_matrix_w_signs = transfer_signs_to_H(states, nbd_matrix, nbd_states, relsigns_fn)
    else:
        nbd_matrix_w_signs = override_nbd_matrix_w_signs

    with stopwatch("vmc_amplitude/compute_local_energies/abs_psi_nbd"):
        log_abs_psi_nbd = log_prob_fn(nbd_states) * 0.5
        log_abs_psi_states = log_abs_psi_nbd[state_indices]
        abs_psi_nbd = safe_exp_numpy(log_abs_psi_nbd)
    # print(f"{log_abs_psi_nbd.min()=}, {log_abs_psi_nbd.max()=}")
    # print(f"{abs_psi_nbd.min()=}, {abs_psi_nbd.max()=}")
    with stopwatch("vmc_amplitude/compute_local_energies/local_energies"):
        log_local_energies = (
            np.log((nbd_matrix_w_signs @ abs_psi_nbd).astype(np.complex128))
            - (log_abs_psi_states)
            + log_abs_psi_nbd.max()
            # safe_exp subtracted log_abs_psi_states.max(), we have to compensate
        ).astype(np.complex128)
    return log_local_energies, nbd_states, state_indices, nbd_matrix, nbd_matrix_w_signs


def compute_local_energies_reference(
    hamiltonian: ls.Operator,
    states: npt.NDArray[np.uint64],
    relsigns_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.int8]],
    log_prob_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    nbd_matrix, nbd_states = find_nbd_reference(hamiltonian, states)
    nbd_matrix_w_signs = transfer_signs_to_H(states, nbd_matrix, nbd_states, relsigns_fn)
    abs_psi_nbd = safe_exp_numpy(log_prob_fn(nbd_states) * 0.5)
    states_indices = np.searchsorted(nbd_states, states)
    abs_psi_states = abs_psi_nbd[states_indices]
    return nbd_matrix_w_signs @ abs_psi_nbd / abs_psi_states
