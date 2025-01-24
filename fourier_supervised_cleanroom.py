from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from misc_utils import (
    hadamard_transform_pytorch_inplace,
    make_packed_configurations,
    make_unpacked_configurations,
)
from spin_systems import SpinSystem


### FROM: GPT-4
def apply_random_permutations(
    states: npt.NDArray, permutations: npt.NDArray | list[list[int]]
):
    states_torch = torch.from_numpy(states.astype(np.int64))
    permutations_torch = torch.tensor(permutations).to(torch.int64)
    random_indices = torch.randint(0, len(permutations), (len(states),))
    chosen_permutations = permutations_torch[random_indices]
    permuted_states = torch.gather(states_torch, 1, chosen_permutations)
    return permuted_states.numpy()


### END FROM


def keep_largest_n(coeffs: npt.NDArray, n: int, inplace=False) -> npt.NDArray:
    """
    For a given coeffs array, keep the largest (by absolute value) n elements and
    set the rest to zero.
    """

    if not inplace:
        coeffs = coeffs.copy()

    if n <= 0:
        coeffs[:] = 0
        return coeffs

    if n >= len(coeffs):
        return coeffs

    abs_signal = np.abs(coeffs)
    n_elements_to_remove = len(coeffs) - n
    remove_idxs = np.argpartition(abs_signal, n_elements_to_remove)
    coeffs[remove_idxs[:n_elements_to_remove]] = 0
    return coeffs


def keep_random_n(coeffs: npt.NDArray, n: int, inplace=False) -> npt.NDArray:
    """
    For a given coeffs array, keep n random elements and set the rest to zero.
    """

    if not inplace:
        coeffs = coeffs.copy()

    if n <= 0:
        coeffs[:] = 0
        return coeffs

    if n >= len(coeffs):
        return coeffs

    remove_idxs = np.random.choice(len(coeffs), size=len(coeffs) - n, replace=False)
    coeffs[remove_idxs] = 0
    return coeffs


def keep_fourier_weight_inplace(coeffs: npt.NDArray, weight: float) -> npt.NDArray:
    weights = coeffs**2
    weights /= np.sum(weights)
    sort_order = np.argsort(weights)[::-1]
    cumsum = np.cumsum(weights[sort_order])
    remove_idxs = sort_order[np.where(cumsum > weight)[0][1:]]
    coeffs[remove_idxs] = 0
    return coeffs


def how_many_terms_to_keep_fourier_weight(coeffs: npt.NDArray, weight: float) -> int:
    weights = coeffs**2
    total_weight = np.sum(weights)
    relative_weights = weights / total_weight
    sorted_relative_weights = np.sort(relative_weights)[::-1]
    cumulative_weights = np.cumsum(sorted_relative_weights)
    terms = np.searchsorted(cumulative_weights, weight) + 1
    return int(terms)


def kept_fourier_weight(coeffs: npt.NDArray, n: int) -> np.float64:
    new_coeffs = coeffs.copy()
    keep_largest_n(new_coeffs, n, inplace=True)
    return np.sum(new_coeffs**2) / np.sum(coeffs**2)


def hadamard_transform(signal: npt.NDArray, inplace=False) -> npt.NDArray:
    """
    Perform the Hadamard transform on a signal.
    """
    if not inplace:
        signal = signal.copy()
    output = hadamard_transform_pytorch_inplace(torch.from_numpy(signal)).numpy()
    return output


def fit_fourier_series(
    states: npt.NDArray[np.uint64],
    signal_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    n_bits: int,
) -> npt.NDArray[np.float64]:
    signal = np.zeros(2**n_bits, dtype=np.float64)
    np.add.at(signal, states, signal_fn(states))
    coeffs = hadamard_transform(signal, inplace=True)
    return coeffs


def sample_from_system(
    system: SpinSystem,
    n_samples: int,
    states_to_sample_from: npt.NDArray[np.uint64] | None = None,
    sampling_power: float = 2.0,
    replace: bool = False,
):
    if states_to_sample_from is None:
        states_to_sample_from = system.basis.states
    logger.debug("Finding probs")

    if sampling_power == 0:
        probs = np.ones(len(states_to_sample_from)) / len(states_to_sample_from)
    else:
        amplitudes = np.abs(
            system.get_ground_state_coeffs(
                states_to_sample_from, apply_symmetries=False
            )
        )
        probs = amplitudes**sampling_power

    if sampling_power != 2:
        # correcting probs to take into account that different
        # representatives have different orbit sizes
        _, _, norms = system.state_info(states_to_sample_from)
        probs *= norms ** (sampling_power - 2)

    probs /= probs.sum()

    logger.debug("Doing np.random.choice")
    sampled_states = np.random.choice(
        states_to_sample_from,
        size=n_samples,
        p=probs,
        replace=replace,
    )
    return sampled_states


def median_split(arr: npt.NDArray) -> npt.NDArray[np.bool_]:
    if arr.shape[0] == 0:
        return np.array([], dtype=bool)
    cumsum = np.cumsum(arr)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2, side="right")
    return np.arange(len(arr)) < median_idx


def median_split_recursively(arr: npt.NDArray, steps: int) -> npt.NDArray[np.bool_]:
    splitting = median_split(arr)
    if steps == 1:
        return splitting.reshape(1, -1)
    return np.vstack(
        [
            splitting,
            np.hstack(
                [
                    median_split_recursively(arr[splitting], steps - 1),
                    median_split_recursively(arr[~splitting], steps - 1),
                ]
            ),
        ]
    )


def amplitude_prob_median_bin_signal(system: SpinSystem, bit=1):
    """
    Binaries amplitude by splitting the amplitudes in half at the median of the
    corresponding probability distribution.

    Expected value of this binarization w.r.t to sampling from the corresponding
    probability distribution (psi^2) is 0.

    If bit > 1, applies the binarization recursively.
    """
    amplitude = np.abs(system.get_ground_state_in_canonical_basis())
    sorted_idxs = np.argsort(amplitude)
    amplitude_sorted = amplitude[sorted_idxs]
    prob_sorted = amplitude_sorted**2
    bits_sorted_order = median_split_recursively(prob_sorted, bit)[-1]
    bits_initial_order = np.zeros_like(bits_sorted_order)
    bits_initial_order[sorted_idxs] = bits_sorted_order

    def wrapper(s):
        return (1 - 2 * bits_initial_order[system.canonical_basis.index(s)]).astype(
            np.float64
        )

    return wrapper


def thresholded_sign(x: npt.NDArray, tol=0.0) -> npt.NDArray[np.float64]:
    return np.where(np.abs(x) < tol, 0, np.sign(x))


def amplitude_median_bin_signal(system: SpinSystem, tol=0.0):
    """
    Binarizes amplitude by splitting the amplitudes in half at the median.
    Half of the amplitudes are set to 1 and the other half to -1.
    Expected value of this binarization w.r.t to uniform (!) sampling is 0.
    """
    median = np.median(np.abs(system.get_ground_state_in_canonical_basis()))

    def wrapper(s):
        return thresholded_sign(
            np.abs(system.get_ground_state_coeffs(s)) - median, tol=tol
        )

    return wrapper


def amplitude_signal(system: SpinSystem):
    def wrapper(s):
        return np.abs(system.get_ground_state_coeffs(s)).astype(np.float64)

    return wrapper


def do_apply_random_symmetries(reprs: npt.NDArray[np.uint64], system: SpinSystem):
    logger.debug("Unpacking")
    reprs_unpacked = system.lattice.unpack_configurations(
        reprs,
    )
    logger.debug("Applying random permutations")
    states_unpacked = apply_random_permutations(
        reprs_unpacked, system.lattice.get_automorphisms()
    )

    logger.debug("Packing")
    states = system.lattice.pack_configurations(
        states_unpacked,
    )

    return states


def mk_train_test(
    system: SpinSystem,
    n_train: int,
    n_test: int,
    sampling_power_train: float = 2,
    sampling_power_test: float = 0,
    replace: bool = False,
    apply_random_symmetries: bool = False,
):
    logger.debug("Sampling train")
    train_states = sample_from_system(
        system,
        n_train,
        sampling_power=sampling_power_train,
        replace=replace,
    )
    logger.debug("Sampling test")

    test_states = sample_from_system(
        system,
        n_test,
        sampling_power=sampling_power_test,
        replace=replace,
    )

    if apply_random_symmetries:
        logger.debug("Applying random symmetries to train")
        train_states = do_apply_random_symmetries(train_states, system)

        logger.debug("Applying random symmetries to test")
        test_states = do_apply_random_symmetries(test_states, system)

    test_states = np.setdiff1d(test_states, train_states)

    return train_states, test_states


def sign_signal(system: SpinSystem, tol=0.0, apply_symmetries=True):
    def wrapper(s):
        ground_state = system.get_ground_state_coeffs(
            s, apply_symmetries=apply_symmetries
        )
        return thresholded_sign(ground_state, tol=tol)

    return wrapper


def ground_state_signal(system: SpinSystem):
    def wrapper(s):
        return system.get_ground_state_coeffs(s).astype(np.float64)

    return wrapper
