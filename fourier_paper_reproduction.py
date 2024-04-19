from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from loguru import logger

from fourier_supervised_cleanroom import (
    hadamard_transform,
    keep_largest_n,
    thresholded_sign,
)
from spin_systems import SpinSystem
from spin_lattices import SpinLattice


def how_many_terms_to_achieve(
    series_coeffs: npt.NDArray[np.float64],
    target_score: float,
    scorer: Callable[[npt.NDArray], np.floating],
):
    """
    Returns the number of terms needed to achieve a given score.

    Parameters
    ----------

    series_coeffs: The series of coefficients of a Fourier series.
    target_score: The score to achieve.
    scorer: A function that takes a Fourier series
        and returns a score.
    """
    max_terms = len(series_coeffs)
    min_terms = 0
    if (score := scorer(series_coeffs)) < target_score:
        logger.debug(f"The series is already below ({score=}) the {target_score=}.")
        return max_terms

    while max_terms - min_terms > 1:
        mid_terms = (min_terms + max_terms) // 2
        if (score := scorer(keep_largest_n(series_coeffs, mid_terms))) < target_score:
            logger.debug(f"Score {score} with {mid_terms} terms is below target score.")
            min_terms = mid_terms
        else:
            logger.debug(f"Score {score} with {mid_terms} terms is above target score.")
            max_terms = mid_terms
    return max_terms


def how_many_terms_to_achieve_relative_weight(
    coeffs: npt.NDArray[np.float64], target_weight: float
) -> int:
    weights = coeffs**2
    total_weight = np.sum(weights)
    relative_weights = weights / total_weight
    sorted_relative_weights = np.sort(relative_weights)[::-1]
    cumulative_weights = np.cumsum(sorted_relative_weights)
    terms = np.searchsorted(cumulative_weights, target_weight, side="left") + 1
    return int(terms)


def rel_fourier_weight_in_largest_terms(series: npt.NDArray, terms: int) -> float:
    truncated_series = keep_largest_n(series, terms)
    return float(np.sum(truncated_series**2) / np.sum(series**2))


def sign_overlap(
    system: SpinSystem,
    signal_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    states: npt.NDArray[np.uint64] | None = None,
):
    if states is None:
        states = system.canonical_basis.states

    ground_truth = np.sign(signal_fn(states))
    probs = system.get_ground_state_coeffs(states) ** 2

    def wrapper(fourier_series: npt.NDArray[np.float64]):
        predictions = np.sign(hadamard_transform(fourier_series)[states])
        return np.sum(ground_truth * predictions * probs) / np.sum(probs)

    return wrapper


def accuracy(
    system: SpinSystem,
    signal_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    states: npt.NDArray[np.uint64] | None = None,
    tol=0.0,
):
    if states is None:
        states = system.canonical_basis.states

    ground_truth = np.sign(signal_fn(states))

    def wrapper(fourier_series: npt.NDArray[np.float64]):
        predictions = thresholded_sign(
            hadamard_transform(fourier_series)[states], tol=tol
        )
        non_zero_terms = (predictions != 0) & (ground_truth != 0)
        return np.mean(ground_truth[non_zero_terms] == predictions[non_zero_terms])

    return wrapper


def get_ipr(series: npt.NDArray[np.float64]):
    return np.sum(series**4) / np.sum(series**2) ** 2


def visualize_coeffs(coeffs, n: int, lat: SpinLattice, up_to_symmetry=False, **kwargs):
    coeffs = coeffs.copy()
    coeffs /= np.sqrt((coeffs**2).sum())
    if up_to_symmetry:
        fourier_basis_data = lat.get_fourier_basis_data()
        selected_coeffs = coeffs[fourier_basis_data.reprs]
    else:
        selected_coeffs = coeffs
    sort_order = np.argsort(np.abs(selected_coeffs))[::-1]

    if up_to_symmetry:
        sorted_subsets = fourier_basis_data.reprs[sort_order][:n]
    else:
        sorted_subsets = sort_order[:n].astype(np.uint64)

    sorted_coeffs = selected_coeffs[sort_order][:n]
    titles = [f"${c:.4f}$" for c in sorted_coeffs]
    print(sorted_coeffs)
    return lat.plot_subsets(sorted_subsets, titles, legend=False, **kwargs)
