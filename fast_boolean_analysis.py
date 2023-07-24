from collections.abc import Callable, Iterable
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from lattice_boolean_analysis import (
    LatticeBooleanFunction,
    ScorerType,
    SignalKind,
    get_scorer,
)
from loguru import logger
from parity import popcount
from misc_utils import hadamard_transform_pytorch_inplace

TruncateStrategy = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]


def keep_largest_n(n: int) -> TruncateStrategy:
    def truncate(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        signal_abs = np.abs(signal)
        out = np.zeros_like(signal_abs, dtype=np.bool_)
        out[np.argpartition(signal_abs, -n)[-n:]] = True
        return out

    return truncate


def keep_everything(s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return np.ones_like(s, dtype=np.bool_)


class FourierSeries:
    def __init__(
        self,
        signal: LatticeBooleanFunction,
        coeffs: npt.NDArray[np.float64],
    ):
        self.signal = signal
        self.coeffs = coeffs

    def predict(self, x: npt.NDArray[np.uint64] | None = None) -> npt.NDArray:
        if x is None:
            x = self.signal.canonical_basis.states
        return hadamard_transform_pytorch_inplace(
            torch.tensor(self.coeffs.copy(), dtype=torch.float64)
        ).numpy()[x]

    def __call__(self, x: npt.NDArray[np.uint64] | None = None) -> npt.NDArray:
        return self.predict(x)

    def truncate(self, strategy: TruncateStrategy) -> "FourierSeries":
        keep_mask = strategy(self.coeffs)
        truncated_coeffs = np.where(keep_mask, self.coeffs, 0)
        return FourierSeries(signal=self.signal, coeffs=truncated_coeffs)

    def truncate_orbitwise(self, strategy: TruncateStrategy) -> "FourierSeries":
        fourier_basis_data = self.signal.lattice.get_fourier_basis_data()
        repr_coeffs = self.coeffs[fourier_basis_data.reprs]

        keep_mask = strategy(repr_coeffs)
        # keep_mask is a boolean array whose index corresponds to the index of the
        # fourier basis states. We need to map this to the index of the long array
        # with coeffs

        long_keep_mask = keep_mask[fourier_basis_data.bits_to_repr_index]

        truncated_coeffs = np.where(long_keep_mask, self.coeffs, 0)

        return FourierSeries(signal=self.signal, coeffs=truncated_coeffs)

    def prediction_score(
        self,
        scorer: str | ScorerType,
        x: npt.NDArray[np.uint64] | None = None,
        prediction: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        if x is None:
            x = self.signal.canonical_basis.states

        signal = self.signal(x)
        prob = self.signal.get_probs(x)
        if prediction is None:
            prediction = self.predict(x)
        score = get_scorer(scorer)(signal, prediction, prob)

        return score, prediction

    def how_many_terms_to_achieve_relative_weight(self, target_weight: float) -> int:
        weights = self.coeffs**2
        total_weight = np.sum(weights)
        relative_weights = weights / total_weight
        sorted_relative_weights = np.sort(relative_weights)[::-1]
        cumulative_weights = np.cumsum(sorted_relative_weights)
        terms = np.argmax(cumulative_weights >= target_weight) + 1
        return int(terms)

    def how_many_terms_to_achieve_score(
        self,
        target_score: float,
        scorer: str | ScorerType,
        x: npt.NDArray[np.uint64] | None = None,
        min_terms: int = 1,
        max_terms: int | None = None,
        orbitwise: bool = False,
    ) -> tuple[bool, int, npt.NDArray[np.float64]]:
        """
        Find the number of terms needed to achieve a target score.

        Parameters:
        -----------
        target_score: float
            The target score to achieve
        scorer: str or ScorerType
            The scorer to use
        min_terms: int
            The minimum number of terms to consider
        max_terms: int or None
            The maximum number of terms to consider. If None, use the number of terms
            in the signal
        orbitwise: bool
            Whether to truncate orbitwise or not

        Returns:
        --------
        success: bool
            Whether the target score was achieved
        terms: int
            The number of terms needed to achieve the target score
        prediction: np.ndarray
            The prediction for the signal achieved with the number of terms returned

        """

        def truncate(max_terms):
            if orbitwise:
                return self.truncate_orbitwise(keep_largest_n(max_terms))
            else:
                return self.truncate(keep_largest_n(max_terms))

        if max_terms is None:
            if orbitwise:
                max_terms = len(self.signal.lattice.get_fourier_basis_data().reprs)
            else:
                max_terms = len(self.coeffs)

        score, max_prediction = truncate(max_terms).prediction_score(scorer, x)
        if score < target_score:
            logger.debug(
                f"At {max_terms=}, score={score} < target_score={target_score}, so we can't achieve the target score"
            )
            return False, max_terms, max_prediction

        score, prediction = truncate(min_terms).prediction_score(scorer, x)
        if score >= target_score:
            logger.debug(
                f"At {min_terms=}, score={score} >= target_score={target_score}, so we can achieve the target score with {min_terms} terms"
            )
            return True, min_terms, prediction

        while max_terms - min_terms > 1:
            logger.debug(f"min_terms={min_terms}, max_terms={max_terms}")
            mid = (max_terms + min_terms) // 2
            score, prediction = truncate(mid).prediction_score(scorer, x)
            logger.debug(f"{mid=}, score={score}")

            if score >= target_score:
                logger.debug("score >= target_score, so we can decrease max_terms")
                max_terms = mid
                max_prediction = prediction
                # this is biased towards the lower end, but that's fine
            else:
                logger.debug("score < target_score, so we can increase min_terms")
                min_terms = mid

        return True, max_terms, max_prediction

    def total_hamming_weight(self, terms: int) -> int:
        """
        Returns total hamming weight of the terms with the largest coefficients.
        Hamming weight is measures up to the spin inversion symmetry, so the
        hamming weight of a term is the minimum of the hamming weight and the
        hamming weight of the spin inversion of the term.

        Parameters:
        -----------
        terms: int
            The number of terms to consider

        Returns:
        --------
        total_hamming_weight: int
            The total hamming weight of the terms with the largest coefficients
        """
        popcounts = popcount(
            np.asarray(np.argpartition(np.abs(self.coeffs), -terms)[-terms:], dtype="uint64")
        )
        inv_popcounts = self.signal.number_spins - popcounts
        return int(np.sum(np.minimum(popcounts, inv_popcounts)))

    def ipr(self, hamming_weighted: bool = False, ignore_free_term: bool = False) -> float:
        """
        Returns the inverse participation ratio of the Fourier series.

        Parameters:
        -----------
        hamming_weighted: bool
            Whether to weight the IPR by the hamming weight of the terms

        Returns:
        --------
        ipr: float
            The inverse participation ratio of the Fourier series
        """
        coeffs = self.coeffs.copy()

        if ignore_free_term:
            indexes = np.arange(len(self.coeffs), dtype="uint64")
            popcounts = popcount(indexes)
            free_terms = (popcounts == 0) | (popcounts == self.signal.number_spins)
            coeffs[free_terms] = 0

        return get_ipr(coeffs, hamming_weighted)

    @staticmethod
    def from_signal(signal: LatticeBooleanFunction) -> "FourierSeries":
        """
        Returns the Fourier series of a signal.

        Parameters:
        -----------
        signal: LatticeBooleanFunction
            The signal to find the Fourier series of

        Returns:
        --------
        fourier_series: FourierSeries
            The Fourier series of the signal
        """
        return fourier_expand(signal)

    @staticmethod
    def from_representatives_coeffs(
        signal: LatticeBooleanFunction, coeffs: npt.NDArray[np.float64]
    ) -> "FourierSeries":
        """
        Extends a Fourier series given by its coefficients on the representatives
        using the symmetries.

        Note that signal is used only to keep information about the lattice and
        number of spins, and is not used to compute the Fourier series.

        It is expected that signal is invariant under the symmetries of the
        lattice, and is non-zero only on the sector of zero magnetization

        Parameters:
        -----------
        signal: LatticeBooleanFunction
            The signal that corresponds to the Fourier series

        coeffs: np.ndarray
            The coefficients of the Fourier series,
            i'th element is the coefficient of the i'th representative

        Returns:
        --------
        fourier_series: FourierSeries
            The Fourier series
        """
        basis_data = signal.lattice.get_fourier_basis_data()
        full_coeffs = coeffs[basis_data.bits_to_repr_index] * basis_data.bits_to_char
        return FourierSeries(signal, full_coeffs)


def get_ipr(coeffs: npt.NDArray[np.float64], hamming_weighted: bool = False) -> float:
    number_spins = int(np.log2(len(coeffs)))
    if len(coeffs) != 2**number_spins:
        raise ValueError("coeffs must be a power of 2 in length")

    weights = coeffs**2
    weights /= weights.sum()

    if not hamming_weighted:
        return (weights**2).sum()

    popcounts = popcount(np.arange(2**number_spins, dtype="uint64"))
    inv_popcounts = number_spins - popcounts
    hamming_ipr = (weights**2 / (np.minimum(popcounts, inv_popcounts) + 1)).sum()
    return hamming_ipr


def fourier_expand(signal: LatticeBooleanFunction) -> FourierSeries:
    x = signal.canonical_basis.states
    logger.debug("Finding signal")
    signal_value = signal.as_long_array(x).copy()
    logger.debug("Doing ")
    return FourierSeries(
        signal=signal,
        coeffs=hadamard_transform_pytorch_inplace(
            torch.tensor(signal_value, dtype=torch.float64)
        ).numpy(),
    )
