from collections.abc import Callable, Iterable
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
from loguru import logger

from lattice_boolean_analysis import (
    LatticeBooleanFunction,
    ScorerType,
    SignalKind,
    get_scorer,
)
from utils import hadamard_transform

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

    def predict(self) -> npt.NDArray:
        return hadamard_transform(self.coeffs)[self.signal.canonical_basis.states]

    def truncate(self, strategy: TruncateStrategy) -> "FourierSeries":
        keep_mask = strategy(self.coeffs)
        truncated_coeffs = np.where(keep_mask, self.coeffs, 0)
        return FourierSeries(signal=self.signal, coeffs=truncated_coeffs)

    def truncate_orbitwise(self, strategy: TruncateStrategy) -> "FourierSeries":
        fourier_basis_data = self.signal.lattice.get_fourier_repr()
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
        else:
            raise NotImplementedError(
                "Only score over the whole domain (x=None) is supported for now"
            )

        signal = self.signal(x)
        prob = self.signal.get_probs(x)
        if prediction is None:
            prediction = self.predict()  # FIXME: should take only part that corresponds to x
        score = get_scorer(scorer)(signal, prediction, prob)

        return score, prediction

    def how_many_terms_to_achieve_score(
        self,
        target_score: float,
        scorer: str | ScorerType,
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
                max_terms = len(self.signal.lattice.get_fourier_repr().reprs)
            else:
                max_terms = len(self.coeffs)

        score, prediction = truncate(max_terms).prediction_score(scorer)
        if score < target_score:
            logger.debug(
                f"At {max_terms=}, score={score} < target_score={target_score}, so we can't achieve the target score"
            )
            return False, max_terms, prediction

        score, prediction = truncate(min_terms).prediction_score(scorer)
        if score >= target_score:
            logger.debug(
                f"At {min_terms=}, score={score} >= target_score={target_score}, so we can achieve the target score with {min_terms} terms"
            )
            return True, min_terms, prediction

        while max_terms - min_terms > 1:
            logger.debug(f"min_terms={min_terms}, max_terms={max_terms}")
            mid = (max_terms + min_terms) // 2
            score, prediction = truncate(mid).prediction_score(scorer)
            logger.debug(f"{mid=}, score={score}")

            if score >= target_score:
                logger.debug("score >= target_score, so we can decrease max_terms")
                max_terms = mid
                # this is biased towards the lower end, but that's fine
            else:
                logger.debug("score < target_score, so we can increase min_terms")
                min_terms = mid

        return True, max_terms, prediction


def fourier_expand(signal: LatticeBooleanFunction, verbose: bool = False) -> FourierSeries:
    x = signal.canonical_basis.states
    if verbose:
        print("Calculating signal value")
    signal_value = signal.as_long_array(x)
    return FourierSeries(signal=signal, coeffs=hadamard_transform(signal_value, verbose=verbose))
