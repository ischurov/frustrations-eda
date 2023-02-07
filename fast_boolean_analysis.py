from collections.abc import Callable, Iterable
from typing import overload

import numpy as np
import numpy.typing as npt
from lattice_boolean_analysis import LatticeBooleanFunction, ScorerType, SignalKind, get_scorer
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
        verbose: bool = False,
    ):
        self.signal = signal
        self.coeffs = coeffs
        self.verbose = verbose

    def predict(self) -> npt.NDArray:
        return hadamard_transform(self.coeffs, verbose=self.verbose)[
            self.signal.canonical_basis.states
        ]

    def truncate(self, strategy: TruncateStrategy) -> "FourierSeries":
        fourier_basis_data = self.signal.lattice.get_fourier_repr()
        repr_coeffs = self.coeffs[fourier_basis_data.reprs]

        keep_mask = strategy(repr_coeffs)
        # keep_mask is a boolean array whose index corresponds to the index of the
        # fourier basis states. We need to map this to the index of the long array
        # with coeffs

        long_keep_mask = keep_mask[fourier_basis_data.bits_to_repr_index]

        truncated_coeffs = np.where(long_keep_mask, self.coeffs, 0)

        return FourierSeries(signal=self.signal, coeffs=truncated_coeffs)

    @overload
    def prediction_score(
        self, scorer: str | ScorerType, x: npt.NDArray[np.uint64] | None = None
    ) -> float:
        ...

    @overload
    def prediction_score(
        self,
        scorer: list[str | ScorerType] | tuple[str | ScorerType],
        x: npt.NDArray[np.uint64] | None = None,
    ) -> dict[str, float]:
        ...

    def prediction_score(
        self,
        scorer: str | ScorerType | list[str | ScorerType] | tuple[str | ScorerType],
        x: npt.NDArray[np.uint64] | None = None,
    ) -> float | dict[str, float]:

        if x is None:
            x = self.signal.canonical_basis.states
        else:
            raise NotImplementedError(
                "Only score over the whole domain (x=None) is supported for now"
            )

        signal = self.signal(x)
        prob = self.signal.get_probs(x)

        prediction = self.predict()  # FIXME: should take only part that corresponds to x

        if isinstance(scorer, Iterable) and not isinstance(scorer, str):
            return {str(s): get_scorer(s)(signal, prediction, prob) for s in scorer}

        return get_scorer(scorer)(signal, prediction, prob)

    def how_many_terms_to_achieve_score(
        self,
        target_score: float,
        scorer: str | ScorerType,
        min_terms: int = 1,
        max_terms: int | None = None,
    ) -> int | None:

        if max_terms is None:
            max_terms = len(self.signal.lattice.get_fourier_repr().reprs)

        if self.truncate(keep_largest_n(max_terms)).prediction_score(scorer) < target_score:
            return None

        if self.truncate(keep_largest_n(min_terms)).prediction_score(scorer) >= target_score:
            return min_terms

        while max_terms - min_terms > 1:
            mid = (max_terms + min_terms) // 2
            if self.truncate(keep_largest_n(mid)).prediction_score(scorer) >= target_score:
                max_terms = mid
            else:
                min_terms = mid

        return max_terms


def fourier_expand(signal: LatticeBooleanFunction, verbose: bool = False) -> FourierSeries:
    x = signal.canonical_basis.states
    if verbose:
        print("Calculating signal value")
    signal_value = signal.as_long_array(x)
    return FourierSeries(signal=signal, coeffs=hadamard_transform(signal_value, verbose=verbose))
