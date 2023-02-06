from collections.abc import Callable

import numpy as np
import numpy.typing as npt

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


class FBATruncation:
    def __init__(self, analyzer: "FastBooleanAnalyzer", strategy: TruncateStrategy):
        self.analyzer = analyzer
        self.strategy = strategy

    def predict(self, x: npt.NDArray[np.uint64] | None) -> npt.NDArray:
        if x is None:
            x = self.analyzer.signal.canonical_basis.states

        repr_coeffs = self.analyzer.coeffs_[self.analyzer.signal.lattice.fourier_basis.states]
        keep_mask = self.strategy(repr_coeffs)
        # keep_mask is a boolean array whose index corresponds to the index of the
        # fourier basis states. We need to map this to the index of the long array
        # with coeffs

        long_keep_mask = keep_mask[self.analyzer.signal.lattice.make_fourier_repr()["indices"]]

        coeffs = np.where(long_keep_mask, self.analyzer.coeffs_, 0)
        return hadamard_transform(coeffs)[x]


class FastBooleanAnalyzer:
    def __init__(self, signal: LatticeBooleanFunction):
        self.signal = signal

    def fit(self, x: npt.NDArray[np.uint64] | None) -> "FastBooleanAnalyzer":
        if x is None:
            x = self.signal.canonical_basis.states
        signal = self.signal.long_array(x)
        self.coeffs_ = hadamard_transform(signal)
        return self

    def _ensure_fitted(self):
        if not hasattr(self, "coeffs_"):
            raise ValueError("Call fit() before truncate()")

    def truncate(self, strategy: TruncateStrategy) -> FBATruncation:
        self._ensure_fitted()

        return FBATruncation(self, strategy)
