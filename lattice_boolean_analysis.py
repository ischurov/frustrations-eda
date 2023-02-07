import hashlib
import io
import json
import lzma
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, overload

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from hadamard_transform import hadamard_transform
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from boolean_fourier_learner import BooleanFourierLearner
from heisenberg_hamiltonians import (
    SpinSystem,
    batched_state_info_df,
    make_unpacked_configurations,
)
from parity import calculate_fourier_transform_matrix, parity, popcount
from spin_lattices import SpinLattice


def camel_case_to_snake_case(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")


class SignalKind:
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        This function defines how to transform data during the training.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return camel_case_to_snake_case(self.__class__.__name__.removesuffix("SignalKind"))


class ValueSignalKind(SignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return eigenstate_coeff


class SignSignalKind(SignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sign(eigenstate_coeff)


class AmplitudeSignalKind(SignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.abs(eigenstate_coeff)


class ProbSignalKind(SignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (np.abs(eigenstate_coeff)) ** 2  # type: ignore
        # see https://github.com/numpy/numpy/issues/20099


class AmplitudeMedianBinSignalKind(SignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        abs_ = np.abs(eigenstate_coeff)
        return np.sign(abs_ - np.median(abs_))


def place_values_into_array(
    x: npt.NDArray[np.uint64], y: npt.NDArray[np.float64], number_spins: int
) -> npt.NDArray[np.float64]:
    """Place values into an array of zeros.

    Args:
        x: Indices of the array to place the values into.
        y: Values to place into the array.
        number_spins: The number of spins in the system.

    Returns:
        An array of zeros with the given values placed into it.
    """
    result = np.zeros(2**number_spins, dtype=np.float64)
    result[x] = y
    return result


class LatticeBooleanFunction:
    def __init__(self, lattice: SpinLattice, canonical_basis: ls.SpinBasis):
        self.lattice = lattice
        self.number_spins = lattice.number_spins
        self.canonical_basis = canonical_basis

    def __call__(self, x: npt.NDArray[np.uint64]) -> npt.NDArray:
        raise NotImplementedError

    def get_probs(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def get_cache_id(self) -> str:
        raise NotImplementedError

    def as_long_array(self, x: npt.NDArray[np.uint64]) -> npt.NDArray:
        return place_values_into_array(x, self(x), self.number_spins)


class LBFFromSpinSystem(LatticeBooleanFunction):
    def __init__(
        self, system: SpinSystem, eigenstate: int = 0, kind: SignalKind = SignSignalKind()
    ):
        system.get_eigenstates(eigenstate + 1)
        super().__init__(system.lattice, canonical_basis=system.canonical_basis)
        self.system = system
        self.system.get_eigenstates(eigenstate + 1)
        self.eigenstate = eigenstate
        self.kind = kind
        self.canonical_basis = system.canonical_basis

    def __call__(self, x: npt.NDArray[np.uint64]) -> npt.NDArray:
        signal_df = self.system.get_df_eigenstate(k=self.eigenstate, canonical_basis=True).loc[
            x, :
        ]  # type: ignore
        # See https://github.com/pandas-dev/pandas-stubs/issues/508

        return self.kind.transform_data(signal_df["eigenstate_coeff"].values)

    def get_probs(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
        signal_df = self.system.get_df_eigenstate(k=self.eigenstate, canonical_basis=True).loc[
            x, :
        ]  # type: ignore
        # See https://github.com/pandas-dev/pandas-stubs/issues/508

        return np.abs(signal_df["eigenstate_coeff"].values) ** 2

    def get_cache_id(self) -> str:
        return f"{self.system.get_cache_id()}-{self.eigenstate}-{self.kind.name}"


class LBFFromNN(LatticeBooleanFunction):
    def __init__(self, lattice: SpinLattice, nn: nn.Module, probs: pd.Series):
        super().__init__(
            lattice,
            canonical_basis=lattice.get_basis(
                use_symmetries=False, hamming_weight=lattice.number_spins // 2, spin_inversion=None
            ),
        )
        self.nn = nn
        self._probs = probs

    def __call__(self, x: npt.NDArray[np.uint64]) -> npt.NDArray:
        net_output = self.nn(
            torch.tensor(
                make_unpacked_configurations(x, self.lattice.number_spins).astype("float32"),
                dtype=torch.float32,
            )
        )

        return 1 - torch.max(net_output, 1)[1].detach().numpy() * 2

    def get_probs(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
        return self._probs.loc[x].values

    def get_cache_id(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.nn.state_dict(), buffer)
        id_ = hashlib.md5(
            buffer.getvalue() + json.dumps(self._probs.to_dict()).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
        return f"{self.nn.__class__.__name__}-{id_}"


TruncateStrategy = Callable[[pd.Series], pd.Series]
ScorerType = Callable[[npt.NDArray, npt.NDArray, npt.NDArray], float]

scorers: dict[str, ScorerType] = {}


def scorer(f: ScorerType):
    scorers[f.__name__.removesuffix("_scorer")] = f
    return f


@scorer
def sign_overlap_scorer(true: npt.NDArray, predict: npt.NDArray, prob: npt.NDArray):
    return (true * np.sign(predict) * prob).sum() / prob.sum()


@scorer
def value_overlap_scorer(true: npt.NDArray, predict: npt.NDArray, prob: npt.NDArray):
    return (true * predict).sum() / prob.sum()


@scorer
def accuracy_scorer(true: npt.NDArray, predict: npt.NDArray, prob: npt.NDArray):
    """Ignores prob"""
    return (true == np.sign(predict)).mean()


@scorer
def neg_mse_scorer(true: npt.NDArray, predict: npt.NDArray, prob: npt.NDArray):
    """Ignores prob"""
    return -float(np.mean((true - predict) ** 2))


@scorer
def f1_scorer(true: npt.NDArray, predict: npt.NDArray, prob: npt.NDArray):
    """Ignores prob"""
    skip = (true == 0) | (predict == 0)
    return float(f1_score(true[~skip], np.sign(predict[~skip])))


def get_scorer(scorer: str | ScorerType) -> ScorerType:
    if isinstance(scorer, str):
        if scorer in scorers:
            scorer = scorers[scorer]
        else:
            raise ValueError(
                f"scorer {scorer} not found. Available scorers: {list(scorers.keys())}"
            )
    return scorer


def keep_largest_n(n: int, offset: int = 0) -> TruncateStrategy:
    def truncate(s: pd.Series) -> pd.Series:
        return (
            s.to_frame()
            .assign(abs_value=lambda x: np.abs(x["coeff"]))
            .sort_values("abs_value", ascending=False)["coeff"]
            .iloc[offset : n + offset]
        )

    return truncate


def keep_everything(s: pd.Series) -> pd.Series:
    return s


class LBATruncation:
    def __init__(self, analyzer: "LatticeBooleanAnalyzer", truncate_strategy: TruncateStrategy):
        self.truncate_strategy = truncate_strategy
        self.analyzer = analyzer

    def get_expanded_coeffs_ser(self) -> pd.Series:
        """
        Returns the Fourier coefficients expanded with the symmetry group.
        If truncate_strategy is not `keep_everything`, the coefficients are
        truncated according to the strategy.

        The strategy is applied to the coefficients before the expansion.

        Besides the expansion, the coefficients are also normalized in such a way
        that Plancherel's theorem holds unless truncation applied.

        Returns
        -------

        pd.Series
            Series with index of the full image of all subsets under the action of
            the symmetry group, and values of the coefficients.

        """
        if self.analyzer.hadamard:
            full_coeffs = self.analyzer.learner.get_full_coeffs_ser()
        else:
            full_coeffs = (
                self.analyzer.fourier_basis_state_info_df.join(
                    self.analyzer.learner.get_coeffs_ser(), on="representative"
                )
                .dropna()
                .assign(coeff_adjusted=lambda x: x["coeff"] * x["character"])["coeff_adjusted"]
                .rename("coeff")
                * self.analyzer.canonical_basis.states.shape[0]
                / 2**self.analyzer.number_spins
            )
        print("Finding indexes to keep")
        idxs_after_truncation = (
            self.analyzer.fourier_basis_state_info_df.join(
                self.truncate_strategy(self.analyzer.learner.get_coeffs_ser()),
                on="representative",
            ).dropna()
        ).index
        print("Truncating")
        coeffs = (
            full_coeffs.loc[idxs_after_truncation]
            .to_frame()
            .assign(abs_coeff=lambda x: np.abs(x["coeff"]))
            .sort_values("abs_coeff", ascending=False)["coeff"]
        )
        return coeffs

    def _predict_hadamard(
        self, x: npt.NDArray[np.uint64], coeffs: pd.Series
    ) -> npt.NDArray[np.float64]:
        return hadamard_transform(
            torch.tensor(
                coeffs.reindex(np.arange(2**self.analyzer.number_spins, dtype="uint64"))
                .fillna(0)
                .values
            )
        ).numpy()[x] * 2 ** (self.analyzer.number_spins / 2)

    def predict(
        self,
        x: npt.NDArray[np.uint64] | None = None,
        max_batch_size: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        FIXME: there is a bug in predict when hadamard is True. Normalization
        is incorrect.
        """

        if x is None:
            x = self.analyzer.canonical_basis.states

        coeffs = self.get_expanded_coeffs_ser()

        if self.analyzer.predict_hadamard:
            print("Using hadamard transform")
            prediction = self._predict_hadamard(x, coeffs)
            print("Done")
            return prediction

        subsets = np.array(coeffs.index, dtype="uint64")
        hamming_weights = popcount(subsets)
        coeffs = coeffs[
            (hamming_weights < self.analyzer.number_spins // 2)
            | (
                (hamming_weights == self.analyzer.number_spins // 2)
                & (subsets < subsets ^ (2**self.analyzer.number_spins - 1))
            )
        ]
        # whe can use the symmetry of the subsets to halve the number of coefficients
        # the coefficient of the subset is the same as the coefficient of the complement
        # modulo the sign, and the value on the complement is the same as the value on the
        # subset modulo the sign, and the correcting sings are the same, thus
        # cancelling out

        predictions = []
        if max_batch_size is None or max_batch_size > len(x):
            max_batch_size = len(x)
        for x_batch in np.array_split(x, len(x) // max_batch_size):
            transform_matrix = calculate_fourier_transform_matrix(
                states=x_batch, subsets=np.array(coeffs.index, dtype="uint64")
            )

            prediction = np.asarray(
                2 * transform_matrix @ np.asarray(coeffs.values, dtype="float64"),
                dtype="float64",
            )
            # the factor 2 is to account for the fact that we
            # halved the number of coefficients

            predictions.append(prediction)

        prediction = np.concatenate(predictions)

        return prediction

    @overload
    def prediction_score(
        self,
        scorer: str | ScorerType,
        x: npt.NDArray[np.uint64] | None = None,
        max_batch_size: int | None = None,
    ) -> float:
        ...

    @overload
    def prediction_score(
        self,
        scorer: list[str | ScorerType],
        x: npt.NDArray[np.uint64] | None = None,
        max_batch_size: int | None = None,
    ) -> dict[str, float]:
        ...

    def prediction_score(
        self,
        scorer: str | ScorerType | list[str | ScorerType],
        x: npt.NDArray[np.uint64] | None = None,
        max_batch_size: int | None = None,
    ) -> float | dict[str, float]:

        if x is None:
            x = self.analyzer.canonical_basis.states

        signal = self.analyzer.signal(x)
        prob = self.analyzer.signal.get_probs(x)

        prediction = self.predict(x, max_batch_size=max_batch_size)

        if isinstance(scorer, Iterable) and not isinstance(scorer, str):
            return {str(s): get_scorer(s)(signal, prediction, prob) for s in scorer}

        return get_scorer(scorer)(signal, prediction, prob)


class LatticeBooleanAnalyzer:
    def __init__(
        self,
        signal: LatticeBooleanFunction,
        hamming_weight: int | Literal["half"] = "half",
        show_progress=False,
        cache_dir: Path | None = None,
        hadamard: bool = False,
        predict_hadamard: bool | None = None,
    ):
        self.signal = signal
        self.lattice = self.signal.lattice
        self.number_spins = self.lattice.number_spins
        self.show_progress = show_progress
        self.hadamard = hadamard
        if hadamard and cache_dir is not None:
            raise ValueError("Cannot use caching with hadamard transform")

        self.predict_hadamard = predict_hadamard if predict_hadamard is not None else hadamard
        if hamming_weight == "half":
            hamming_weight = self.number_spins // 2
        self.basis = self.lattice.get_basis(
            use_symmetries=True, hamming_weight=hamming_weight, spin_inversion=1
        )
        self.canonical_basis = self.lattice.get_basis(
            use_symmetries=False, hamming_weight=hamming_weight, spin_inversion=None
        )
        self.state_info_df = self.lattice.get_state_info_df(
            use_symmetries=True, hamming_weight=hamming_weight, spin_inversion=1
        )

        (
            self.subsets,
            self.fourier_basis_state_info_df,
        ) = self.lattice.make_fourier_basis_state_info_sym_df(show_progress=show_progress)

        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fit_cache_id(
        self,
        x: npt.NDArray[np.uint64],
    ) -> str:
        signal_id = self.signal.get_cache_id()
        return (
            md5(
                (
                    (
                        signal_id + ", ".join(map(str, x)) + ", ".join(map(str, self.subsets))
                    ).encode("utf8")
                ),
                usedforsecurity=False,
            ).hexdigest()
            + "-"
            + signal_id
        )

    def fit(
        self, x: npt.NDArray[np.uint64] | None = None, batch_size=100, from_cache_only=False
    ) -> "LatticeBooleanAnalyzer":

        if x is None:
            x = self.canonical_basis.states

        if self.cache_dir is not None:
            cache_id = self.fit_cache_id(x)
            cache_path = self.cache_dir / f"{cache_id}.lz"
            if cache_path.exists():
                return self.restore_learner_from_pickle(cache_path)
        else:
            cache_path = None

        if from_cache_only:
            raise ValueError(f"Cache not found: {cache_path}")

        self.learner = BooleanFourierLearner(
            self.number_spins, self.subsets, hadamard=self.hadamard
        )

        signal_values = self.signal(x)

        self.learner.fit(
            x,
            signal_values,
            batch_size=batch_size if not self.hadamard else None,
            show_progress=self.show_progress,
        )

        if cache_path is not None:
            if self.show_progress:
                print(f"Saving learner to cache: {cache_path}")
            with lzma.open(cache_path, "wb") as f:
                pickle.dump(self.learner, f)

        return self

    def restore_learner_from_pickle(self, path: str | Path) -> "LatticeBooleanAnalyzer":
        if self.show_progress:
            print(f"Restoring learner from cache: {path}")
        if str(path).endswith(".lz"):
            _open = lzma.open
        else:
            _open = open

        with _open(path, "rb") as f:
            self.learner = pickle.load(f)

        return self

    def truncate(self, strategy: TruncateStrategy) -> LBATruncation:
        return LBATruncation(analyzer=self, truncate_strategy=strategy)

    def how_many_terms_to_achieve_score(
        self,
        scorer: str | ScorerType,
        target_score: float = 0.95,
        min_terms: int = 1,
        max_terms: int = 101,
        step: int = 1,
        show_progress: bool = True,
        additional_scorers: list[str | ScorerType] | None = None,
    ) -> tuple[int | None, dict[str | ScorerType, float]]:
        """
        How many terms are needed to reconstruct the sign structure of the ground state
        with the given score?

        Parameters
        ----------

        scorer : str | ScorerType

        target_score : float, optional
            The target score, by default 0.95

        min_terms : int, optional
            The minimum number of terms to try, by default 1

        max_terms : int, optional
            The maximum number of terms to try, by default 101

        step : int, optional
            The step size, by default 10

        show_progress : bool, optional
            Show progress messages, by default True

        additional_scorers : list[str|ScorerType], optional
            Additional scorers to calculate, by default None

        Returns
        -------
        int | None
            The number of terms needed to achieve the target score, or None if the target
            score could not be achieved.

        dict[str|ScorerType, float]
            The scores achieved
        """

        evaluation_set = self.basis.states
        prediction = np.zeros_like(evaluation_set, dtype="float64")
        state_info_df = self.state_info_df

        if additional_scorers is None:
            additional_scorers = []

        scores: dict[str | ScorerType, float] = {}

        for n_terms in range(min_terms, max_terms, step):
            if n_terms == min_terms:
                predictor = self.truncate(keep_largest_n(n_terms))
            else:
                predictor = self.truncate(keep_largest_n(step, offset=max(0, n_terms - step)))

            prediction += predictor.predict(evaluation_set)

            true = self.signal(self.canonical_basis.states)
            prob = self.signal.get_probs(self.canonical_basis.states)

            prediction_expanded = np.asarray(
                state_info_df.merge(
                    pd.Series(prediction, name="prediction", index=evaluation_set),
                    left_on="representative",
                    right_index=True,
                    how="left",
                )["prediction"].values
            )

            score = get_scorer(scorer)(true, prediction_expanded, prob)
            scores[scorer] = score
            for additional_scorer in additional_scorers:
                scores[additional_scorer] = get_scorer(additional_scorer)(
                    true, prediction_expanded, prob
                )

            if show_progress:
                print(f"{n_terms} terms: {score:.3f}")
            if score >= target_score:
                return n_terms, scores
        return None, scores
