import lzma
import pickle
import warnings
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
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from boolean_fourier_learner import BooleanFourierLearner
from spin_systems import (
    SpinSystem,
    batched_state_info_df,
    make_unpacked_configurations,
)
from parity import calculate_fourier_transform_matrix, parity, popcount

warnings.warn("This module is deprecated and will be removed in the future.")
warnings.warn("Use lattice_boolean_analysis.py instead.")


def camel_case_to_snake_case(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")


class SignalKind:
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        This function defines how to transform data during the training.
        """
        raise NotImplementedError

    @staticmethod
    def transform_predict(predict: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        This function defines how to transform the output of the model during
        prediction.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return camel_case_to_snake_case(self.__class__.__name__.removesuffix("SignalKind"))


class BinarySignalKind(SignalKind):
    @staticmethod
    def transform_predict(predict: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sign(predict)


class RealSignalKind(SignalKind):
    @staticmethod
    def transform_predict(predict: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return predict


class ValueSignalKind(RealSignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return eigenstate_coeff


class SignSignalKind(BinarySignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sign(eigenstate_coeff)


class AmplitudeSignalKind(RealSignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.abs(eigenstate_coeff)


class ProbSignalKind(RealSignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (np.abs(eigenstate_coeff)) ** 2  # type: ignore
        # see https://github.com/numpy/numpy/issues/20099


class AmplitudeMedianBinSignalKind(BinarySignalKind):
    @staticmethod
    def transform_data(eigenstate_coeff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        abs_ = np.abs(eigenstate_coeff)
        return np.sign(abs_ - np.median(abs_))


Weights = Callable[[pd.DataFrame], npt.NDArray[np.float64]]


def uniform_weights(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    return np.ones(df["prob"].shape[0], dtype="float64")


def prob_weights(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    return np.asarray(df["prob"].values, dtype="float64") * df.shape[0]  # type: ignore
    # see https://github.com/numpy/numpy/issues/20099


def inv_prob_weights(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    invprobs = 1 / np.asarray(df["prob"].values, dtype="float64")
    return invprobs / invprobs.sum() * df.shape[0]


@dataclass
class SignalOption:
    kind: SignalKind = SignSignalKind()
    eigenstate: int = 0
    weights: Weights = uniform_weights

    def __str__(self):
        return f"SignalOption(kind={self.kind.__class__.__name__}, eigenstate={self.eigenstate}, weights={self.weights.__name__})"


ScorerType = Callable[[pd.DataFrame, npt.NDArray], float]

scorers: dict[str, ScorerType] = {}


def register_as_scorer(f: ScorerType):
    scorers[f.__name__.removesuffix("_scorer")] = f
    return f


@register_as_scorer
def sign_overlap_scorer(true: pd.DataFrame, predict: npt.NDArray):
    return (true["y"] * predict * true["prob"]).sum() / true["prob"].sum()


@register_as_scorer
def value_overlap_scorer(true: pd.DataFrame, predict: npt.NDArray):
    return (true["y"] * predict).sum() / true["prob"].sum()


@register_as_scorer
def accuracy_scorer(true: pd.DataFrame, predict: npt.NDArray):
    return (true["y"] == np.sign(predict)).mean()


@register_as_scorer
def neg_mse_scorer(true: pd.DataFrame, predict: npt.NDArray) -> float:
    return -float(np.mean((true["y"] - predict) ** 2))


def get_scorer(scorer: str | ScorerType) -> ScorerType:
    if isinstance(scorer, str):
        if scorer in scorers:
            scorer = scorers[scorer]
        else:
            raise ValueError(
                f"scorer {scorer} not found. Available scorers: {list(scorers.keys())}"
            )
    return scorer


TruncateStrategy = Callable[[pd.Series], pd.Series]


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


class BFATruncation:
    def __init__(self, analyzer: "BooleanFourierAnalyzer", truncate_strategy: TruncateStrategy):
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

        full_coeffs = (
            self.analyzer.fourier_basis_state_info_df.join(
                self.analyzer.learner.get_coeffs_ser(), on="representative"
            )
            .dropna()
            .assign(coeff_adjusted=lambda x: x["coeff"] * x["character"])["coeff_adjusted"]
            .rename("coeff")
            * self.analyzer.system.canonical_basis.states.shape[0]
            / 2**self.analyzer.system.number_spins
        )

        idxs_after_truncation = (
            self.analyzer.fourier_basis_state_info_df.join(
                self.truncate_strategy(self.analyzer.learner.get_coeffs_ser()),
                on="representative",
            ).dropna()
        ).index

        coeffs = (
            full_coeffs.loc[idxs_after_truncation]
            .to_frame()
            .assign(abs_coeff=lambda x: np.abs(x["coeff"]))
            .sort_values("abs_coeff", ascending=False)["coeff"]
        )
        return coeffs

    def predict(
        self,
        x: npt.NDArray[np.uint64],
        transform_according_to_signal=True,
        max_batch_size: int | None = None,
    ) -> npt.NDArray[np.float64]:
        coeffs = self.get_expanded_coeffs_ser()

        if self.analyzer.use_subset_symmetries:
            subsets = np.array(coeffs.index, dtype="uint64")
            hamming_weights = popcount(subsets)
            coeffs = coeffs[
                (hamming_weights < self.analyzer.system.number_spins // 2)
                | (
                    (hamming_weights == self.analyzer.system.number_spins // 2)
                    & (subsets < subsets ^ (2**self.analyzer.system.number_spins - 1))
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
                (1 + self.analyzer.use_subset_symmetries)
                * transform_matrix
                @ np.asarray(coeffs.values, dtype="float64"),
                dtype="float64",
            )
            # the factor (1 + self.use_subset_symmetries) is to account for the fact that we
            # halved the number of coefficients if self.use_subset_symmetries is True

            predictions.append(prediction)

        prediction = np.concatenate(predictions)

        if transform_according_to_signal:
            prediction = self.analyzer.signal_opt.kind.transform_predict(prediction)

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
            x = self.analyzer.system.canonical_basis.states

        signal_df = self.analyzer.get_signal_df(x)
        prediction = self.predict(x, max_batch_size=max_batch_size)

        if isinstance(scorer, Iterable) and not isinstance(scorer, str):
            return {str(s): get_scorer(s)(signal_df, prediction) for s in scorer}

        return get_scorer(scorer)(signal_df, prediction)


class BooleanFourierAnalyzer:
    def __init__(
        self,
        system: SpinSystem,
        use_subset_symmetries=True,
        eigenstates: int = 1,
        show_progress: bool = False,
        cache_dir: Path | None = None,
    ):
        self.system = system
        self.use_subset_symmetries = use_subset_symmetries
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        number_spins = self.system.number_spins

        if use_subset_symmetries:
            if show_progress:
                print("Making Fourier basis state info with symmetries...")
            (
                self.subsets,
                self.fourier_basis_state_info_df,
            ) = self.system.lattice.make_fourier_basis_state_info_sym_df(
                show_progress=show_progress
            )

        else:
            if show_progress:
                print("Making Fourier basis state info without symmetries...")

            canonical_fourier_basis = ls.SpinBasis(
                symmetries=[],
                number_spins=number_spins,
                hamming_weight=None,
                spin_inversion=None,
            )

            canonical_fourier_basis.build()

            self.fourier_basis_state_info_df = batched_state_info_df(
                canonical_fourier_basis,
                np.arange(2**self.system.number_spins, dtype="uint64"),
            ).drop("norm", axis=1)

            self.subsets = canonical_fourier_basis.states
        if show_progress:
            print("Finding system ground state")
        self.system.get_eigenstates(eigenstates)

        self.show_progress = show_progress

    def get_signal_df(self, states: npt.NDArray[np.uint64]) -> pd.DataFrame:
        """
        Returns the data frame with the signal values for the given set of states
        and the given signal option.

        Parameters
        ----------
        states : npt.NDArray[np.uint64]

            Array of states for which the signal values are to be calculated.

        Returns
        -------

        pd.DataFrame

        Data frame with the index of the states and the following columns:

        - eigenstate_coeff: the coefficient of the given eigenstate
        - amplitude: the amplitude of the given state
        - prob: the probability of the given state
        - y: the signal value (i.e. sign for signal_opt.kind == SignSignalKind())
        """

        signal_df = (
            self.system.get_df_eigenstate(k=self.signal_opt.eigenstate, canonical_basis=True)
            .assign(prob=lambda df: df["amplitude"] ** 2)
            .loc[states, :]
        )  # type: ignore
        # See https://github.com/pandas-dev/pandas-stubs/issues/508

        signal_df["y"] = self.signal_opt.kind.transform_data(
            signal_df["eigenstate_coeff"].values.astype("float64")
        )

        return signal_df

    def fit_cache_id(
        self, x: npt.NDArray[np.uint64], signal_opt: SignalOption = SignalOption()
    ) -> str:

        return md5(
            (
                (
                    self.system.get_cache_id()
                    + ", ".join(map(str, x))
                    + ", ".join(map(str, self.subsets))
                    + str(signal_opt)
                ).encode("utf8")
            ),
            usedforsecurity=False,
        ).hexdigest()

    def fit(
        self,
        x: npt.NDArray[np.uint64],
        signal_opt: SignalOption = SignalOption(),
        batch_size=100,
        from_cache_only=False,
    ) -> "BooleanFourierAnalyzer":

        if self.cache_dir is not None:
            cache_id = self.fit_cache_id(x, signal_opt)
            cache_path = (
                self.cache_dir / f"{cache_id}-{self.system.get_cache_id()}-{str(signal_opt)}.lz"
            )

            if cache_path.exists():
                return self.restore_learner_from_pickle(cache_path, signal_opt)
        else:
            cache_path = None
        if from_cache_only:
            raise ValueError(f"Cache not found: {cache_path}")

        self.signal_opt = signal_opt
        signal_df = self.get_signal_df(x)

        self.learner = BooleanFourierLearner(self.system.number_spins, self.subsets)

        weights = self.signal_opt.weights(signal_df)

        self.learner.fit(
            x,
            signal_df["y"].values.astype("float64"),
            weights=weights,
            batch_size=batch_size,
            show_progress=self.show_progress,
        )

        if cache_path is not None:
            if self.show_progress:
                print(f"Saving learner to cache: {cache_path}")
            with lzma.open(cache_path, "wb") as f:
                pickle.dump(self.learner, f)

        return self

    def restore_learner_from_pickle(
        self, path: str | Path, signal_opt: SignalOption = SignalOption()
    ) -> "BooleanFourierAnalyzer":
        self.signal_opt = signal_opt
        if self.show_progress:
            print(f"Restoring learner from cache: {path}")
        if str(path).endswith(".lz"):
            _open = lzma.open
        else:
            _open = open

        with _open(path, "rb") as f:
            self.learner = pickle.load(f)

        return self

    def truncate(self, strategy: TruncateStrategy) -> BFATruncation:
        return BFATruncation(analyzer=self, truncate_strategy=strategy)

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

        evaluation_set = self.system.basis.states
        prediction = np.zeros_like(evaluation_set, dtype="float64")
        state_info_df = batched_state_info_df(
            self.system.basis, self.system.canonical_basis.states
        )

        if additional_scorers is None:
            additional_scorers = []

        scores: dict[str | ScorerType, float] = {}

        for n_terms in range(min_terms, max_terms, step):
            if n_terms == min_terms:
                predictor = self.truncate(keep_largest_n(n_terms))
            else:
                predictor = self.truncate(keep_largest_n(step, offset=max(0, n_terms - step)))

            prediction += predictor.predict(evaluation_set, transform_according_to_signal=False)

            true = self.get_signal_df(self.system.canonical_basis.states)
            prediction_expanded = self.signal_opt.kind.transform_predict(
                np.asarray(
                    state_info_df.merge(
                        pd.Series(prediction, name="prediction", index=evaluation_set),
                        left_on="representative",
                        right_index=True,
                        how="left",
                    )["prediction"].values
                )
            )

            score = get_scorer(scorer)(true, prediction_expanded)
            scores[scorer] = score
            for additional_scorer in additional_scorers:
                scores[additional_scorer] = get_scorer(additional_scorer)(
                    true, prediction_expanded
                )

            if show_progress:
                print(f"{n_terms} terms: {score:.3f}")
            if score >= target_score:
                return n_terms, scores
        return None, scores
