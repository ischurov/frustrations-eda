from pathlib import Path, PosixPath

FOURIER_BASIS_DIR = Path("fourier-bases")
FOURIER_BASIS_DIR.mkdir(exist_ok=True)


import lzma
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import md5
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
from heisenberg_hamiltonians import (
    SpinSystem,
    batched_state_info_df,
    make_unpacked_configurations,
)
from parity import calculate_fourier_transform_matrix


@dataclass(frozen=True)
class SignalOption:
    kind: Literal["sign", "value"] = "sign"
    eigenstate: int = 0
    weights: Literal["none", "prob", "invprob"] = "none"


ScorerType = Callable[[pd.DataFrame, npt.NDArray], float]

scorers: dict[str, ScorerType] = {}


def register_as_scorer(f: ScorerType):
    scorers[f.__name__.removesuffix("_scorer")] = f
    return f


@register_as_scorer
def overlap_scorer(true: pd.DataFrame, predict: npt.NDArray):
    return (true["y"] * predict * true["prob"]).sum() / true["prob"].sum()


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


def find_character(x: pd.DataFrame, n_spins: int) -> np.ndarray:
    if (n_spins // 2) % 2 == 0:
        return np.ones_like(x["representative"])
    return np.where(
        x["representative"] != x["representative_x"],
        -1,
        np.where(x["representative"] != x["representative_y"], 1, 0),
    )


def make_fourier_basis_state_info_sym(
    system: SpinSystem,
) -> tuple[npt.NDArray[np.uint64], pd.DataFrame]:
    fourier_basis = ls.SpinBasis(
        symmetries=system.symmetries,
        number_spins=system.number_spins,
        hamming_weight=None,
        spin_inversion=None,
    )
    fourier_basis.build()

    all_subsets = np.arange(2**system.number_spins, dtype="uint64")
    mask = 2**system.number_spins - 1

    fourier_basis_state_info = batched_state_info_df(fourier_basis, all_subsets).drop(
        "norm", axis=1
    )
    sign_flip_basis_correspondence = (
        batched_state_info_df(fourier_basis, fourier_basis.states ^ mask)
        .assign(initial_representative=fourier_basis.states)
        .drop(["character", "norm"], axis=1)
    )
    fourier_basis_state_info_df = (
        fourier_basis_state_info.reset_index()
        .rename(columns={"index": "state"})
        .merge(
            sign_flip_basis_correspondence,
            left_on="representative",
            right_on="initial_representative",
            how="left",
        )
        .assign(representative=lambda x: np.minimum(x["representative_x"], x["representative_y"]))
        .assign(character=lambda x: find_character(x, system.number_spins))
        .drop(["representative_x", "representative_y", "initial_representative"], axis=1)
        .set_index("state")
        .reindex(columns=["representative", "character"])
    )

    subsets = sign_flip_basis_correspondence[
        lambda x: x["initial_representative"] <= x["representative"]  # type: ignore
    ]["initial_representative"].values
    # see https://github.com/pandas-dev/pandas-stubs/issues/256#issuecomment-1235774506

    return subsets, fourier_basis_state_info_df


class BooleanFourierAnalyser:
    def __init__(
        self,
        system: SpinSystem,
        use_subset_symmetries=True,
        eigenstates: int = 1,
        show_progress: bool = False,
    ):
        self.system = system
        self.use_subset_symmetries = use_subset_symmetries
        self.truncate_strategy: TruncateStrategy = keep_everything

        number_spins = self.system.number_spins

        self.canonical_fourier_basis = ls.SpinBasis(
            symmetries=ls.Symmetries([]),
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        self.canonical_fourier_basis.build()

        if use_subset_symmetries:
            (
                self.subsets,
                self.fourier_basis_state_info_df,
            ) = make_fourier_basis_state_info_sym(self.system)

        else:
            self.fourier_basis_state_info_df = batched_state_info_df(
                self.canonical_fourier_basis,
                np.arange(2**self.system.number_spins, dtype="uint64"),
            ).drop("norm", axis=1)

            self.subsets = self.canonical_fourier_basis.states

        print("Finding system ground state")
        self.system.get_eigenstates(eigenstates)

        self.show_progress = show_progress

    def _get_signal_df(self, states: npt.NDArray[np.uint64]) -> pd.DataFrame:
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
        - y: the signal value (i.e. sign for signal_opt.kind == "sign")
        """

        signal_df = (
            self.system.get_df_eigenstate(k=self.signal_opt.eigenstate, canonical_basis=True)
            .assign(prob=lambda df: df["amplitude"] ** 2)
            .loc[states, :]
        )  # type: ignore
        # See https://github.com/pandas-dev/pandas-stubs/issues/508

        y = signal_df["eigenstate_coeff"].values.astype("float64")

        if self.signal_opt.kind == "sign":
            y = np.sign(y)

        signal_df["y"] = y

        return signal_df

    def fit(
        self,
        x: npt.NDArray[np.uint64],
        signal_opt: SignalOption = SignalOption(),
        batch_size=100,
    ) -> "BooleanFourierAnalyser":

        self.signal_opt = signal_opt
        signal_df = self._get_signal_df(x)

        self.learner = BooleanFourierLearner(self.system.number_spins, self.subsets)

        if signal_opt.weights == "prob":
            weights = signal_df["prob"].values.astype("float64") * signal_df.shape[0]
        elif signal_opt.weights == "invprob":
            weights = 1.0 / signal_df["prob"].values.astype("float64")
        else:
            weights = None

        self.learner.fit(
            x,
            signal_df["y"].values.astype("float64"),
            weights=weights,
            batch_size=batch_size,
            show_progress=self.show_progress,
        )

        return self

    def restore_learner_from_pickle(
        self, path: str | PosixPath, signal_opt: SignalOption = SignalOption()
    ) -> "BooleanFourierAnalyser":
        self.signal_opt = signal_opt

        if str(path).endswith(".lz"):
            _open = lzma.open
        else:
            _open = open

        with _open(path, "rb") as f:
            self.learner = pickle.load(f)

        return self

    def set_truncate_strategy(self, strategy: TruncateStrategy):
        self.truncate_strategy = strategy
        return self

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
            self.fourier_basis_state_info_df.join(
                self.learner.get_coeffs_ser(), on="representative"
            )
            .dropna()
            .assign(coeff_adjusted=lambda x: x["coeff"] * x["character"])["coeff_adjusted"]
            .rename("coeff")
            * self.system.canonical_basis.states.shape[0]
            / 2**self.system.number_spins
        )

        idxs_after_truncation = (
            self.fourier_basis_state_info_df.join(
                self.truncate_strategy(self.learner.get_coeffs_ser()),
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

    def predict(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
        coeffs = self.get_expanded_coeffs_ser()

        if self.use_subset_symmetries:
            coeffs = coeffs[
                np.asarray(coeffs.index, dtype="uint64")
                < np.asarray(coeffs.index, dtype="uint64") ^ (2**self.system.number_spins - 1)
            ]
            # whe can use the symmetry of the subsets to halve the number of coefficients
            # the coefficient of the subset is the same as the coefficient of the complement
            # modulo the sign, and the value on the complement is the same as the value on the
            # subset modulo the sign

        transform_matrix = calculate_fourier_transform_matrix(
            states=x, subsets=np.array(coeffs.index, dtype="uint64")
        )

        return np.asarray(
            (1 + self.use_subset_symmetries)
            * transform_matrix
            @ np.asarray(coeffs.values, dtype="float64"),
            dtype="float64",
        )
        # the factor (1 + self.use_subset_symmetries) is to account for the fact that we
        # halved the number of coefficients if self.use_subset_symmetries is True

    # def visualize_spectre_support_barplot(
    #     self, signal: SignalOption = SignalOption(), abs=False, elements=20, ax=None
    # ):
    #     if ax is None:
    #         ax = plt.gca()
    #     return (
    #         self.get_spectre_df(signal)
    #         .iloc[:elements]
    #         .plot.bar(y="abs_coeff" if abs else "coeff", ax=ax)
    #     )

    # def visualize_spectre_support_lattice(
    #     self, signal: SignalOption = SignalOption(), m: int = 0, ax=None
    # ):
    #     if ax is None:
    #         ax = plt.gca()
    #     spectre = self.get_spectre_df(signal)
    #     subset = np.array(spectre.index[m : m + 1])
    #     subset_configuration = make_unpacked_configurations(
    #         subset, self.system.number_spins
    #     )[0]
    #     self.system.lat.plot(spins=subset_configuration, ax=ax)
    #     ax.set_title(f"subset: {subset}")

    # def visualize_spectre_values_hist(
    #     self, signal: SignalOption = SignalOption(), abs=False, bins=50, ax=None
    # ):
    #     if ax is None:
    #         ax = plt.gca()
    #     return self.get_spectre_df(signal)["abs_coeff" if abs else "coeff"].hist(
    #         bins=bins, ax=ax
    #     )

    # def spectre_entropy(self, signal: SignalOption = SignalOption()):
    #     abs_coeff = self.get_spectre_df(signal)["abs_coeff"]
    #     return entropy(abs_coeff / abs_coeff.sum())
    @overload
    def prediction_score(self, x: npt.NDArray[np.uint64], scorer: str | ScorerType) -> float:
        ...

    @overload
    def prediction_score(
        self, x: npt.NDArray[np.uint64], scorer: Iterable[str | ScorerType]
    ) -> dict[str, float]:
        ...

    def prediction_score(
        self,
        x: npt.NDArray[np.uint64],
        scorer: str | ScorerType | Iterable[str | ScorerType] = "overlap",
    ) -> float | dict[str, float]:

        signal_df = self._get_signal_df(x)
        prediction = self.predict(x)
        if self.signal_opt.kind == "sign":
            prediction = np.sign(prediction)

        if isinstance(scorer, Iterable) and not isinstance(scorer, str):
            return {str(s): get_scorer(s)(signal_df, prediction) for s in scorer}

        return get_scorer(scorer)(signal_df, prediction)
