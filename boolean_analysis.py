from pathlib import Path

FOURIER_BASIS_DIR = Path("fourier-bases")
FOURIER_BASIS_DIR.mkdir(exist_ok=True)


from dataclasses import dataclass
from hashlib import md5
from typing import Any, Callable, Literal, Optional, Union

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


TruncateStrategy = Callable[[pd.Series], pd.Series]


def keep_largest_n(n: int) -> TruncateStrategy:
    def truncate(s: pd.Series) -> pd.Series:
        return (
            s.to_frame()
            .assign(abs_value=lambda x: np.abs(x["coeff"]))
            .sort_values("abs_value", ascending=False)["coeff"]
            .iloc[:n]
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
        fourier_basis_state_info.merge(
            sign_flip_basis_correspondence,
            left_on="representative",
            right_on="initial_representative",
            how="left",
        )
        .assign(representative=lambda x: np.minimum(x["representative_x"], x["representative_y"]))
        .assign(character=lambda x: find_character(x, system.number_spins))
        .drop(["representative_x", "representative_y", "initial_representative"], axis=1)
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
            self.subsets, self.fourier_basis_state_info_df = make_fourier_basis_state_info_sym(
                self.system
            )

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
            weights = signal_df["prob"].values.astype("float64")
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

    def set_truncate_strategy(self, strategy: TruncateStrategy):
        self.truncate_strategy = strategy
        return self

    def get_expanded_coeffs_ser(self) -> pd.Series:
        """
        Returns the Fourier coefficients expanded to the full set of subsets.

        Besides the expansion, the coefficients are also normalized in such a way
        that Plancherel's theorem holds.

        Returns
        -------

        pd.Series
            Series with index of the full image of all subsets under the action of
            the symmetry group, and values of the coefficients.

        """

        return (
            self.fourier_basis_state_info_df.join(
                self.learner.get_coeffs_ser(), on="representative"
            )
            .dropna()
            .assign(coeff_adjusted=lambda x: x["coeff"] * x["character"])["coeff_adjusted"]
            .rename("coeff")
            * self.system.canonical_basis.states.shape[0]
            / 2**self.system.number_spins
        )

    def predict(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:

        coeffs = self.truncate_strategy(self.get_expanded_coeffs_ser())

        transform_matrix = calculate_fourier_transform_matrix(
            states=x, subsets=np.array(coeffs.index, dtype="uint64")
        )

        return np.asarray(
            transform_matrix @ np.asarray(coeffs.values, dtype="float64"), dtype="float64"
        )

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

    def prediction_score(
        self,
        x: npt.NDArray[np.uint64],
        scorer: Union[str, ScorerType] = "overlap",
    ) -> float:

        if isinstance(scorer, str) and scorer in scorers:
            scorer = scorers[scorer]
        else:
            raise ValueError(
                f"scorer {scorer} not found. Available scorers: {list(scorers.keys())}"
            )

        signal_df = self._get_signal_df(x)
        prediction = self.predict(x)
        if self.signal_opt.kind == "sign":
            prediction = np.sign(prediction)
        return scorer(signal_df, prediction)
