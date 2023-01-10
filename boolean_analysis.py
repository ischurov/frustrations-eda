from pathlib import Path

FOURIER_BASIS_DIR = Path("fourier-bases")
FOURIER_BASIS_DIR.mkdir(exist_ok=True)


from dataclasses import dataclass
from hashlib import md5
from typing import Callable, Literal, Optional, Union

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
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
    return (true["y"] * predict * true["amplitude"]).sum() / true["prob"].sum()


@register_as_scorer
def accuracy_scorer(true: pd.DataFrame, predict: npt.NDArray):
    return (true["y"] == np.sign(predict)).mean()


class BooleanFourierAnalyser:
    def __init__(
        self,
        system: SpinSystem,
        use_subset_symmetries=True,
        eigenstates: int = 1,
        show_progress: bool = False,
    ):
        if system.number_spins % 2 != 0:
            raise ValueError("Only even number of spins are supported so far")

        self.system = system
        self.use_subset_symmetries = use_subset_symmetries

        number_spins = self.system.number_spins

        self.canonical_fourier_basis = ls.SpinBasis(
            ls.Group([]),
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        self.canonical_fourier_basis.build()

        if use_subset_symmetries:
            self.fourier_basis = ls.SpinBasis(
                self.system.symmetry_group,
                number_spins=number_spins,
                hamming_weight=None,
                spin_inversion=1,
            )
        else:
            self.fourier_basis = self.canonical_fourier_basis

        self.fourier_basis.build()

        print("Finding system ground state")
        self.system.get_eigenstates(eigenstates)

        self.show_progress = show_progress

        self.fourier_basis_state_info_df = batched_state_info_df(
            self.fourier_basis, np.arange(2**self.system.number_spins, dtype="uint64")
        )

    def get_signal_df(
        self, states: npt.NDArray[np.uint64], signal_opt: SignalOption = SignalOption()
    ) -> pd.DataFrame:
        """
        Returns the data frame with the signal values for the given set of states
        and the given signal option.

        Parameters
        ----------
        states : npt.NDArray[np.uint64]

            Array of states for which the signal values are to be calculated.

        signal_opt : SignalOption
            Signal option.

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
            self.system.get_df_eigenstate(k=signal_opt.eigenstate, canonical_basis=True)
            .assign(prob=lambda df: df["amplitude"] ** 2)
            .loc[states, :]
        )

        y = signal_df["eigenstate_coeff"].values.astype("float64")

        if signal_opt.kind == "sign":
            y = np.sign(y)

        signal_df["y"] = y

        return signal_df

    def fit(
        self,
        x: npt.NDArray[np.uint64],
        signal_opt: SignalOption = SignalOption(),
        batch_size=100,
    ) -> "BooleanFourierAnalyser":

        signal_df = self.get_signal_df(x, signal_opt)

        self.learner = BooleanFourierLearner(
            self.system.number_spins, self.fourier_basis.states
        )

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

    def expand_fourier_coeffs_ser(self, coeffs: pd.Series) -> pd.Series:
        """
        Expands the Fourier coefficients to the full set of subsets. Each subset
        is replaced by all its images under the action of the symmetry group
        (including the sign symmetry), and the coefficients are duplicated accordingly.

        Parameters
        ----------
        coeffs : pd.Series
            Series with index of the subsets and values of the coefficients.
            This can be produced by self.learner.get_coeffs_df() or contain
            trimmed version (i.e. only the largest coefficients kept, etc.)

        Returns
        -------

        pd.Series
            Series with index of the full image of all subsets under the action of
            the symmetry group, and values of the coefficients.

        """
        return self.fourier_basis_state_info_df.join(coeffs, on="representative")[
            "coeff"
        ].dropna()

    def predict(
        self, x: npt.NDArray[np.uint64], coeffs: pd.Series
    ) -> npt.NDArray[np.float64]:

        transform_matrix = calculate_fourier_transform_matrix(
            states=x, subsets=np.array(coeffs.index, dtype="uint64")
        )

        return np.array(
            transform_matrix @ np.array(coeffs.values, dtype="float64"), dtype="float64"
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
        coeffs: pd.Series,
        signal_opt: SignalOption = SignalOption(),
        scorer: Union[str, ScorerType] = "overlap",
    ) -> float:

        if isinstance(scorer, str) and scorer in scorers:
            scorer = scorers[scorer]
        else:
            raise ValueError(
                f"scorer {scorer} not found. Available scorers: {list(scorers.keys())}"
            )

        signal_df = self.get_signal_df(x, signal_opt)
        prediction = self.predict(x, coeffs)
        if signal_opt.kind == "sign":
            prediction = np.sign(prediction)
        return scorer(signal_df, prediction)
