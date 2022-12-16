from pathlib import Path

FOURIER_BASIS_DIR = Path("fourier-bases")
FOURIER_BASIS_DIR.mkdir(exist_ok=True)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lattice_symmetries as ls
from tqdm import tqdm
from typing import Literal, Optional

from hashlib import md5

from heisenberg_hamiltonians import (
    make_unpacked_configurations,
    HeisenbergJ1J2,
    SpinSystem,
    batched_state_info_df,
    pad_right,
)

from dataclasses import dataclass


from scipy.stats import entropy


def parity_of_1s(x: np.ndarray, n: int, show_progress=False):
    """
    Calculates the parities of number of 1's for all elements of x

    Params
    ------
    x : np.ndarray
        should have dtype == 'uint64'

    n : int
        number of bits (all elements of x should be less than 2 ** n)

    show_progress : bool
                    whether to show progress bar

    Returns
    -------
    parity : np.ndarray
             dtype == 'uint8'
    """

    parity = np.zeros_like(x, dtype="uint8")
    x = x.copy()
    for i in [lambda _: _, tqdm][show_progress](range(n)):
        parity ^= x & 1
        x >>= 1
    return parity


assert (
    np.isclose(
        parity_of_1s(np.array([[1, 2, 3], [4, 5, 6]], dtype="uint64"), 3),
        np.array([[1, 1, 0], [1, 0, 0]], dtype="uint64"),
    )
).all()


def get_fourier_transform_matrix(
    states: np.ndarray,
    subsets: np.ndarray,
    number_spins: int,
    fourier_transform_matrix_cache: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Fourier Transform Matrix (FTM) is a matrix defined as follows:
    
    - rows: state vectors (usually of a canonical basis)
    - columns: subset masks
    - values: value of the parity function defined by the subset 
              mask on the state
              
    This function returns the FTM with the following caching:
    - First local cache tried (if fourier_transform_matrix_cache specified) 
    - Then FOURIER_BASIS_DIR is looked up for the saved matrix
    - Finally, the matrix is calculated using calculate_fourier_transform_matrix
    
    After this function is invoked, the caches are updated accordingly
              
    Params
    ------
    
    states, subsets: rows and columns of a matrix
    number_spins: number of spins
    fourier_transform_matrix_cache: a dictionary that stores local cache
    
    Returns
    -------
    
    Fourier Transform Matrix
    """
    if states.dtype != "uint64" or subsets.dtype != "uint64":
        raise ValueError("states and subsets dtype should be uint64")
    fourier_basis_id = md5(
        f"{number_spins}{states!r}" f"{subsets!r}".encode("utf8"),
        usedforsecurity=False,
    ).hexdigest()
    fourier_transform_matrix_path = FOURIER_BASIS_DIR / Path(
        f"fourier_transform_matrix-{fourier_basis_id}.feather"
    )

    fourier_transform_matrix = None

    if fourier_transform_matrix_cache is not None:
        if (
            cached_basis := fourier_transform_matrix_cache.get(fourier_basis_id)
        ) is not None:
            print("Found cached fourier basis, will use it")
            fourier_transform_matrix = cached_basis

    if fourier_transform_matrix is None:
        if fourier_transform_matrix_path.exists():
            print(f"Fourier transform matrix found at {fourier_transform_matrix_path}")
            fourier_transform_matrix_df = pd.read_feather(
                fourier_transform_matrix_path
            ).set_index("index")
            if (fourier_transform_matrix_df.index != states).any():
                raise ValueError(
                    f"basis states in file {fourier_transform_matrix_path} do not coincide with "
                    f"states argument"
                )
            fourier_transform_matrix = fourier_transform_matrix_df.values.astype("int8")

    if fourier_transform_matrix is None:
        print(
            "Fourier transform matrix not found, we have to calculate it. "
            "This can take some time, you can get a coffee."
        )
        fourier_transform_matrix = calculate_fourier_transform_matrix(
            states=states, subsets=subsets, number_spins=number_spins
        )

    if not fourier_transform_matrix_path.exists():
        print(f"Saving basis to file {fourier_transform_matrix_path}")
        pd.DataFrame(
            fourier_transform_matrix, index=states, columns=[str(i) for i in subsets],
        ).reset_index().to_feather(fourier_transform_matrix_path)

    if fourier_transform_matrix_cache is not None:
        fourier_transform_matrix_cache[fourier_basis_id] = fourier_transform_matrix
    return fourier_transform_matrix


def calculate_fourier_transform_matrix(
    states: np.ndarray, subsets: np.ndarray, number_spins: int
) -> np.ndarray:
    """
    This is a low-level function that calculates the Fourier Transform Matrix.
    
    See details in get_fourier_transform_matrix
    """

    masks = subsets.reshape(1, -1)
    masked = states.reshape(-1, 1) & masks
    parities = parity_of_1s(masked, number_spins, show_progress=True)
    return parities.astype("int8") * 2 - 1


@dataclass(frozen=True)
class SignalOption:
    kind: Literal["sign", "value"] = "sign"
    eigenstate: int = 0


class BooleanFourierAnalyser:
    def __init__(
        self,
        system: SpinSystem,
        use_symmetries=True,
        fourier_transform_matrix_cache=None,
        eigenstates: int = 1,
    ):

        self.system = system
        self.use_symmetries = use_symmetries
        self.fourier_transform_matrix_cache = fourier_transform_matrix_cache

        number_spins = self.system.number_spins
        canonical_basis = self.system.canonical_basis

        self.canonical_fourier_basis = ls.SpinBasis(
            ls.Group([]),
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        self.canonical_fourier_basis.build()

        if use_symmetries:
            self.fourier_basis = ls.SpinBasis(
                self.system.symmetry_group,
                number_spins=number_spins,
                hamming_weight=None,
                spin_inversion=None,
            )
        else:
            self.fourier_basis = self.canonical_fourier_basis

        self.fourier_basis.build()

        fourier_basis_id = md5(
            f"{number_spins}{canonical_basis.hamming_weight}"
            f"{self.fourier_basis.states!r}".encode("utf8"),
            usedforsecurity=False,
        ).hexdigest()

        fourier_transform_matrix_path = FOURIER_BASIS_DIR / Path(
            f"fourier_transform_matrix-{fourier_basis_id}.feather"
        )

        self.fourier_transform_matrix = get_fourier_transform_matrix(
            self.system.canonical_basis.states,
            self.fourier_basis.states,
            number_spins,
            fourier_transform_matrix_cache=self.fourier_transform_matrix_cache,
        )

        if use_symmetries:
            self.canonical_fourier_transform_matrix = None
            # FTM with full set of Fourier basis vectors
            # if symmetries are used, it is much larger than self.fourier_transform_matrix
            # we need it on inference, will calculate if needed

        else:
            self.canonical_fourier_transform_matrix = self.fourier_transform_matrix

        print("Finding system ground state")
        self.system.get_eigenstates(eigenstates)

        self.spectre_cache = {}

    def fourier_decomposition(self, signal):
        return (
            self.fourier_transform_matrix.T @ signal / (2 ** self.system.number_spins)
        )

    def get_spectre_df(self, signal: SignalOption = SignalOption()):
        if signal in self.spectre_cache:
            return self.spectre_cache[signal]

        signal_ = self.system.get_df_eigenstate(
            k=signal.eigenstate, canonical_basis=True
        )["eigenstate_coeff"]

        if signal.kind == "sign":
            signal_ = np.sign(signal_)

        spectre_df = (
            pd.DataFrame(
                dict(
                    coeff=self.fourier_decomposition(signal_),
                ),
                index=self.fourier_basis.states,
            )
            .assign(abs_coeff=lambda x: np.abs(x["coeff"]))
            .sort_values("abs_coeff", ascending=False)
        )

        self.spectre_cache[signal] = spectre_df

        return spectre_df

    def visualize_spectre_support_barplot(
        self, signal: SignalOption = SignalOption(), abs=False, elements=20, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        return (
            self.get_spectre_df(signal)
            .iloc[:elements]
            .plot.bar(y="abs_coeff" if abs else "coeff", ax=ax)
        )

    def visualize_spectre_support_lattice(
        self, signal: SignalOption = SignalOption(), m: int = 0, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        spectre = self.get_spectre_df(signal)
        subset = np.array(spectre.index[m: m+1])
        subset_configuration = make_unpacked_configurations(subset, self.system.number_spins)[0] 
        self.system.lat.plot(spins=subset_configuration, ax=ax)
        ax.set_title(f"subset: {subset}")

    def visualize_spectre_values_hist(
        self, signal: SignalOption = SignalOption(), abs=False, bins=50, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        return self.get_spectre_df(signal)["abs_coeff" if abs else "coeff"].hist(
            bins=bins, ax=ax
        )

    def spectre_entropy(self, signal: SignalOption = SignalOption()):
        abs_coeff = self.get_spectre_df(signal)["abs_coeff"]
        return entropy(abs_coeff / abs_coeff.sum())

    def predict(
        self, signal: SignalOption = SignalOption(), keep_first=None
    ) -> np.array:
        coeffs = self.get_spectre_df(signal)["coeff"]
        if keep_first is not None:
            coeffs = coeffs.iloc[:keep_first]

        if self.use_symmetries:
            state_info_df = batched_state_info_df(
                self.fourier_basis, self.canonical_fourier_basis.states
            )
            coeffs = state_info_df.join(coeffs, on="representative")["coeff"].dropna()

        if keep_first is None:
            if self.canonical_fourier_transform_matrix is None:
                self.canonical_fourier_transform_matrix = get_fourier_transform_matrix(
                    states=self.system.canonical_basis.states,
                    subsets=self.canonical_fourier_basis.states,
                    number_spins=self.system.number_spins,
                    fourier_transform_matrix_cache=self.fourier_transform_matrix_cache,
                )

            coeffs = coeffs.loc[self.canonical_fourier_basis.states]

            return self.canonical_fourier_transform_matrix @ coeffs
        else:
            transform_matrix = calculate_fourier_transform_matrix(
                states=self.system.canonical_basis.states,
                subsets=np.array(coeffs.index, dtype="uint64"),
                number_spins=self.system.number_spins,
            )

            return transform_matrix @ coeffs
