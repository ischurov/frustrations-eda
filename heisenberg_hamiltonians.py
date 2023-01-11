import pickle
from pathlib import Path
from typing import Optional

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import scipy.sparse.linalg

from spin_lattices import SpinLattice
from utils import make_unpacked_configurations


def batched_state_info_df(basis: ls.SpinBasis, states: npt.NDArray[np.uint64]):
    """
    Parameters
    ----------
    basis : ls.SpinBasis
        Basis to use for the state info
    states : npt.NDArray[np.uint64]
        States to get the info for

    Returns
    -------
    pd.DataFrame

    Returns a DataFrame with index states and the following columns:

    - representative: representative of the group trajectory containing the state
    - character: character of the group element that takes the representative to the state
    - norm: normalizing coefficient
    """
    representative, eigenvalue, norm = basis.state_info(states)

    return pd.DataFrame(
        dict(representative=representative, character=eigenvalue, norm=norm),
        index=states,
    )


class SpinSystem:
    def __init__(
        self,
        lat: SpinLattice,
        basis: ls.SpinBasis,
        hamiltonian: ls.Operator,
        symmetries: ls.Symmetries,
        ground_state_cache_dir: Path | None = None,
    ):
        self.lat = lat
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.number_spins = len(lat.sites)
        self.symmetries = symmetries
        self.ground_state_cache_dir = ground_state_cache_dir

        self.canonical_basis = ls.SpinBasis(
            symmetries=ls.Symmetries([]),
            number_spins=self.number_spins,
            hamming_weight=self.number_spins // 2,
            spin_inversion=None,
        )

        self.canonical_basis.build()

        self.eigenstates = None
        self.eigenvalues = None
        self.ground_energy = None
        self.ground_state = None

    def unpack_configurations(self):
        """
        Unpacks all configurations in the basis into an np.array
        of one-dimensional np.arrays with 0's and 1's
        """
        return make_unpacked_configurations(self.basis.states, self.number_spins)

    def _find_cached_eigenstate(self, k) -> tuple[np.ndarray, np.ndarray] | None:
        if self.ground_state_cache_dir is None:
            return None
        for eigenstate in range(k, 20):
            eigenstate_path = self.eigenstate_path(eigenstate)
            if eigenstate_path and eigenstate_path.exists():
                print(f"Using cached version of eigenvalues / eigenstates from {eigenstate_path}")
                eigenvalues, eigenstates = pickle.loads(eigenstate_path.read_bytes())
                return eigenvalues, eigenstates

    def get_eigenstates(self, k=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Records ground_energy and ground_state into to self.ground_energy and
        self.ground_state (only one value, the smallest energy),returns k
        smallest eigenvalue / eigenvectors and records them into
        self.eigenvalues / self.eigenstates

        Returns
        -------
        eigenvalues : np.array
            k smallest eigenvalues

        eigenstates : np.array
            k eigenvectors (as vector-columns) corresponding to the k smalles
            eigenvalues

        """
        if (cached_eigenstate := self._find_cached_eigenstate(k)) is not None:
            eigenvalues, eigenstates = cached_eigenstate
        else:
            # Diagonalize the Hamiltonian using ARPACK
            print("Calculating eigenvalues / eigenstates")
            eigenvalues, eigenstates = scipy.sparse.linalg.eigsh(self.hamiltonian, k=k, which="SA")
            eigenstates = eigenstates * np.sign(eigenstates[0, :]).reshape(1, -1)
            # make sure that the first element of each eigenvector is positive
            # for reproducibility

            if self.ground_state_cache_dir is not None:
                self.ground_state_cache_dir.mkdir(exist_ok=True)
                if eigenstate_path := self.eigenstate_path(k):
                    eigenstate_path.write_bytes(pickle.dumps((eigenvalues, eigenstates)))

        print("Ground state energy is {:.10f}".format(eigenvalues[0]))

        self.ground_energy = eigenvalues[0]
        self.ground_state = eigenstates[:, 0]

        self.eigenvalues = eigenvalues
        self.eigenstates = eigenstates

        return eigenvalues, eigenstates

    def get_df_eigenstate(
        self,
        k: int,
        unpack_configurations=False,
        expand_basis_columns=False,
        canonical_basis=False,
    ) -> pd.DataFrame:
        """
        Returns the dataframe with the k'th eigenstate indexed by basis states
        Optionally, basis configurations can be unpacked
        If expand_basis_columns, they will be unpacked as separate columns
        """

        if self.eigenstates is None:
            raise ValueError(f"Eigenstate not found; run .get_eigenstates({k}) first")
        elif self.eigenstates.shape[1] <= k:
            raise ValueError(f"Not enough eigenstates found; run .get_eigenstates({k}) first")

        df = pd.DataFrame(
            dict(eigenstate_coeff=self.eigenstates[:, k]),
            index=self.basis.states,
        )

        if canonical_basis:
            df = self.transform_df_to_canonical(df)

        df["amplitude"] = np.abs(df["eigenstate_coeff"])

        if unpack_configurations:
            unpacked_configurations = make_unpacked_configurations(df.index, self.number_spins)
            if expand_basis_columns:
                spins_df = pd.DataFrame(
                    unpacked_configurations,
                    columns=[f"s{i}" for i in range(self.number_spins)],
                    index=df.index,
                )
            else:
                spins_df = pd.DataFrame(
                    dict(configuration=list(unpacked_configurations)), index=df.index
                )
            return df.join(spins_df)

        return df

    def get_df_ground_state(
        self,
        unpack_configurations=False,
        expand_basis_columns=False,
        canonical_basis=False,
    ) -> pd.DataFrame:
        """
        Alias for get_df_eigenstate(k=0, ...)
        """
        return self.get_df_eigenstate(
            k=0,
            unpack_configurations=unpack_configurations,
            expand_basis_columns=expand_basis_columns,
            canonical_basis=canonical_basis,
        )

    def transform_df_to_canonical(self, df):

        state_info_df = batched_state_info_df(self.basis, self.canonical_basis.states)

        return (
            state_info_df.merge(df, left_on="representative", right_index=True)
            .assign(
                eigenstate_adjusted=lambda x: np.real_if_close(
                    x["eigenstate_coeff"] * x["character"] * x["norm"]
                )
            )
            .drop(
                [
                    "eigenstate_coeff",
                    "representative",
                    "character",
                    "norm",
                ],
                axis=1,
            )
            .rename(
                columns={
                    "eigenstate_adjusted": "eigenstate_coeff",
                }
            )
            .reindex(self.canonical_basis.states)
        )

    def visualize_probable_configurations(self, m=0, canonical_basis=True):
        """
        Visualizes m'th most probable configuration
        """
        df = self.get_df_ground_state(
            unpack_configurations=True, canonical_basis=canonical_basis
        ).sort_values("amplitude", ascending=False)
        self.lat.plot(spins=df.iloc[m]["configuration"])
        plt.title(
            f"Plotted {m}'s most probable state, wavefunction value "
            f"= {df.iloc[m]['eigenstate_coeff']}"
        )

    def eigenstate_path(self, k: int) -> Optional[Path]:
        raise NotImplementedError


class HeisenbergJ1J2(SpinSystem):
    def __init__(
        self,
        lat: SpinLattice,
        J1: float = 1.0,
        J2: float = 1.0,
        use_symmetries=True,
        spin_inversion: Optional[int] = 1,
        ground_state_cache_dir: Path | None = None,
    ):
        J1 = float(J1)
        J2 = float(J2)
        self.J1 = J1
        self.J2 = J2
        self.use_symmetries = use_symmetries
        self.spin_inversion = spin_inversion
        number_spins = len(lat.sites)

        print(f"{number_spins=}")
        hamming_weight = number_spins // 2  # Hamming weight (i.e. number of spin ups)

        # Constructing symmetries

        if use_symmetries:
            symmetries_lst = [
                ls.Symmetry(automorphism, sector=0) for automorphism in lat.get_automorphisms()
            ]
        else:
            symmetries_lst = []

        # Constructing the group
        symmetries = ls.Symmetries(symmetries_lst)
        print("Symmetry group contains {} elements".format(len(symmetries)))

        # Constructing the basis
        basis = ls.SpinBasis(
            symmetries=symmetries,
            number_spins=number_spins,
            spin_inversion=spin_inversion,
            hamming_weight=hamming_weight,
        )

        basis.build()  # Build the list of representatives, we need it since we're doing ED
        print("Hilbert space dimension is {}".format(basis.number_states))

        # Constructing the Hamiltonian
        expr_str = "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + σᶻ₀ σᶻ₁"

        # fmt: off
        expr = (J1 * ls.Expr(expr_str, sites=lat.kind_to_edges[1]) + 
                J2 * ls.Expr(expr_str, sites=lat.kind_to_edges[2]))
        # fmt: on

        hamiltonian = ls.Operator(basis, expr)

        super().__init__(
            lat=lat,
            basis=basis,
            hamiltonian=hamiltonian,
            symmetries=symmetries,
            ground_state_cache_dir=ground_state_cache_dir,
        )

    def eigenstate_path(self, k: int) -> Path | None:
        if self.ground_state_cache_dir is None:
            return None
        return Path(
            self.ground_state_cache_dir / f"{self.__class__.__name__}-{self.lat.file_stem}-"
            f"{self.J1!r}-{self.J2!r}-{self.use_symmetries}-{self.spin_inversion}-{k}.pickle"
        )
