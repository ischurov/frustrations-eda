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
from utils import batched_state_info_df, make_unpacked_configurations


class SpinSystem:
    def __init__(
        self,
        lattice: SpinLattice,
        basis: ls.SpinBasis,
        hamiltonian: ls.Operator,
        symmetries: ls.Symmetries,
        ground_state_cache_dir: Path | None = None,
        show_progress: bool = True,
    ):
        self.lattice = lattice
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.number_spins = len(lattice.sites)
        self.symmetries = symmetries
        self.ground_state_cache_dir = ground_state_cache_dir
        self.show_progress = show_progress

        self.canonical_basis = self.lattice.get_basis(
            use_symmetries=False, hamming_weight=self.number_spins // 2, spin_inversion=None
        )

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
                if self.show_progress:
                    print(
                        f"Using cached version of eigenvalues / eigenstates from {eigenstate_path}"
                    )
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
            if self.show_progress:
                print("Calculating eigenvalues / eigenstates")
            eigenvalues, eigenstates = scipy.sparse.linalg.eigsh(self.hamiltonian, k=k, which="SA")
            eigenstates = eigenstates * np.sign(eigenstates[0, :]).reshape(1, -1)
            # make sure that the first element of each eigenvector is positive
            # for reproducibility

            if self.ground_state_cache_dir is not None:
                self.ground_state_cache_dir.mkdir(exist_ok=True)
                if eigenstate_path := self.eigenstate_path(k):
                    eigenstate_path.write_bytes(pickle.dumps((eigenvalues, eigenstates)))

        if self.show_progress:
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
        self.lattice.plot(spins=df.iloc[m]["configuration"])
        plt.title(
            f"Plotted {m}'s most probable state, wavefunction value "
            f"= {df.iloc[m]['eigenstate_coeff']}"
        )

    def eigenstate_path(self, k: int) -> Optional[Path]:
        raise NotImplementedError

    def system_id(self) -> str:
        raise NotImplementedError

    def sample_elements(
        self,
        representatives: npt.NDArray[np.uint64],
        n: int,
        prob: pd.Series | None = None,
    ) -> npt.NDArray[np.uint64]:
        """
        Sample n elements from the union of the orbits of the given representatives.
        It is guaranteed that each orbit is sampled at most once.

        Parameters
        ----------
        basis
            The basis taking account of the symmetries of the system.
        canonical_basis
            The canonical basis of the system.
        representatives
            The representatives of the orbits to sample from.
        n
            The number of elements to sample.
        prob
            The probability of sampling each state. It should be pd.DataFrame
            with keys being the states and values being the probabilities, like
            returned by `.get_df_ground_state()`.

            If None, all states are equally likely.

        Returns
        -------
        The sampled elements.
        """

        if n > len(representatives):
            raise ValueError("n must be smaller than the length of the representatives array")

        if prob is not None and (prob < 0).any():
            raise ValueError("probabilities must be positive")

        if prob is not None:
            prob = prob / np.sum(prob)

        state_info_df = batched_state_info_df(self.basis, self.canonical_basis.states).merge(
            pd.DataFrame(index=representatives),
            left_on="representative",
            right_index=True,
            how="right",
        )
        if state_info_df.isna().any().any():
            raise ValueError("Some representatives are not in the canonical basis")

        if prob is not None:
            state_info_df = state_info_df.merge(
                prob.rename("prob"), left_index=True, right_index=True, how="left"
            )
        else:
            state_info_df["prob"] = 1
        state_info_df["prob"] = state_info_df["prob"] / state_info_df["prob"].sum()

        repr_prob = state_info_df.groupby("representative")["prob"].sum()
        selected_repr = np.random.choice(repr_prob.index, size=n, replace=False, p=repr_prob)

        return (
            state_info_df.merge(
                pd.DataFrame(index=selected_repr),
                left_on="representative",
                right_index=True,
                how="right",
            )
            .reset_index()
            .rename(columns={"index": "state"})
            .groupby("representative")
            .agg(lambda x: np.random.choice(x))["state"]
            .values
        )  # type: ignore


class HeisenbergJ1J2(SpinSystem):
    def __init__(
        self,
        lattice: SpinLattice,
        J1: float = 1.0,
        J2: float = 1.0,
        use_symmetries=True,
        spin_inversion: Optional[int] = 1,
        ground_state_cache_dir: Path | None = None,
        show_progress: bool = True,
    ):
        J1 = float(J1)
        J2 = float(J2)
        self.J1 = J1
        self.J2 = J2
        self.use_symmetries = use_symmetries
        self.spin_inversion = spin_inversion
        self.show_progress = show_progress

        number_spins = len(lattice.sites)
        if self.show_progress:
            print(f"{number_spins=}")
        hamming_weight = number_spins // 2  # Hamming weight (i.e. number of spin ups)

        # Constructing symmetries

        if use_symmetries:
            symmetries = lattice.get_heisenberg_symmetries()
        else:
            symmetries = ls.Symmetries([])

        if show_progress:
            print("Symmetry group contains {} elements".format(len(symmetries)))

        # Constructing the basis
        basis = lattice.get_basis(
            use_symmetries=use_symmetries,
            spin_inversion=spin_inversion,
            hamming_weight=hamming_weight,
        )

        if show_progress:
            print("Hilbert space dimension is {}".format(basis.number_states))

        # Constructing the Hamiltonian
        expr_str = "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + σᶻ₀ σᶻ₁"

        # fmt: off
        expr = (J1 * ls.Expr(expr_str, sites=lattice.kind_to_edges[1]) + 
                J2 * ls.Expr(expr_str, sites=lattice.kind_to_edges[2]))
        # fmt: on

        hamiltonian = ls.Operator(basis, expr)

        super().__init__(
            lattice=lattice,
            basis=basis,
            hamiltonian=hamiltonian,
            symmetries=symmetries,
            ground_state_cache_dir=ground_state_cache_dir,
            show_progress=show_progress,
        )

    def eigenstate_path(self, k: int) -> Path | None:
        if self.ground_state_cache_dir is None:
            return None
        return Path(self.ground_state_cache_dir / f"{self.system_id()}-{k}.pickle")

    def system_id(self) -> str:
        return (
            f"{self.__class__.__name__}-{self.lattice.file_stem}-"
            f"{self.J1!r}-{self.J2!r}-{self.use_symmetries}-{self.spin_inversion}"
        )
