import pickle
from pathlib import Path
from typing import Optional

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from spin_lattices import SpinLattice
from utils import make_unpacked_configurations

GROUND_STATE_DIR = Path("groundstates")  # type: ignore


def pad_right(arr, n):
    # Pad the array with zeros to the left to create an n x 8 matrix
    arr = arr.reshape(-1, 1)
    return np.pad(arr, [(0, 0), (0, n - 1)], "constant", constant_values=0)


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
    representative, eigenvalue, norm = basis.batched_state_info(pad_right(states, 8))
    representative = representative[:, 0]
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
        symmetry_group: ls.Group,
    ):
        self.lat = lat
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.number_spins = len(lat.sites)
        self.symmetry_group = symmetry_group

        self.canonical_basis = ls.SpinBasis(
            ls.Group([]),
            number_spins=self.number_spins,
            hamming_weight=basis.hamming_weight,
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
        for eigenstate in range(k, 20):
            eigenstate_path = self.eigenstate_path(eigenstate)
            if eigenstate_path and eigenstate_path.exists():
                print(
                    f"Using cached version of eigenvalues / eigenstates from {eigenstate_path}"
                )
                eigenvalues, eigenstates = pickle.loads(eigenstate_path.read_bytes())
                break
        else:
            # Diagonalize the Hamiltonian using ARPACK
            print("Calculating eigenvalues / eigenstates")
            eigenvalues, eigenstates = ls.diagonalize(self.hamiltonian, k=k)
            eigenstates = eigenstates * np.sign(eigenstates[0, :]).reshape(1, -1)
            # make sure that the first element of each eigenvector is positive
            # for reproducibility

            GROUND_STATE_DIR.mkdir(exist_ok=True)
            if eigenstate_path := self.eigenstate_path(k):
                eigenstate_path.write_bytes(pickle.dumps((eigenvalues, eigenstates)))

        print("Ground state energy is {:.10f}".format(eigenvalues[0]))

        self.ground_energy = eigenvalues[0]
        self.ground_state = eigenstates[:, 0]

        self.eigenvalues = eigenvalues
        self.eigenstates = eigenstates

        return eigenvalues, eigenstates
        # assert np.isclose(eigenvalues[0], -18.06178542)

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
            raise ValueError(
                f"Not enough eigenstates found; run .get_eigenstates({k}) first"
            )

        df = pd.DataFrame(
            dict(eigenstate_coeff=self.eigenstates[:, k]),
            index=self.basis.states,
        )

        if canonical_basis:
            df = self.transform_df_to_canonical(df)

        df["amplitude"] = np.abs(df["eigenstate_coeff"])

        if unpack_configurations:
            unpacked_configurations = make_unpacked_configurations(
                df.index, self.number_spins
            )
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
    # noinspection NonAsciiCharacters
    def __init__(
        self,
        lat: SpinLattice,
        J1: float = 1.0,
        J2: float = 1.0,
        use_symmetries=True,
        spin_inversion: Optional[int] = 1,
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
            symmetries = [
                ls.Symmetry(automorphism, sector=0)
                for automorphism in lat.get_automorphisms()
            ]
        else:
            symmetries = []

        # Constructing the group
        symmetry_group = ls.Group(symmetries)
        print("Symmetry group contains {} elements".format(len(symmetry_group)))

        # Constructing the basis
        basis = ls.SpinBasis(
            symmetry_group,
            number_spins=number_spins,
            hamming_weight=hamming_weight,
            spin_inversion=spin_inversion,
        )

        basis.build()  # Build the list of representatives, we need it since we're doing ED
        print("Hilbert space dimension is {}".format(basis.number_states))

        # Heisenberg Hamiltonian
        # fmt: off
        σ_x = np.array([ [0, 1]
                       , [1, 0] ])
        σ_y = np.array([ [0j, -1j]
                       , [1j,   0j] ])
        σ_z = np.array([ [1,  0]
                       , [0, -1] ])
        # fmt: on
        σ_p = σ_x + 1j * σ_y
        σ_m = σ_x - 1j * σ_y

        matrix = 0.5 * (np.kron(σ_p, σ_m) + np.kron(σ_m, σ_p)) + np.kron(σ_z, σ_z)

        hamiltonian = ls.Operator(
            basis,
            [
                ls.Interaction(J * matrix, edges)
                for J, (_, edges) in zip([J1, J2], sorted(lat.kind_to_edges.items()))
            ],
        )

        super().__init__(
            lat=lat, basis=basis, hamiltonian=hamiltonian, symmetry_group=symmetry_group
        )

    def eigenstate_path(self, k: int):
        return (
            GROUND_STATE_DIR / f"{self.__class__.__name__}-{self.lat.file_stem}-"
            f"{self.J1!r}-{self.J2!r}-{self.use_symmetries}-{self.spin_inversion}-{k}.pickle"
        )
