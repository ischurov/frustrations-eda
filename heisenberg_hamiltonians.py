import matplotlib.pyplot as plt
from itertools import product
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import lattice_symmetries as ls
from more_itertools import sliced
import yaml

from spin_lattices import SpinLattice


def make_unpacked_configurations(states, number_spins):
    return (
        (np.arange(2 ** number_spins).reshape(-1, 1) >> np.arange(number_spins)) & 1
    )[states]


class HeisenbergJ1J2:
    def __init__(
        self,
        lat: SpinLattice,
        J1: float = 1.0,
        J2: float = 1.0,
        use_symmetries=True,
        spin_inversion=1,
    ):
        self.lat = lat

        self.number_spins = len(lat.sites)  # System size

        print(f"{self.number_spins=}")
        hamming_weight = (
            self.number_spins // 2
        )  # Hamming weight (i.e. number of spin ups)

        # Constructing symmetries

        if use_symmetries:
            symmetries = [
                ls.Symmetry(automorphism, sector=0)
                for automorphism in lat.get_automorphisms()
            ]
        else:
            symmetries = []

        # Constructing the group
        self.symmetry_group = ls.Group(symmetries)
        print("Symmetry group contains {} elements".format(len(self.symmetry_group)))

        # Constructing the basis
        self.basis = ls.SpinBasis(
            self.symmetry_group,
            number_spins=self.number_spins,
            hamming_weight=hamming_weight,
            spin_inversion=spin_inversion,
        )
        self.basis.build()  # Build the list of representatives, we need it since we're doing ED
        print("Hilbert space dimension is {}".format(self.basis.number_states))

        self.canonical_basis = ls.SpinBasis(
            ls.Group([]),
            number_spins=self.number_spins,
            hamming_weight=hamming_weight,
            spin_inversion=None,
        )

        # this can probably be optimized / avoided
        self.canonical_basis.build()

        # Heisenberg Hamiltonian
        # fmt: off
        σ_x = np.array([ [0, 1]
                       , [1, 0] ])
        σ_y = np.array([ [0 , -1j]
                       , [1j,   0] ])
        σ_z = np.array([ [1,  0]
                       , [0, -1] ])
        # fmt: on
        σ_p = σ_x + 1j * σ_y
        σ_m = σ_x - 1j * σ_y

        matrix = 0.5 * (np.kron(σ_p, σ_m) + np.kron(σ_m, σ_p)) + np.kron(σ_z, σ_z)

        self.hamiltonian = ls.Operator(
            self.basis,
            [
                ls.Interaction(J * matrix, edges)
                for J, (_, edges) in zip([J1, J2], sorted(lat.kind_to_edges.items()))
            ],
        )

        self.ground_state = None
        self.ground_energy = None

    def unpack_configurations(self):
        """
        Unpacks all configurations in the basis into an np.array
        of one-dimensional np.arrays with 0's and 1's
        """
        return make_unpacked_configurations(self.basis.states, self.number_spins)

    def get_ground_state(self, k=1) -> tuple[np.array, np.array]:
        """
        Records ground_energy and ground_state into to self.ground_energy and
        self.ground_state (only one value, the smallest energy) and returns k
        smallest eigenvalue / eigenvectors

        Returns
        -------
        eigenvalues : np.array
            k smallest eigenvalues

        eigenstates : np.array
            k eigenvectors (as vector-columns) corresponding to the k smalles
            eigenvalues

        """
        # Diagonalize the Hamiltonian using ARPACK
        eigenvalues, eigenstates = ls.diagonalize(self.hamiltonian, k=k)
        print("Ground state energy is {:.10f}".format(eigenvalues[0]))

        self.ground_energy = eigenvalues[0]
        self.ground_state = eigenstates[:, 0]

        return eigenvalues, eigenstates
        # assert np.isclose(eigenvalues[0], -18.06178542)

    def get_df_ground_state(
        self,
        unpack_configurations=False,
        expand_basis_columns=False,
        canonical_basis=False,
    ) -> pd.DataFrame:
        """
        Returns the dataframe with two columns: unpacked configuration and the corresponding
        value in the ground state
        """
        if self.ground_state is None:
            raise ValueError("Ground State not found; run .get_ground_state() first")

        df = pd.DataFrame(
            dict(basis_state=self.basis.states, ground_state_coeff=self.ground_state)
        )

        if canonical_basis:
            df = self.transform_df_to_canonical(df)

        df["amplitude"] = np.abs(df["ground_state_coeff"])

        if unpack_configurations:
            unpacked_configurations = make_unpacked_configurations(
                df["basis_state"], self.number_spins
            )
            if expand_basis_columns:
                spins_df = pd.DataFrame(
                    unpacked_configurations,
                    columns=[f"s{i}" for i in range(self.number_spins)],
                )
            else:
                spins_df = pd.DataFrame(
                    dict(configuration=list(unpacked_configurations))
                )
            return df.join(spins_df)

        return df

    def transform_df_to_canonical(self, df):
        def pad_right(arr, n):
            # Pad the array with zeros to the left to create an n x 8 matrix
            arr = arr.reshape(-1, 1)
            return np.pad(arr, [(0, 0), (0, n - 1)], "constant", constant_values=0)

        assert (
            pad_right(np.array([1, 2, 3]), 8)
            == np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        ).all()

        def my_batched_state_info(basis, bits):
            representative, eigenvalue, norm = basis.batched_state_info(
                pad_right(bits, 8)
            )
            representative = representative[:, 0]
            return representative, eigenvalue, norm

        representative, eigenvalue, norm = my_batched_state_info(
            self.basis, self.canonical_basis.states
        )

        state_info_df = pd.DataFrame(
            dict(
                basis_state=self.canonical_basis.states,
                representative=representative,
                character=eigenvalue,
                norm=norm,
            )
        )

        return (
            state_info_df.merge(df, left_on="representative", right_on="basis_state")
            .assign(
                ground_state_adjusted=lambda x: np.real_if_close(
                    x["ground_state_coeff"] * x["character"] * x["norm"]
                )
            )
            .drop(
                [
                    "ground_state_coeff",
                    "basis_state_y",
                    "representative",
                    "character",
                    "norm",
                ],
                axis=1,
            )
            .rename(
                columns={
                    "ground_state_adjusted": "ground_state_coeff",
                    "basis_state_x": "basis_state",
                }
            )
        )

    def visualize_probable_configurations(self, k=0):
        """
        Visualizes k'th most probable configuration
        """
        df = self.get_df_ground_state(unpack_configurations=True).sort_values(
            "amplitude", ascending=False
        )
        self.lat.plot(spins=df.iloc[k]["configuration"])
        plt.title(
            f"Plotted {k}'s most probable state, wavefunction value "
            f"= {df.iloc[k]['ground_state_coeff']}"
        )
