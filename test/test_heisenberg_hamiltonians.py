from unittest import TestCase

import lattice_symmetries as ls
import numpy as np

from heisenberg_hamiltonians import HeisenbergJ1J2
from spin_lattices import ChainLattice, KagomeLattice, SpinLattice, SquareLattice


class TestDiagonalization(TestCase):
    def test_diagonalization_nosym(self):
        system_nosym = HeisenbergJ1J2(
            ChainLattice(width=12),
            J1=1,
            J2=0,
            use_symmetries=False,
            spin_inversion=None,
        )
        energy, state = system_nosym.get_eigenstates()
        self.assertTrue(np.isclose(energy, -21.5495636698).all())

    def test_diagonalization_sym(self):
        system_sym = HeisenbergJ1J2(
            ChainLattice(width=12),
            J1=1,
            J2=0,
            use_symmetries=True,
            spin_inversion=1,
        )
        energy, state = system_sym.get_eigenstates()
        self.assertTrue(np.isclose(energy, -21.5495636698).all())

    def test_canonical_basis(self):
        J2 = 0.7
        for lattice in [
            SquareLattice(width=4, height=4),
            # KagomeLattice(width=2, height=2),
            KagomeLattice(width=2, height=3),
        ]:
            print(lattice.__class__.__name__)
            system = HeisenbergJ1J2(
                lattice=lattice,
                J1=1,
                J2=J2,
                use_symmetries=True,
                spin_inversion=1,
            )
            system.get_eigenstates(1)

            system_nosym = HeisenbergJ1J2(
                lattice=lattice,
                J1=1,
                J2=J2,
                use_symmetries=False,
                spin_inversion=None,
            )
            system_nosym.get_eigenstates(1)

            self.assertTrue(
                system_nosym.get_df_eigenstate(0)
                .join(
                    system.get_df_eigenstate(0, canonical_basis=True),
                    how="outer",
                    lsuffix="_x",
                    rsuffix="_y",
                )
                .assign(ok=lambda x: np.isclose(x["eigenstate_coeff_x"], x["eigenstate_coeff_y"]))[
                    "ok"
                ]
                .all()
            )

            # Order of elements in the dataframe correponds to order of elements in basis.states
            self.assertTrue(
                (
                    system.get_df_eigenstate(0, canonical_basis=True).index
                    == system.canonical_basis.states
                ).all()
            )
            self.assertTrue(
                (
                    system.get_df_eigenstate(0, canonical_basis=False).index == system.basis.states
                ).all()
            )

    def test_neel(self):
        """
        Tests whether the most probable state for the unfrustrated Heisenberg model
        is Néel (i.e. checkerboard pattern)
        """
        for use_symmetries in [True, False]:
            system = HeisenbergJ1J2(
                SquareLattice(width=4, height=4),
                J1=1,
                J2=0,
                use_symmetries=use_symmetries,
                spin_inversion=1 if use_symmetries else None,
            )
            system.get_eigenstates()
            most_probable_config = (
                system.get_df_ground_state(unpack_configurations=True, canonical_basis=True)
                .sort_values("amplitude", ascending=False)
                .iloc[0]["configuration"]
            )
            self.assertTrue(
                all(
                    [
                        most_probable_config[i] != most_probable_config[j]
                        for i, j in system.lattice.kind_to_edges[1]
                    ]
                )
            )

    def test_get_eigenstate_in_full_basis(self):
        for lattice in [ChainLattice(8), SquareLattice(4, 4), KagomeLattice(2, 3)]:
            system_nosym = HeisenbergJ1J2(
                lattice,
                J1=1,
                J2=0.7,
                use_symmetries=False,
                spin_inversion=None,
            )
            system_nosym.get_eigenstates(1)

            system_sym = HeisenbergJ1J2(
                lattice,
                J1=1,
                J2=0.7,
                use_symmetries=True,
                spin_inversion=None,
            )
            system_sym.get_eigenstates(1)

            eigenstate_in_full_basis = system_sym.get_ground_state_in_full_basis()
            np.testing.assert_allclose(
                eigenstate_in_full_basis[system_sym.canonical_basis.states],
                system_nosym.get_eigenstates(2)[1][:, 0],
            )

            eigenstate_in_full_basis[system_sym.canonical_basis.states] = 0
            self.assertTrue(np.isclose(eigenstate_in_full_basis, 0).all())


class TestHamiltonianProperties(TestCase):
    def test_invariance(self):
        system = HeisenbergJ1J2(
            SquareLattice(width=4, height=4),
            J1=1,
            J2=2,
            use_symmetries=True,
            spin_inversion=1,
        )

        for generator in system.symmetries.generators:
            self.assertTrue(
                system.hamiltonian.expression
                == system.hamiltonian.expression.replace_indices(
                    dict(enumerate(generator.permutation))
                )
            )

    def test_heisenberg(self):
        J1 = 1
        J2 = 2

        system = HeisenbergJ1J2(
            SquareLattice(width=4, height=4),
            J1=J1,
            J2=J2,
            use_symmetries=True,
            spin_inversion=1,
        )

        expr_str = "σˣ₀ σˣ₁ + σʸ₀ σʸ₁ + σᶻ₀ σᶻ₁"
        # fmt: off
        expr = (J1 * ls.Expr(expr_str, sites=system.lattice.kind_to_edges[1]) + 
                J2 * ls.Expr(expr_str, sites=system.lattice.kind_to_edges[2]))
        # fmt: on

        self.assertTrue(system.hamiltonian.expression == expr)
        self.assertFalse(system.hamiltonian.expression == 2 * expr)
