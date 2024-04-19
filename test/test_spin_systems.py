import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from unittest import TestCase

import lattice_symmetries as ls
import numpy as np
from sympy import Rational
from spin_systems import (
    SpinSystem,
    basis_factory,
    spin_system,
    zero_sector_basis,
    no_symmetries_basis,
    heisenberg,
    ground_state_basis,
)
from spin_lattices import (
    ChainLattice,
    KagomeLattice,
    ParallelogramSpinLattice,
    SpinLattice,
    SquareLattice,
    TriangularLattice,
)


class TestDiagonalization(TestCase):
    def test_diagonalization_nosym(self):
        system_nosym = spin_system(
            heisenberg(ChainLattice(width=12), J1=1, J2=0),
            basis=no_symmetries_basis(),
            ground_state_cache_dir=None,
        )
        energy = system_nosym.ground_energy
        self.assertTrue(np.isclose(energy, -21.5495636698))

    def test_diagonalization_sym(self):
        system_sym = spin_system(
            heisenberg(ChainLattice(width=12), J1=1, J2=0),
            basis=zero_sector_basis(spin_inversion=1),
            ground_state_cache_dir=None,
        )
        energy = system_sym.ground_energy
        self.assertTrue(np.isclose(energy, -21.5495636698))

    # def test_canonical_basis(self):
    #     J2 = 0.7
    #     for lattice in [
    #         SquareLattice(width=2, height=2),
    #         SquareLattice(width=2, height=4),
    #         SquareLattice(width=4, height=4),
    #         TriangularLattice(width=2, height=2),
    #         TriangularLattice(width=2, height=4),
    #         TriangularLattice(width=4, height=4),
    #         ChainLattice(width=4),
    #         ChainLattice(width=8),
    #         ChainLattice(width=12),
    #         # SquareLattice(width=4, height=6),
    #         # KagomeLattice(width=2, height=4),
    #         # KagomeLattice(width=4, height=2),
    #         # TriangleLattice(width=4, height=6),
    #         # the last are known to be working, but takes too long
    #     ]:
    #         print(lattice.__class__.__name__)
    #         system = HeisenbergJ1J2(
    #             lattice=lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=True,
    #             spin_inversion=1,
    #             skip_symmetries_whitelist=True,
    #             ground_state_cache_dir=None,
    #         )
    #         system.get_eigenstates(1)

    #         system_nosym = HeisenbergJ1J2(
    #             lattice=lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=False,
    #             spin_inversion=None,
    #             ground_state_cache_dir=None,
    #         )
    #         system_nosym.get_eigenstates(1)

    #         self.assertTrue(
    #             system_nosym.get_df_eigenstate(0)
    #             .join(
    #                 system.get_df_eigenstate(0, canonical_basis=True),
    #                 how="outer",
    #                 lsuffix="_x",
    #                 rsuffix="_y",
    #             )
    #             .assign(
    #                 ok=lambda x: np.isclose(
    #                     x["eigenstate_coeff_x"], x["eigenstate_coeff_y"]
    #                 )
    #             )["ok"]
    #             .all()
    #         )

    #         # Order of elements in the dataframe correponds to order of elements in basis.states
    #         self.assertTrue(
    #             (
    #                 system.get_df_eigenstate(0, canonical_basis=True).index
    #                 == system.canonical_basis.states
    #             ).all()
    #         )
    #         self.assertTrue(
    #             (
    #                 system.get_df_eigenstate(0, canonical_basis=False).index
    #                 == system.basis.states
    #             ).all()
    #         )

    def test_neel(self):
        """
        Tests whether the most probable state for the unfrustrated Heisenberg model
        is Néel (i.e. checkerboard pattern)
        """

        system = spin_system(
            heisenberg(SquareLattice(width=4, height=4), J1=1, J2=0),
            basis=no_symmetries_basis(),
            ground_state_cache_dir=None,
        )
        ground_state = system.ground_state
        most_probable_config = system.lattice.unpack_configurations(
            np.array([system.basis.states[np.argmax(np.abs(ground_state))]])
        )[0]

        self.assertTrue(
            all(
                [
                    most_probable_config[i] != most_probable_config[j]
                    for i, j in system.lattice.kind_to_edges[1]
                ]
            )
        )

    def test_symmetries_square(self):
        lattice = SquareLattice(4, 4)
        system_nosym = spin_system(
            heisenberg(lattice),
            basis=no_symmetries_basis(),
            ground_state_cache_dir=None,
        )
        bases = [
            zero_sector_basis(),
            basis_factory(
                lambda expr: [
                    (expr.lattice.x_translation, Rational(0)),
                    (expr.lattice.y_translation, Rational(0)),
                ]
            ),
            basis_factory(
                lambda expr: [
                    (expr.lattice.x_translation, Rational(0)),
                ]
            ),
            basis_factory(
                lambda expr: [
                    (expr.lattice.y_translation, Rational(0)),
                ]
            ),
        ]
        for basis in bases:
            system_sym = spin_system(
                heisenberg(lattice), basis=basis, ground_state_cache_dir=None
            )
            self.assertAlmostEqual(system_nosym.ground_energy, system_sym.ground_energy)

    def test_symmetries_kagome(self):
        expr = heisenberg(KagomeLattice(2, 4), J1=1, J2=1)
        system_sym = spin_system(
            expr,
            basis=basis_factory(
                lambda expr: [
                    (expr.lattice.x_translation, Rational(0, 1)),
                    (expr.lattice.y_translation, Rational(1, 2)),
                ]
            ),
            ground_state_cache_dir=None,
        )
        system_nosym = spin_system(
            expr, basis=no_symmetries_basis(), ground_state_cache_dir=None
        )
        self.assertAlmostEqual(system_sym.ground_energy, system_nosym.ground_energy)

    def test_to_ground_state_sector(self):
        lattice = KagomeLattice(2, 2)
        system = spin_system(
            heisenberg(lattice, J1=1, J2=1),
            basis=no_symmetries_basis(),
            ground_state_cache_dir=None,
        )
        system_ground_state_sector = system.to_ground_state_sector()
        self.assertAlmostEqual(
            system.ground_energy,
            system_ground_state_sector.ground_energy,
        )
        self.assertTrue(
            any(
                moment != 0
                for permutation, moment in system_ground_state_sector.basis.symmetries
            )
        )

    def test_ground_state_basis(self):
        lattice = KagomeLattice(2, 2)
        system_nosym = spin_system(
            heisenberg(lattice, J1=1, J2=1),
            basis=no_symmetries_basis(),
            ground_state_cache_dir=None,
        )
        system_ground_state_basis = spin_system(
            heisenberg(lattice, J1=1, J2=1),
            basis=ground_state_basis(),
            ground_state_cache_dir=None,
        )
        self.assertAlmostEqual(
            system_nosym.ground_energy,
            system_ground_state_basis.ground_energy,
        )

    # def test_get_eigenstate_in_full_basis(self):
    #     J2 = 0.5
    #     for lattice in [SquareLattice(4, 4), SquareLattice(2, 4)]:
    #         system_nosym = HeisenbergJ1J2(
    #             lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=False,
    #             spin_inversion=None,
    #         )
    #         system_nosym.get_eigenstates(1)

    #         system_sym = HeisenbergJ1J2(
    #             lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=True,
    #             spin_inversion=1,
    #         )
    #         system_sym.get_eigenstates(1)

    #         eigenstate_in_full_basis = system_sym.get_ground_state_in_full_basis()
    #         np.testing.assert_allclose(
    #             eigenstate_in_full_basis[system_sym.canonical_basis.states],
    #             system_nosym.get_eigenstates(1)[1][:, 0],
    #             atol=1e-15,
    #             rtol=10,
    #             # Small coefficients can be found with some numeric error,
    #             # so relative tolerance is so high
    #         )

    #         eigenstate_in_full_basis[system_sym.canonical_basis.states] = 0
    #         self.assertTrue((eigenstate_in_full_basis == 0).all())

    # def test_get_eigenstate_coeffs(self):
    #     J2 = 0.5
    #     for lattice in [SquareLattice(4, 4), SquareLattice(2, 4)]:
    #         system_nosym = HeisenbergJ1J2(
    #             lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=False,
    #             spin_inversion=None,
    #         )
    #         system_nosym.get_eigenstates(1)

    #         system_sym = HeisenbergJ1J2(
    #             lattice,
    #             J1=1,
    #             J2=J2,
    #             use_symmetries=True,
    #             spin_inversion=1,
    #         )
    #         system_sym.get_eigenstates(1)

    #         eigenstate_in_full_basis = system_sym.get_ground_state_in_full_basis()
    #         np.testing.assert_allclose(
    #             eigenstate_in_full_basis[system_sym.canonical_basis.states],
    #             system_nosym.get_ground_state_coeffs(system_sym.canonical_basis.states),
    #             atol=1e-15,
    #             rtol=10,
    #             # Small coefficients can be found with some numeric error,
    #             # so relative tolerance is so high
    #         )

    #         eigenstate_in_full_basis[system_sym.canonical_basis.states] = 0
    #         self.assertTrue((eigenstate_in_full_basis == 0).all())

    def test_lattice_equivalence(self):
        class OneDiagonalSquareLattice(ParallelogramSpinLattice):
            def __init__(self, width=1, height=1):
                r"""
                Generates square J1-J2 lattice with one diagonal
                (Should be equivalent to triangle lattice)

                The fundamental domain:

                ```
                C ----- D
                | \\    |
                |  \\   |
                |   \\  |
                |    \\ |
                A ----- B
                ```

                Size of the fundamentail domain is 1×1
                """
                u = np.array([1, 0])
                v = np.array([0, 1])

                named_sites = {
                    "A": np.array([0, 0]),
                    "B": np.array([1, 0]),
                    "C": np.array([0, 1]),
                    "D": np.array([1, 1]),
                }

                named_edges = [("AB", 1), ("AC", 1), ("CD", 1), ("BD", 1), ("CB", 2)]

                super().__init__(
                    u=u,
                    v=v,
                    named_sites=named_sites,
                    named_edges=named_edges,
                    fundamental_domain_size=1,
                    width=width,
                    height=height,
                )

        self.assertTrue(
            heisenberg(lattice=TriangularLattice(3, 3)).expr
            == heisenberg(lattice=OneDiagonalSquareLattice(3, 3)).expr
        )


class TestHamiltonianProperties(TestCase):
    def test_invariance(self):
        expression = heisenberg(
            SquareLattice(width=4, height=4),
            J1=1,
            J2=2,
        )

        for generator in expression.expr.permutation_group():
            self.assertTrue(
                expression.expr
                == expression.expr.replace_indices(dict(enumerate(generator)))
            )

    def test_heisenberg(self):
        J1 = 1
        J2 = 2

        expression = heisenberg(
            SquareLattice(width=4, height=4),
            J1=J1,
            J2=J2,
        )

        expr_str = "σˣ₀ σˣ₁ + σʸ₀ σʸ₁ + σᶻ₀ σᶻ₁"
        # fmt: off
        expr = (J1 * ls.Expr(expr_str, sites=expression.lattice.kind_to_edges[1]) + 
                J2 * ls.Expr(expr_str, sites=expression.lattice.kind_to_edges[2]))
        # fmt: on

        self.assertTrue(expression.expr == expr)
        self.assertFalse(expression.expr == 2 * expr)
