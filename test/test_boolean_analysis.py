from unittest import TestCase

import lattice_symmetries as ls
import numpy as np
import pandas as pd
import pandas.testing as pdt

from boolean_analysis import BooleanFourierAnalyser, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from spin_lattices import ChainLattice, KagomeLattice, SpinLattice, SquareLattice
from utils import make_unpacked_configurations


class TestBooleanFourierAnalyser(TestCase):
    def test_marshall(self):
        analyzer_sym = BooleanFourierAnalyser(
            system=HeisenbergJ1J2(
                SquareLattice(width=4, height=4),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer_sym.fit(x=analyzer_sym.system.canonical_basis.states)
        # first_subset_idx = analyzer_sym.learner.get_coeffs_ser().index[0]
        # assert isinstance(first_subset_idx, np.uint64)
        first_subset = make_unpacked_configurations(
            analyzer_sym.learner.get_coeffs_ser().index[0],  # type: ignore
            analyzer_sym.system.number_spins,
        )[0]

        self.assertTrue(
            all(
                [
                    first_subset[i] != first_subset[j]
                    for i, j in analyzer_sym.system.lat.kind_to_edges[1]
                ]
            )
        )  # bipartite sublattices

        self.assertTrue(
            np.isclose(
                analyzer_sym.set_truncate_strategy(keep_largest_n(1)).prediction_score(
                    analyzer_sym.system.canonical_basis.states,
                    scorer="overlap",
                ),
                1,
            )
        )  # first term is enough to reconstruct the sign structure

        self.assertTrue(
            np.isclose(
                analyzer_sym.set_truncate_strategy(keep_largest_n(1)).prediction_score(
                    analyzer_sym.system.canonical_basis.states,
                    scorer="accuracy",
                ),
                1,
            )
        )

    def test_marshall_kagome(self):
        system = HeisenbergJ1J2(
            KagomeLattice(width=2, height=3),
            J1=1,
            J2=0,
            use_symmetries=True,
            spin_inversion=1,
        )
        analyzer = BooleanFourierAnalyser(
            system=system,
            use_subset_symmetries=True,
        )
        analyzer.fit(system.basis.states)
        pdt.assert_series_equal(
            analyzer.set_truncate_strategy(keep_largest_n(1)).get_expanded_coeffs_ser(),
            pd.Series(
                [-0.185471, 0.185471],
                index=pd.Series(np.array([5285, 256858], dtype="uint64")).rename("state"),
            ).rename("coeff"),
        )

    def test_symmetries(self):
        lattices = [
            SquareLattice(width=4, height=4),
            SquareLattice(width=3, height=2),
            ChainLattice(width=6),
            ChainLattice(width=8),
        ]
        J1 = 1
        J2 = 0.5
        for lat in lattices:
            analyzer_sym = BooleanFourierAnalyser(
                system=HeisenbergJ1J2(
                    lat,
                    J1=J1,
                    J2=J2,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                use_subset_symmetries=True,
            )
            analyzer_sym.fit(x=analyzer_sym.system.canonical_basis.states)

            analyzer_nosym = BooleanFourierAnalyser(
                system=HeisenbergJ1J2(
                    lat,
                    J1=J1,
                    J2=J2,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                use_subset_symmetries=False,
            )
            analyzer_nosym.fit(x=analyzer_nosym.system.canonical_basis.states)

            self.assertTrue(
                np.allclose(
                    analyzer_sym.predict(analyzer_sym.system.canonical_basis.states),
                    analyzer_nosym.predict(analyzer_sym.system.canonical_basis.states),
                )
            )

    def test_normalization(self):
        analyzer = BooleanFourierAnalyser(
            system=HeisenbergJ1J2(
                SquareLattice(width=4, height=2),
                J1=1,
                J2=0.5,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(x=analyzer.system.canonical_basis.states)
        self.assertTrue(
            np.allclose(np.abs(analyzer.predict(analyzer.system.canonical_basis.states)), 1)
        )
