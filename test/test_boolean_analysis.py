import tempfile
from pathlib import Path
from unittest import TestCase

import lattice_symmetries as ls
import numpy as np
import pandas as pd
import pandas.testing as pdt

from boolean_analysis import (
    AmplitudeSignalKind,
    BooleanFourierAnalyzer,
    SignalOption,
    SignSignalKind,
    ValueSignalKind,
    keep_everything,
    keep_largest_n,
)
from heisenberg_hamiltonians import HeisenbergJ1J2
from parity import popcount
from spin_lattices import ChainLattice, KagomeLattice, SpinLattice, SquareLattice
from utils import make_unpacked_configurations


class TestBooleanFourierAnalyzer(TestCase):
    def test_marshall(self):
        analyzer_sym = BooleanFourierAnalyzer(
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
                    for i, j in analyzer_sym.system.lattice.kind_to_edges[1]
                ]
            )
        )  # bipartite sublattices

        self.assertTrue(
            np.isclose(
                analyzer_sym.truncate(keep_largest_n(1)).prediction_score(
                    scorer="sign_overlap",
                ),
                1,
            )
        )  # first term is enough to reconstruct the sign structure

        self.assertTrue(
            np.isclose(
                analyzer_sym.truncate(keep_largest_n(1)).prediction_score(
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
        analyzer = BooleanFourierAnalyzer(
            system=system,
            use_subset_symmetries=True,
        )
        analyzer.fit(system.basis.states)
        pdt.assert_series_equal(
            analyzer.truncate(keep_largest_n(1)).get_expanded_coeffs_ser(),
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
            analyzer_sym = BooleanFourierAnalyzer(
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

            self.assertTrue(
                (
                    popcount(np.asarray(analyzer_sym.learner.get_coeffs_ser().index))
                    <= analyzer_sym.system.number_spins // 2
                ).all()
            )

            analyzer_nosym = BooleanFourierAnalyzer(
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
                    analyzer_sym.truncate(keep_everything).predict(
                        analyzer_sym.system.canonical_basis.states
                    ),
                    analyzer_nosym.truncate(keep_everything).predict(
                        analyzer_sym.system.canonical_basis.states
                    ),
                )
            )

    def test_normalization(self):
        analyzer = BooleanFourierAnalyzer(
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
            np.allclose(
                np.abs(
                    analyzer.truncate(keep_everything).predict(
                        analyzer.system.canonical_basis.states
                    )
                ),
                1,
            )
        )

    def test_learn_value(self):
        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=2),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(
            analyzer.system.canonical_basis.states, signal_opt=SignalOption(kind=ValueSignalKind())
        )
        prediction = analyzer.truncate(keep_everything).predict(analyzer.system.basis.states)
        true = (
            analyzer.system.get_df_ground_state(canonical_basis=True)
            .loc[analyzer.system.basis.states]["eigenstate_coeff"]
            .values
        )
        assert analyzer.system.ground_state is not None
        self.assertTrue(np.allclose(true, prediction))
        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="neg_mse", x=analyzer.system.basis.states
                ),
                0,
            )
        )
        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="value_overlap", x=analyzer.system.basis.states
                ),
                1,
            )
        )

    def test_learn_amplitude(self):
        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=2),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(
            analyzer.system.canonical_basis.states,
            signal_opt=SignalOption(kind=AmplitudeSignalKind()),
        )
        prediction = analyzer.truncate(keep_everything).predict(analyzer.system.basis.states)
        true = (
            analyzer.system.get_df_ground_state(canonical_basis=True)
            .loc[analyzer.system.basis.states]["amplitude"]
            .values
        )
        assert analyzer.system.ground_state is not None
        self.assertTrue(np.allclose(true, prediction))
        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="neg_mse",
                    x=analyzer.system.basis.states,
                ),
                0,
            )
        )
        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="value_overlap", x=analyzer.system.basis.states
                ),
                1,
            )
        )

    def test_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            analyzer = BooleanFourierAnalyzer(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=2),
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                use_subset_symmetries=True,
                cache_dir=Path(cache_dir),
            )
            analyzer.fit(
                analyzer.system.canonical_basis.states,
                signal_opt=SignalOption(kind=AmplitudeSignalKind()),
            )
            predict1 = analyzer.truncate(keep_everything).predict(analyzer.system.basis.states)

            analyzer = BooleanFourierAnalyzer(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=2),
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                use_subset_symmetries=True,
                cache_dir=Path(cache_dir),
            )
            analyzer.fit(
                analyzer.system.canonical_basis.states,
                signal_opt=SignalOption(kind=ValueSignalKind()),
            )
            analyzer.fit(
                analyzer.system.canonical_basis.states,
                signal_opt=SignalOption(kind=AmplitudeSignalKind()),
                from_cache_only=True,
            )
            predict2 = analyzer.truncate(keep_everything).predict(analyzer.system.basis.states)
            self.assertTrue(np.allclose(predict1, predict2))
            with self.assertRaises(ValueError):
                analyzer.fit(
                    analyzer.system.canonical_basis.states,
                    signal_opt=SignalOption(kind=SignSignalKind()),
                    from_cache_only=True,
                )

    def test_how_many_terms_to_achieve_score(self):
        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(
            analyzer.system.canonical_basis.states,
            signal_opt=SignalOption(kind=SignSignalKind()),
        )

        terms = analyzer.how_many_terms_to_achieve_score(
            scorer="sign_overlap", target_score=0.95, min_terms=1, max_terms=101, step=1
        )[0]

        assert terms is not None

        self.assertTrue(
            analyzer.truncate(keep_largest_n(terms)).prediction_score("sign_overlap") >= 0.95
        )
        self.assertTrue(
            analyzer.truncate(keep_largest_n(terms - 1)).prediction_score("sign_overlap") < 0.95
        )

    def test_how_many_terms_to_achieve_score2(self):
        # Marshall

        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(analyzer.system.canonical_basis.states)

        self.assertEqual(
            analyzer.how_many_terms_to_achieve_score(scorer="accuracy", target_score=0.99)[0], 1
        )
        self.assertEqual(
            analyzer.how_many_terms_to_achieve_score(scorer="sign_overlap", target_score=0.99)[0],
            1,
        )
        self.assertTrue(
            analyzer.how_many_terms_to_achieve_score(scorer="accuracy", target_score=1.01)[0]
            is None
        )

        # Frustrated

        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(analyzer.system.canonical_basis.states)

        def assert_is_large(terms: int | None):
            if terms is not None:
                self.assertTrue(terms > 1)

        assert_is_large(
            analyzer.how_many_terms_to_achieve_score(scorer="accuracy", target_score=0.99)[0]
        )
        assert_is_large(
            analyzer.how_many_terms_to_achieve_score(scorer="sign_overlap", target_score=0.99)[0]
        )

    def test_predict_max_batch_size(self):
        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=2),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            use_subset_symmetries=True,
        )
        analyzer.fit(analyzer.system.canonical_basis.states)

        self.assertTrue(
            np.allclose(
                analyzer.truncate(keep_everything).predict(analyzer.system.basis.states),
                analyzer.truncate(keep_everything).predict(
                    analyzer.system.basis.states, max_batch_size=17
                ),
            )
        )
