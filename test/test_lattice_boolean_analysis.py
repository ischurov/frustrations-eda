import tempfile
from pathlib import Path
from unittest import TestCase, skip

import lattice_symmetries as ls
import numpy as np
import pandas.testing as pdt

from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LatticeBooleanAnalyzer,
    LBFFromSpinSystemGS,
    SignSignalKind,
    keep_everything,
    keep_largest_n,
)
from spin_lattices import KagomeLattice, SquareLattice
from utils import make_unpacked_configurations


class TestLatticeBooleanAnalysis(TestCase):
    def setUp(self):
        self.analyzer = LatticeBooleanAnalyzer(
            LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=2),
                    J1=1.0,
                    J2=0.0,
                ),
                kind=AmplitudeSignalKind(),
            )
        )
        self.analyzer.fit()

    def test_reconstruction(self):
        prediction = self.analyzer.truncate(keep_everything).predict()
        assert isinstance(self.analyzer.signal, LBFFromSpinSystemGS)
        self.assertTrue(
            np.allclose(
                prediction,
                np.abs(
                    self.analyzer.signal.system.get_df_eigenstate(k=0, canonical_basis=True)
                    .loc[self.analyzer.canonical_basis.states, "eigenstate_coeff"]  # type: ignore
                    .values
                ),
            )
        )

        self.assertTrue(
            np.isclose(
                self.analyzer.truncate(keep_everything).prediction_score(
                    scorer="value_overlap",
                ),
                1,
            )
        )

        self.assertTrue(
            np.isclose(
                self.analyzer.truncate(keep_everything).prediction_score(
                    scorer="neg_mse",
                ),
                0,
            )
        )

    def test_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            system = HeisenbergJ1J2(
                KagomeLattice(width=2, height=2),
                J1=1.0,
                J2=0.0,
            )
            analyzer1 = LatticeBooleanAnalyzer(
                LBFFromSpinSystemGS(
                    system,
                    kind=AmplitudeSignalKind(),
                ),
                cache_dir=Path(cache_dir),
            )
            analyzer1.fit()
            coeff1 = analyzer1.truncate(keep_everything).get_expanded_coeffs_ser()

            analyzer2 = LatticeBooleanAnalyzer(
                LBFFromSpinSystemGS(
                    system,
                    kind=SignSignalKind(),
                ),
                cache_dir=Path(cache_dir),
            )
            analyzer2.fit()

            analyzer3 = LatticeBooleanAnalyzer(
                LBFFromSpinSystemGS(
                    system,
                    kind=AmplitudeSignalKind(),
                ),
                cache_dir=Path(cache_dir),
            )
            analyzer3.fit(from_cache_only=True)

            coeff3 = analyzer3.truncate(keep_everything).get_expanded_coeffs_ser()

            pdt.assert_series_equal(coeff1, coeff3)

            with self.assertRaises(ValueError):
                analyzer3.fit(analyzer3.basis.states, from_cache_only=True)

            with self.assertRaises(ValueError):
                analyzer4 = LatticeBooleanAnalyzer(
                    LBFFromSpinSystemGS(
                        system,
                        kind=AmplitudeMedianBinSignalKind(),
                    ),
                    cache_dir=Path(cache_dir),
                )
                analyzer4.fit(from_cache_only=True)

    def test_marshall(self):
        for hadamard in [True, False]:
            analyzer = LatticeBooleanAnalyzer(
                signal=LBFFromSpinSystemGS(
                    system=HeisenbergJ1J2(
                        lattice=SquareLattice(width=4, height=4),
                        J1=1,
                        J2=0,
                        use_symmetries=True,
                        spin_inversion=1,
                    )
                ),
                hadamard=hadamard,
            )
            analyzer.fit()
            # first_subset_idx = analyzer_sym.learner.get_coeffs_ser().index[0]
            # assert isinstance(first_subset_idx, np.uint64)
            first_subset = make_unpacked_configurations(
                analyzer.learner.get_coeffs_ser().index[0],  # type: ignore
                analyzer.number_spins,
            )

            self.assertTrue(
                all(
                    [
                        first_subset[i] != first_subset[j]
                        for i, j in analyzer.lattice.kind_to_edges[1]
                    ]
                )
            )  # bipartite sublattices

            self.assertTrue(
                np.isclose(
                    analyzer.truncate(keep_largest_n(1)).prediction_score(
                        scorer="sign_overlap",
                    ),
                    1,
                )
            )  # first term is enough to reconstruct the sign structure

            self.assertTrue(
                np.isclose(
                    analyzer.truncate(keep_largest_n(1)).prediction_score(
                        scorer="accuracy",
                    ),
                    1,
                )
            )

    @skip("There's a bug that should be fixed. Or rather switch to fast_lattice_analysis instead")
    def test_hadamard(self):
        signal = LBFFromSpinSystemGS(
            system=HeisenbergJ1J2(
                lattice=SquareLattice(width=4, height=4),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            )
        )
        analyzer = LatticeBooleanAnalyzer(
            signal=signal,
            hadamard=True,
        )
        analyzer.fit()

        reference_analyzer = LatticeBooleanAnalyzer(
            signal=signal,
        )

        reference_analyzer.fit()

        np.random.seed(123)

        x = np.random.choice(reference_analyzer.basis.states, 10)
        # x = np.arange(2**reference_analyzer.number_spins, dtype=np.uint64)
        obtained = analyzer.truncate(keep_largest_n(10)).predict(x)
        expected = reference_analyzer.truncate(keep_largest_n(10)).predict(x)
        print(f"{obtained=}")
        print(f"{expected=}")

        self.assertTrue(np.isclose(obtained, expected).all())

    def test_how_many_terms_to_achieve_score(self):
        analyzer = LatticeBooleanAnalyzer(
            signal=LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=3),
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                kind=SignSignalKind(),
            )
        )
        analyzer.fit()

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

        analyzer = LatticeBooleanAnalyzer(
            signal=LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=3),
                    J1=1,
                    J2=0,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                kind=SignSignalKind(),
            )
        )
        analyzer.fit()

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

        analyzer = LatticeBooleanAnalyzer(
            signal=LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=3),
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                kind=SignSignalKind(),
            )
        )
        analyzer.fit()

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
        analyzer = LatticeBooleanAnalyzer(
            signal=LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=2),
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                ),
                kind=SignSignalKind(),
            )
        )
        analyzer.fit()

        self.assertTrue(
            np.allclose(
                analyzer.truncate(keep_everything).predict(analyzer.basis.states),
                analyzer.truncate(keep_everything).predict(
                    analyzer.basis.states, max_batch_size=17
                ),
            )
        )
