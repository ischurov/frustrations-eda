import tempfile
from pathlib import Path
from unittest import TestCase, skip

import lattice_symmetries as ls
import numpy as np
import pandas as pd
import pandas.testing as pdt
import torch
import torch.nn as nn

from spin_systems import HeisenbergJ1J2
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LatticeBooleanAnalyzer,
    LBFFromNN,
    LBFFromSpinSystem,
    SignSignalKind,
    keep_everything,
    keep_largest_n,
)
from misc_utils import make_unpacked_configurations
from spin_lattices import KagomeLattice, SquareLattice


class TestLatticeBooleanAnalysis(TestCase):
    def test_reconstruction(self):
        analyzer = LatticeBooleanAnalyzer(
            LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=4, height=4),
                    J1=1.0,
                    J2=0.5,
                ),
                eigenstate=0,
                kind=AmplitudeSignalKind(),
            )
        )
        analyzer.fit()

        prediction = analyzer.truncate(keep_everything).predict()
        assert isinstance(analyzer.signal, LBFFromSpinSystem)
        np.testing.assert_allclose(
            prediction,
            np.abs(
                analyzer.signal.system.get_df_eigenstate(k=0, canonical_basis=True)
                .loc[analyzer.canonical_basis.states, "eigenstate_coeff"]  # type: ignore
                .values
            ),
        )

        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="value_overlap",
                ),
                1,
            )
        )

        self.assertTrue(
            np.isclose(
                analyzer.truncate(keep_everything).prediction_score(
                    scorer="neg_mse",
                ),
                0,
            )
        )

    def test_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            system = HeisenbergJ1J2(
                SquareLattice(width=4, height=2),
                J1=1.0,
                J2=0.5,
            )
            analyzer1 = LatticeBooleanAnalyzer(
                LBFFromSpinSystem(
                    system,
                    eigenstate=0,
                    kind=AmplitudeSignalKind(),
                ),
                cache_dir=Path(cache_dir),
            )
            analyzer1.fit()
            coeff1 = analyzer1.truncate(keep_everything).get_expanded_coeffs_ser()

            analyzer2 = LatticeBooleanAnalyzer(
                LBFFromSpinSystem(
                    system,
                    eigenstate=0,
                    kind=SignSignalKind(),
                ),
                cache_dir=Path(cache_dir),
            )
            analyzer2.fit()

            analyzer3 = LatticeBooleanAnalyzer(
                LBFFromSpinSystem(
                    system,
                    eigenstate=0,
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
                    LBFFromSpinSystem(
                        system,
                        eigenstate=0,
                        kind=AmplitudeMedianBinSignalKind(),
                    ),
                    cache_dir=Path(cache_dir),
                )
                analyzer4.fit(from_cache_only=True)

    def test_marshall(self):
        analyzer = LatticeBooleanAnalyzer(
            signal=LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    lattice=SquareLattice(width=4, height=4),
                    J1=1,
                    J2=0,
                    use_symmetries=True,
                    spin_inversion=1,
                )
            ),
            hadamard=False,
        )
        analyzer.fit()
        # first_subset_idx = analyzer_sym.learner.get_coeffs_ser().index[0]
        # assert isinstance(first_subset_idx, np.uint64)
        first_subset = make_unpacked_configurations(
            analyzer.learner.get_coeffs_ser().index[0],  # type: ignore
            analyzer.number_spins,
        )

        self.assertTrue(
            all([first_subset[i] != first_subset[j] for i, j in analyzer.lattice.kind_to_edges[1]])
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

    @skip("Doesn't work correctly; better switch to fast_boolean_analysis")
    def test_hadamard(self):
        signal = LBFFromSpinSystem(
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
            signal=LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=2, height=4),
                    J1=1,
                    J2=0.5,
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
            signal=LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=4, height=4),
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
            signal=LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=4, height=4),
                    J1=1,
                    J2=0.5,
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
            signal=LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=2, height=4),
                    J1=1,
                    J2=0.5,
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

    def test_lbf_from_nn_batch(self):
        lattice = SquareLattice(width=4, height=4)
        number_spins = lattice.number_spins
        net = nn.Sequential(
            nn.Linear(number_spins, 2, dtype=torch.float64),
        )
        net[0].bias.data = torch.zeros(2, dtype=torch.float64)
        print("Making signal1")
        signal1 = LBFFromNN(
            lattice=lattice,
            nn=net,
            probs=pd.Series(1, index=np.arange(2**number_spins)),
            batch_size=10,
        )

        print("Making signal2")
        signal2 = LBFFromNN(
            lattice=lattice,
            nn=net,
            probs=pd.Series(1, index=np.arange(2**number_spins)),
            batch_size=2**number_spins,
        )

        x = np.arange(2**number_spins, dtype=np.uint64)
        print("Predicting signal1")
        y1 = signal1(x)
        print("Predicting signal2")
        y2 = signal2(x)
        print("Asserting")
        np.testing.assert_equal(y1, y2)
        self.assertTrue((y1 != 1).any() and (y1 != -1).any())
        self.assertTrue((y2 != 1).any() and (y2 != -1).any())
