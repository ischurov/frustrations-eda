from unittest import TestCase

import numpy as np

from fast_boolean_analysis import (
    FourierSeries,
    fourier_expand,
    keep_everything,
    keep_largest_n,
)
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import (
    AmplitudeSignalKind,
    LBFFromSpinSystemGS,
    SignSignalKind,
)
from spin_lattices import KagomeLattice, SquareLattice
from utils import make_unpacked_configurations


class TestTruncateStrategies(TestCase):
    def test_keep_largest_n(self):

        np.random.seed(123)

        x = np.random.uniform(-1, 1, size=1000)
        assert len(set(x)) == 1000

        obtained = keep_largest_n(10)(x)
        x_abs_sorted = np.sort(np.abs(x))
        expected = np.abs(x) >= x_abs_sorted[-10]

        np.testing.assert_equal(obtained, expected)
        self.assertEqual(np.sum(obtained), 10)
        self.assertTrue(max(x[obtained]) > min(x[~obtained]))


class TestFourierSeries(TestCase):
    def test_reconstruction(self):
        signals = [
            LBFFromSpinSystemGS(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=2),
                    J1=1.0,
                    J2=0.0,
                ),
                kind=kind,
            )
            for kind in [AmplitudeSignalKind(), SignSignalKind()]
        ]
        for signal in signals:
            fourier = fourier_expand(signal)
            prediction = fourier.predict()
            np.testing.assert_allclose(prediction, signal(signal.canonical_basis.states))
            np.testing.assert_almost_equal(
                fourier.prediction_score(
                    scorer={"amplitude": "value_overlap", "sign": "sign_overlap"}[signal.kind.name]
                )[0],
                1.0,
            )

    def test_marshall(self):
        signal = LBFFromSpinSystemGS(
            system=HeisenbergJ1J2(
                lattice=SquareLattice(width=4, height=4),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            )
        )
        fourier = fourier_expand(signal)
        first_subset = make_unpacked_configurations(
            np.argmax(np.abs(fourier.coeffs)),
            signal.number_spins,
        )

        self.assertTrue(
            all([first_subset[i] != first_subset[j] for i, j in signal.lattice.kind_to_edges[1]])
        )  # bipartite sublattices

        self.assertTrue(
            np.isclose(
                fourier.truncate(keep_largest_n(1)).prediction_score(
                    scorer="sign_overlap",
                )[0],
                1,
            )
        )  # first term is enough to reconstruct the sign structure

        self.assertTrue(
            np.isclose(
                fourier.truncate(keep_largest_n(1)).prediction_score(
                    scorer="accuracy",
                )[0],
                1,
            )
        )

    def test_how_many_terms_to_achieve_score(self):
        signal = LBFFromSpinSystemGS(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        terms = fourier.how_many_terms_to_achieve_score(
            target_score=0.95, scorer="sign_overlap", min_terms=1, max_terms=101
        )[0]

        assert terms is not None

        self.assertTrue(
            fourier.truncate_orbitwise(keep_largest_n(terms)).prediction_score("sign_overlap")[0]
            >= 0.95
        )
        self.assertTrue(
            fourier.truncate_orbitwise(keep_largest_n(terms - 1)).prediction_score("sign_overlap")[
                0
            ]
            < 0.95
        )

    def test_how_many_terms_to_achieve_score2(self):
        # Marshall

        signal = LBFFromSpinSystemGS(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        self.assertEqual(
            fourier.how_many_terms_to_achieve_score(target_score=0.99, scorer="accuracy")[0], 1
        )
        self.assertEqual(
            fourier.how_many_terms_to_achieve_score(target_score=0.99, scorer="sign_overlap")[0],
            1,
        )
        self.assertTrue(
            fourier.how_many_terms_to_achieve_score(target_score=1.01, scorer="accuracy")[0]
            is None
        )

        # Frustrated

        signal = LBFFromSpinSystemGS(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        def assert_is_large(terms: int | None):
            if terms is not None:
                self.assertTrue(terms > 1)

        assert_is_large(
            fourier.how_many_terms_to_achieve_score(target_score=0.99, scorer="accuracy")[0]
        )

        assert_is_large(
            fourier.how_many_terms_to_achieve_score(target_score=0.99, scorer="sign_overlap")[0]
        )
