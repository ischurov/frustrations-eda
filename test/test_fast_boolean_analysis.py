import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from itertools import product
from unittest import TestCase

import numpy as np

from fast_boolean_analysis import (
    FourierSeries,
    fourier_expand,
    keep_everything,
    keep_largest_n,
)
from spin_systems import HeisenbergJ1J2
from lattice_boolean_analysis import (
    AmplitudeSignalKind,
    LBFFromSpinSystem,
    SignSignalKind,
)
from spin_lattices import KagomeLattice, SquareLattice
from misc_utils import make_unpacked_configurations


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
            LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    SquareLattice(width=4, height=4),
                    J1=1.0,
                    J2=0.5,
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
        signal = LBFFromSpinSystem(
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
        signal = LBFFromSpinSystem(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=4),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        success, terms, _ = fourier.how_many_terms_to_achieve_score(
            target_score=0.95, scorer="sign_overlap", min_terms=1, max_terms=1001, orbitwise=True
        )

        assert success

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

    def test_how_many_terms_to_achieve_relative_weight(self):
        signal = LBFFromSpinSystem(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=4),
                J1=1,
                J2=0.8,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        for target in [0.1, 0.5, 0.8]:
            terms = fourier.how_many_terms_to_achieve_relative_weight(target)
            weights = fourier.coeffs**2
            weights /= np.sum(weights)
            self.assertTrue(np.sum(weights[np.argsort(weights)[-terms:]]) >= target)
            self.assertTrue(np.sum(weights[np.argsort(weights)[-terms + 1 :]]) < target)

    def test_how_many_terms_to_achieve_score_not_orbitwise(self):
        np.random.seed(123)
        for J2, (scorer, target_score), x_type in product(
            [0.7, 0.8], [("sign_overlap", 0.95), ("f1", 0.8)], ["full", "random"]
        ):
            signal = LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    KagomeLattice(width=2, height=3),
                    J1=1,
                    J2=J2,
                    use_symmetries=False,
                    spin_inversion=None,
                ),
                kind=SignSignalKind(),
            )

            if x_type == "random":
                x = np.random.choice(signal.canonical_basis.states, size=100)
            else:
                x = None

            fourier = fourier_expand(signal)

            success, terms, prediction = fourier.how_many_terms_to_achieve_score(
                target_score=target_score, x=x, scorer=scorer, min_terms=1, max_terms=None
            )

            assert success
            print(
                f"J2={J2}, scorer={scorer}, terms={terms}. Checking that the score is at least {target_score}"
            )
            self.assertTrue(
                fourier.truncate(keep_largest_n(terms)).prediction_score(scorer, x=x)[0]
                >= target_score
            )
            print("Checking that the score is lower when we truncate by one term")
            self.assertTrue(
                fourier.truncate(keep_largest_n(terms - 1)).prediction_score(scorer, x=x)[0]
                < target_score
            )

            np.testing.assert_allclose(
                fourier.truncate(keep_largest_n(terms)).predict(x=x), prediction
            )

    def test_how_many_terms_to_achieve_score2(self):
        # Marshall

        signal = LBFFromSpinSystem(
            system=HeisenbergJ1J2(
                SquareLattice(width=4, height=4),
                J1=1,
                J2=0,
                use_symmetries=True,
                spin_inversion=1,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        self.assertEqual(
            fourier.how_many_terms_to_achieve_score(
                target_score=0.99, scorer="accuracy", orbitwise=True
            )[1],
            1,
        )
        self.assertEqual(
            fourier.how_many_terms_to_achieve_score(
                target_score=0.99, scorer="sign_overlap", orbitwise=True
            )[1],
            1,
        )
        self.assertFalse(
            fourier.how_many_terms_to_achieve_score(
                target_score=1.01, scorer="accuracy", orbitwise=True
            )[0]
        )

        # Frustrated

        signal = LBFFromSpinSystem(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=False,
                spin_inversion=None,
            ),
            kind=SignSignalKind(),
        )

        fourier = fourier_expand(signal)

        success, terms, _ = fourier.how_many_terms_to_achieve_score(
            target_score=0.99, scorer="accuracy", orbitwise=True
        )

        self.assertTrue(not success or terms > 1)

        success, terms, _ = fourier.how_many_terms_to_achieve_score(
            target_score=0.99, scorer="sign_overlap", orbitwise=True
        )

        self.assertTrue(not success or terms > 1)

    def test_predict(self):
        signal = LBFFromSpinSystem(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=3),
                J1=1,
                J2=0.8,
                use_symmetries=False,
                spin_inversion=None,
            ),
            kind=SignSignalKind(),
        )
        fourier = fourier_expand(signal).truncate(keep_largest_n(10))
        prediction_canonical_basis = fourier.predict()
        np.random.seed(123)
        x = np.random.choice(signal.canonical_basis.states, size=10)
        prediction = fourier.predict(x)
        np.testing.assert_allclose(
            prediction, prediction_canonical_basis[signal.canonical_basis.index(x)]
        )

    def test_from_representative_coeffs(self):
        for lattice in [KagomeLattice(2, 2), KagomeLattice(2, 3)]:
            signal = LBFFromSpinSystem(
                system=HeisenbergJ1J2(
                    lattice,
                    J1=1,
                    J2=0.8,
                    use_symmetries=True,
                    spin_inversion=1,
                    skip_symmetries_whitelist=True,
                ),
                kind=SignSignalKind(),
            )
            fourier = FourierSeries.from_signal(signal)
            basis_data = lattice.get_fourier_basis_data()
            fourier2 = FourierSeries.from_representatives_coeffs(
                signal, fourier.coeffs[basis_data.reprs]
            )
            np.testing.assert_allclose(fourier.coeffs, fourier2.coeffs)
