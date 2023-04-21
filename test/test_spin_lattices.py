import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=False)

from unittest import TestCase, skip

import numpy as np

from spin_lattices import KagomeLattice, SquareLattice


# TODO
class TestSpinLattices(TestCase):
    @skip("TODO")
    def test_indicies_tensor(self):
        lattice = SquareLattice(width=4, height=3)
        indicies = lattice.indicies_tensor()
        self.assertEqual(indicies.shape, (3, 4))
        for i in range(3 - 1):
            for j in range(4 - 1):
                pass

    def test_translations(self):
        lattice = KagomeLattice(width=2, height=4)

        # order of x translation is 2
        self.assertTrue(
            np.allclose(
                np.array(lattice.x_translation)[lattice.x_translation],
                np.arange(lattice.number_spins),
            )
        )

        # order of y translation is 4
        y_translation = np.array(lattice.y_translation)  # degree 1
        y_translation = y_translation[lattice.y_translation]  # degree 2
        y_translation = y_translation[lattice.y_translation]  # degree 3
        y_translation = y_translation[lattice.y_translation]  # degree 4
        self.assertTrue(
            np.allclose(
                y_translation,
                np.arange(lattice.number_spins),
            )
        )

        # commutativity
        self.assertTrue(
            np.allclose(
                np.array(lattice.x_translation)[lattice.y_translation],
                np.array(lattice.y_translation)[lattice.x_translation],
            )
        )

        # translations are automorphisms
        self.assertTrue(lattice.x_translation in lattice.get_automorphisms())
        self.assertTrue(lattice.y_translation in lattice.get_automorphisms())
