from misc_utils import hadamard_transform
import numpy as np
import matplotlib.pyplot as plt
from spin_lattices import KagomeLattice
from spin_systems import heisenberg, spin_system, no_symmetries_basis
import pandas as pd
import seaborn as sns
import numpy.typing as npt
import lattice_symmetries as ls
import loguru
from parity import calculate_fourier_transform_matrix
from misc_utils import kronecker_power, rotation_matrix, eigenstate_in_full_basis
from tqdm.auto import tqdm
from ising_sign_reconstruction import custom_signs_hadamard_spread
import unittest


class TestCustomSignHadamardSpread(unittest.TestCase):
    def test_custom_signs_hadamard_spread_reproduction(self):
        lattice = KagomeLattice(2, 4)
        system = spin_system(heisenberg(lattice, J2=1), no_symmetries_basis())
        states = system.basis.states[:20].copy()
        # np.random.shuffle(states)
        new_states = system.basis.states[10:40]

        sign_fn = custom_signs_hadamard_spread(
            np.sign(system.get_ground_state_coeffs(states, apply_symmetries=False)),
            states,
        )

        self.assertTrue(
            (
                sign_fn(states)
                == np.sign(
                    system.get_ground_state_coeffs(states, apply_symmetries=False)
                )
            ).all()
        )

        self.assertTrue(
            (
                sign_fn(new_states)[:10]
                == np.sign(
                    system.get_ground_state_coeffs(states[10:], apply_symmetries=False)
                )
            ).all()
        )

    def test_custom_signs_hadamard_spread_reproduction_shuffle(self):
        lattice = KagomeLattice(2, 4)
        system = spin_system(heisenberg(lattice, J2=1), no_symmetries_basis())
        states = system.basis.states[:20].copy()
        np.random.shuffle(states)

        sign_fn = custom_signs_hadamard_spread(
            np.sign(system.get_ground_state_coeffs(states, apply_symmetries=False)),
            states,
        )

        self.assertTrue(
            (
                sign_fn(states)
                == np.sign(
                    system.get_ground_state_coeffs(states, apply_symmetries=False)
                )
            ).all()
        )

    def test_custom_signs_hadamard_spread_prediction_quality(self):
        np.random.seed(42)
        lattice = KagomeLattice(2, 4)
        system = spin_system(heisenberg(lattice, J2=1), no_symmetries_basis())

        train_size = 10000
        states = np.random.choice(
            system.basis.states,
            p=system.ground_state**2,
            size=train_size,
            replace=False,
        )
        test_states = np.random.choice(
            system.basis.states, p=system.ground_state**2, size=10000, replace=False
        )
        sign_fn = custom_signs_hadamard_spread(
            np.sign(system.get_ground_state_coeffs(states, apply_symmetries=False)),
            states,
        )
        self.assertTrue(
            (
                sign_fn(test_states)
                == np.sign(
                    system.get_ground_state_coeffs(test_states, apply_symmetries=False)
                )
            ).mean()
            > 0.9
        )
