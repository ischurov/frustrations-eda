from misc_utils import hadamard_transform
import numpy as np
from spin_lattices import KagomeLattice
from spin_systems import (
    heisenberg_transversal_field,
    spin_system,
    no_symmetries_basis,
)
from spin_lattices import KagomeLattice

# from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
import numpy as np
from misc_utils import hadamard_transform
from hadamard_learning import evaluate_hadamard_learning_test_set

if __name__ == "__main__":
    lattice = KagomeLattice(2, 4)
    outputs = []
    sample_power = 2

    for h in np.linspace(0, 1, 21):
        for J2 in [0.3, 1]:
            system = spin_system(
                heisenberg_transversal_field(lattice, J2=J2, h=h),
                no_symmetries_basis(hamming_weight=None, spin_inversion=None),
            )
            wavefunction = system.ground_state
            transformed_wavefunction = hadamard_transform(wavefunction)
            outputs.append(
                evaluate_hadamard_learning_test_set(
                    wavefunction=wavefunction,
                    basis=system.basis,
                    sample_power=sample_power,
                    test_size=50000,
                    eps_train=1e-2,
                )
                | {
                    "h": h,
                    "lattice": "kagome2x4",
                    "overlap_with_transformed": transformed_wavefunction
                    @ wavefunction
                    / np.linalg.norm(wavefunction)
                    / np.linalg.norm(transformed_wavefunction),
                    "J2": J2,
                }
            )
