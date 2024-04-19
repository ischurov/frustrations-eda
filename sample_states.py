import pickle

import numpy as np
import numpy.typing as npt
import torch
from sympy.combinatorics import Permutation, PermutationGroup

from fourier_supervised_cleanroom import apply_random_permutations, sample_from_system
from spin_systems import HeisenbergJ1J2, SpinSystem
from misc_utils import make_packed_configurations, make_unpacked_configurations
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice


def transfer_ground_state(source_system: SpinSystem, destination_system: SpinSystem):
    """
    Finds the ground state of destination system using precomputed ground state of source_system

    source_system and destination_system are expected to be the same system
    with different set of symmetries, destination_system has smaller set of symmetries
    """
    _, characters, norms = destination_system.basis.state_info(destination_system.basis.states)
    reconstructed_gs_coeffs = (
        source_system.get_ground_state_coeffs(destination_system.basis.states) / norms / characters
    )
    return reconstructed_gs_coeffs


def _put_to_cache(system: SpinSystem, eigenvalues, eigenstates):
    assert eigenvalues.shape[0] == eigenstates.shape[1]
    k = eigenvalues.shape[0]
    system.ground_state_cache_dir.mkdir(exist_ok=True)
    eigenstate_path = system.eigenstate_path(k)
    eigenstate_path.write_bytes(pickle.dumps((eigenvalues, eigenstates)))


if __name__ == "__main__":
    lattice = SquareLattice(4, 4)
    lattice_translations = SquareLattice(4, 4, automorphisms="translations")
    J2 = 0.5
    system_sym = HeisenbergJ1J2(
        lattice,
        J1=1,
        J2=J2,
        use_symmetries=True,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )
    system_sym.get_eigenstates(1)

    system_nosym = HeisenbergJ1J2(
        lattice,
        J1=1,
        J2=J2,
        use_symmetries=False,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )
    system_nosym.get_eigenstates(1)

    system_translations = HeisenbergJ1J2(
        lattice_translations,
        J1=1,
        J2=J2,
        use_symmetries=True,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )
    system_translations.get_eigenstates(1)

    reprs_train = sample_from_system(system_translations, n_samples=1000, replace=True)
    reprs_train_unpacked = make_unpacked_configurations(
        reprs_train,
        number_spins=system_translations.number_spins,
    )

    states_train = make_packed_configurations(
        apply_random_permutations(
            reprs_train_unpacked, system_translations.lattice.get_automorphisms()
        ),
        number_spins=system_translations.number_spins,
    )
    print(states_train)

    assert (system_translations.basis.state_info(states_train)[0] == reprs_train).all()

    assert np.allclose(
        transfer_ground_state(system_sym, system_translations),
        system_translations.get_ground_state_coeffs(
            system_translations.basis.states, apply_symmetries=False
        ),
    )

    lattice = SquareLattice(6, 6)

    system_sym = HeisenbergJ1J2(
        lattice,
        J1=1,
        J2=0.5,
        use_symmetries=True,
        spin_inversion=1,
        skip_symmetries_whitelist=True,
    )
    system_sym.get_eigenstates(1)

    # system_no_spin_inversion = HeisenbergJ1J2(
    #     lattice,
    #     J1=1,
    #     J2=0.5,
    #     use_symmetries=True,
    #     spin_inversion=None,
    #     skip_symmetries_whitelist=True,
    # )
    # system_no_spin_inversion.get_eigenstates(1)
    # assert np.isclose(
    #     (
    #         system_no_spin_inversion.get_ground_state()
    #         @ (system_no_spin_inversion.hamiltonian @ system_no_spin_inversion.get_ground_state())
    #     ).real,
    #     system_no_spin_inversion.eigenvalues[0],
    # )

    # reconstructed_gs_coeffs = transfer_ground_state(system_sym, system_no_spin_inversion)
    # _put_to_cache(
    #     system_no_spin_inversion, system_sym.eigenvalues, reconstructed_gs_coeffs.reshape(-1, 1)
    # )

    lattice_translations = SquareLattice(6, 6, automorphisms="translations")
    system_translations = HeisenbergJ1J2(
        lattice_translations,
        J1=1,
        J2=0.5,
        use_symmetries=True,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )
    # reconstructed_gs_coeffs = transfer_ground_state(system_sym, system_translations)
    # _put_to_cache(
    #     system_translations, system_sym.eigenvalues, reconstructed_gs_coeffs.reshape(-1, 1)
    # )

    system_translations.get_eigenstates(1)
    assert np.isclose(
        (
            system_translations.get_ground_state()
            @ (system_translations.hamiltonian @ system_translations.get_ground_state())
        ).real,
        system_translations.eigenvalues[0],
    )
