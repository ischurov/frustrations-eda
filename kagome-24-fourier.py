#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lattice_symmetries as ls
import yaml
from heisenberg_hamiltonians import (
    make_unpacked_configurations,
    HeisenbergJ1J2,
)

from spin_lattices import (
    SpinLattice,
    ChainLattice,
    SquareLattice,
    KagomeLattice,
)

from boolean_fourier_learner import BooleanFourierLearner

J2 = 0.8

system = HeisenbergJ1J2(
    KagomeLattice(width=2, height=4), J1=1, J2=J2, use_symmetries=True
)
system.get_eigenstates()

ground_state = system.get_df_eigenstate(k=0, canonical_basis=True)

number_spins = system.number_spins
fourier_basis = ls.SpinBasis(
    system.symmetry_group,
    number_spins=number_spins,
    hamming_weight=None,
    spin_inversion=None,
)
fourier_basis.build()

learner = BooleanFourierLearner(number_spins=number_spins, subsets=fourier_basis.states)
gs = ground_state[lambda x: x["amplitude"] > 5e-6]
x = np.array(gs.index, dtype="uint64")
y = np.sign(gs["eigenstate_coeff"]).astype("float64").values

if __name__ == "__main__":
    learner.fit(
        x,
        y,
        batch_size=100,
        stochastic_iterations=10000,
        pickle_progress_to=f"kagome24/fourier-learner-{J2}-{{i}}.pickle.lz",
        pickle_each=100,
    )
