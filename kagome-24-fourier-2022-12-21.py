#!/usr/bin/env python
# coding: utf-8


from itertools import chain
from pathlib import Path

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from boolean_fourier_learner import BooleanFourierLearner
from heisenberg_hamiltonians import HeisenbergJ1J2, make_unpacked_configurations
from spin_lattices import ChainLattice, KagomeLattice, SpinLattice, SquareLattice


def main():
    train_size = 20000

    experiment_dir = Path("kagome24-2022-12-21")
    experiment_dir.mkdir(exist_ok=True)

    for J2 in chain(
        *zip(
            [0.502, 0.504, 0.506, 0.508, 0.51, 0.52, 0.53],
            reversed([0.512, 0.514, 0.516, 0.518, 0.54, 0.55, 0.56]),
        )
    ):
        system = HeisenbergJ1J2(KagomeLattice(width=2, height=4), J1=1, J2=J2, use_symmetries=True)
        system.get_eigenstates()
        number_spins = system.number_spins

        fourier_basis = ls.SpinBasis(
            system.symmetries,
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        fourier_basis.build()

        ground_state = system.get_df_eigenstate(k=0, canonical_basis=True).assign(
            prob=lambda x: x["amplitude"] ** 2
        )

        learner = BooleanFourierLearner(number_spins=number_spins, subsets=fourier_basis.states)
        gs = ground_state[lambda x: x["amplitude"] > 5e-6]

        train = gs.sample(n=train_size, weights="prob")
        train.reset_index().to_feather(experiment_dir / f"train-{J2!r}.feather")

        x = np.array(train.index, dtype="uint64")
        y = np.sign(train["eigenstate_coeff"]).astype("float64").values

        learner.fit(
            x,
            y,
            batch_size=100,
            pickle_progress_to=str(experiment_dir / f"fourier-learner-{J2!r}-{{i}}.pickle.lz"),
            pickle_each=50,
            show_progress=True,
        )


if __name__ == "__main__":
    main()
