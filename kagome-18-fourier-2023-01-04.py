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
from pathlib import Path
from itertools import chain


def main():
    train_size = 200000

    experiment_dir = Path("kagome18-2023-01-04")
    experiment_dir.mkdir(exist_ok=True)

    for J2 in [
        0.0,
        0.2,
        0.4,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]:
        system = HeisenbergJ1J2(
            KagomeLattice(width=2, height=3), J1=1, J2=J2, use_symmetries=True
        )
        system.get_eigenstates()
        number_spins = system.number_spins

        fourier_basis = ls.SpinBasis(
            system.symmetry_group,
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        fourier_basis.build()

        ground_state = system.get_df_eigenstate(k=0).assign(
            prob=lambda x: x["amplitude"] ** 2
        )

        learner = BooleanFourierLearner(
            number_spins=number_spins, subsets=fourier_basis.states
        )
        gs = ground_state[lambda x: x["amplitude"] > 5e-6]  # type: ignore
        # See (https://github.com/pandas-dev/pandas-stubs/issues/256)

        train = gs.sample(n=train_size, weights="prob")
        train.reset_index().to_feather(experiment_dir / f"train-{J2!r}.feather")

        x = np.array(train.index, dtype="uint64")
        y = np.sign(train["eigenstate_coeff"]).astype("float64").values

        learner.fit(
            x,
            y,
            weights=1.0 / train["prob"].values,
            batch_size=100,
            pickle_progress_to=str(
                experiment_dir / f"fourier-learner-{J2!r}-{{i}}.pickle.lz"
            ),
            pickle_each=50,
            show_progress=True,
        )


if __name__ == "__main__":
    main()
