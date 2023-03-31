from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heisenberg_hamiltonians import HeisenbergJ1J2
from mvmc_configs import MVMCConfig
from spin_lattices import KagomeLattice, SquareLattice

system = HeisenbergJ1J2(
    KagomeLattice(2, 4), J1=1.0, J2=1.0, ground_state_cache_dir=Path("groundstates")
)
self_name = Path(__file__).name
experiment_dir = Path("experiments") / self_name.removesuffix(".py")
experiment_dir.mkdir(parents=True, exist_ok=True)
trials = 15

if __name__ == "__main__":
    for trial in range(trials):
        (experiment_dir / f"trial_{trial}").mkdir(exist_ok=True)
        mvmc_config = MVMCConfig(
            system,
            total_spin=0,
            in_orbital_type_to_t={1: 1.0, 2: 1.0},
        )
        mvmc_config.do_monte_carlo_optimization()
        out = mvmc_config.extract_out()
        (experiment_dir / "out").mkdir(exist_ok=True)
        out.to_csv(experiment_dir / "out" / f"out_{trial}.csv")
        wavefunction = mvmc_config.extract_wavefunction()
        (experiment_dir / "wavefunctions").mkdir(exist_ok=True)
        wavefunction.to_csv(experiment_dir / "wavefunctions" / f"wavefunction_{trial}.csv")
