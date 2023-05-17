import json
import re
import sys
from dataclasses import astuple, dataclass
from itertools import product
from pathlib import Path
from typing import NamedTuple

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import parse
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from fast_boolean_analysis import (
    FourierSeries,
    ScorerType,
    get_scorer,
    keep_everything,
    keep_largest_n,
)
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LBFFromSpinSystem,
    SignalKind,
    SignSignalKind,
)
from parity import popcount
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangleLattice
from utils import read_jsonl_to_df

ground_state_cache_dir = Path("groundstates")

source_lattice = KagomeLattice(2, 4, enumerate_along="x")
destination_lattice = KagomeLattice(2, 5, enumerate_along="x")
J2 = 0.8
source_system = HeisenbergJ1J2(
    lattice=source_lattice,
    J1=1,
    J2=J2,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=ground_state_cache_dir,
    skip_symmetries_whitelist=True,
)
source_system.get_eigenstates(1)

destination_system = HeisenbergJ1J2(
    lattice=destination_lattice,
    J1=1,
    J2=J2,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=ground_state_cache_dir,
    skip_symmetries_whitelist=True,
)
destination_system.get_eigenstates(1)

source_signal = LBFFromSpinSystem(source_system, kind=AmplitudeSignalKind())
destination_signal = LBFFromSpinSystem(destination_system, kind=AmplitudeSignalKind())

source_fourier = FourierSeries.from_signal(source_signal)

source_fourier_repr = source_lattice.get_fourier_basis_data()
destination_fourier_repr = destination_lattice.get_fourier_basis_data()

new_coeffs = np.zeros(2**destination_lattice.number_spins, dtype=np.float64)
new_coeffs[source_fourier_repr.reprs] = source_fourier.coeffs[source_fourier_repr.reprs]
new_representatives_coeffs = new_coeffs[destination_fourier_repr.reprs]

transferred_series = FourierSeries.from_representatives_coeffs(
    signal=destination_signal,
    coeffs=new_representatives_coeffs,
)

score, _ = transferred_series.prediction_score(scorer="value_overlap")
logger.info(score)
print(score)
