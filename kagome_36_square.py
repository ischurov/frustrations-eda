import itertools
from pathlib import Path

import igraph
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from igraph import Graph
from sympy.combinatorics import Permutation, PermutationGroup

from heisenberg_hamiltonians import HeisenbergJ1J2
from misc_utils import one
from spin_lattices import ChainLattice, FactorLattice, KagomeLattice, SpinLattice

lat = KagomeLattice(3, 4, isotropic=True)
assert lat.number_spins == 36

system = HeisenbergJ1J2(
    lattice=lat,
    J1=1.0,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=Path("groundstates"),
    skip_symmetries_whitelist=True,
)

print(system.get_eigenstates(1))
