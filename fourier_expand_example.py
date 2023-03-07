from pathlib import Path

import numpy as np

from fast_boolean_analysis import fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem, SignSignalKind
from spin_lattices import KagomeLattice

lattice = KagomeLattice(2, 4)
system = HeisenbergJ1J2(lattice, J1=1, J2=0.5, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
series = fourier_expand(lbf)
ground_state = system.get_ground_state_in_canonical_basis()
# ground_state is an array of the same size as the number of states
# in the canonical (not symmetrized) basis.
# ground_state[i] is the coefficient of the i-th basis element in the ground state
# of the system.
# You can get the actual state by doing:
# basis_element = lbf.canonical_basis.states[i]

# The canonical basis contains less than 2**number_spins state
# as we are working in the zero magnetization subspace.

assert (series.predict() == np.sign(ground_state)).all()
# for untrunctaed series, prediction of the signal is perfect

prediction = series.truncate(keep_largest_n(3)).predict()
# here is how we can truncate the series to keep only the largest 3 terms
# and then predict the signal
