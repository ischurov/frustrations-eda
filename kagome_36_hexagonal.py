from spin_systems import spin_system, heisenberg, zero_sector_basis
from kagome_round import get_kagome36
import numpy as np

kagome36, _ = get_kagome36()
system = spin_system(heisenberg(kagome36), zero_sector_basis())
system.get_eigenstates(1)

system_spin_inv = spin_system(heisenberg(kagome36), zero_sector_basis(spin_inversion=1))
system_spin_inv.get_eigenstates(1)

assert np.isclose(system_spin_inv.ground_energy, system.ground_energy)

print(f"{system.get_eigenstates(2)=}")
print(f"{system_spin_inv.get_eigenstates(2)=}")
