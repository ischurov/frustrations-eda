import lattice_symmetries as ls
import scipy
import scipy.sparse.linalg
from spin_lattices import ChainLattice
from heisenberg_hamiltonians import HeisenbergJ1J2

def test_weird_segfault_1():
    lattice = ChainLattice(8)
    J2=0
    system = HeisenbergJ1J2(
                lattice=lattice,
                J1=1,
                J2=J2,
                use_symmetries=True,
                spin_inversion=1,
                skip_symmetries_whitelist=True,
                ground_state_cache_dir=None,
    )
    system.get_eigenstates(1)


test_weird_segfault_1()
