# %%
import itertools
import operator
from dataclasses import dataclass
from functools import reduce

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt

from spin_lattices import (
    ChainLattice,
    ParallelogramSpinLattice,
    SpinLattice,
    SquareLatticeNoDiag,
)
from spin_systems import SpinSystem, heisenberg, no_symmetries_basis

# %%
lattice = SquareLatticeNoDiag(4, 4, enumerate_along="x")

# %%
# Data(
#     edge_index=[2, 45],
#     x_nodes=[10, 2],
#     x_edges=[45, 1],
#     y_node_rdms=[10, 2, 2],
#     y_edge_rdms=[45, 4, 4],
#     y_energy=-8.740063667297363,
#     grid_extent=[2],
#     pbc=False,
# )

# %% [markdown]
# $$H = -\left(\sum_{i=0}^{44} \mathtt{x\_edges[i, 0]} \sigma^z_{\mathtt{edge\_index[0, i]}} \sigma^z_{\mathtt{edge\_index[1, i]}} + \right.\\
# \sum_{j=0}^{9} \mathtt{x\_nodes[j, 0]}\sigma^x_j +
# \left.\sum_{j=0}^{9} \mathtt{x\_nodes[j, 1]} \sigma^z_j\right)$$


# %%
@dataclass
class Data:
    edge_index: npt.NDArray[
        np.int64
    ]  # shape (2, n_edges), where n_edges = n_nodes * (n_nodes - 1) / 2
    x_nodes: npt.NDArray[np.float64]  # shape (n_nodes, 2)
    x_edges: npt.NDArray[np.float64]  # shape (n_edges, 1)
    pbc: bool  # periodic boundary conditions

    y_node_rdms: npt.NDArray[np.float64] | None = None  # shape (n_nodes, 2, 2)
    y_edge_rdms: npt.NDArray[np.float64] | None = None  # shape (n_edges, 4, 4)
    y_energy: np.float64 | None = None
    grid_extent: tuple[int, int] | None = None
    real_edge_index: npt.NDArray[np.int64] | None = None  # shape (2, n_real_edges)

    def to_hamiltonian_expression(self):
        assert self.x_edges.shape[1] == 1
        return (
            reduce(
                operator.add,
                (
                    coeff * ls.Expr("σᶻ₀ σᶻ₁", sites=[[int(i), int(j)]])
                    for coeff, (i, j) in zip(self.x_edges[:, 0], self.edge_index.T)
                ),
            )
            + reduce(
                operator.add,
                (
                    coeff * ls.Expr("σˣ₀", sites=[[int(i)]])
                    for i, coeff in enumerate(self.x_nodes[:, 0])
                ),
            )
            + reduce(
                operator.add,
                (
                    coeff * ls.Expr("σᶻ₀", sites=[[int(i)]])
                    for i, coeff in enumerate(self.x_nodes[:, 1])
                ),
            )
        )


# %%
def random_sample_coeffs(lattice: SquareLatticeNoDiag) -> Data:
    lattice = SquareLatticeNoDiag(4, 4, enumerate_along="x", boundary_conditions="open")
    all_edges = np.array(
        [
            sorted(edge)
            for edge in (itertools.combinations(range(lattice.number_spins), 2))
        ]
    ).T
    edge_to_idx = {tuple((edge)): idx for idx, edge in enumerate(all_edges.T)}
    real_edges_idxs = np.array(
        [
            edge_to_idx[tuple(sorted(edge))]
            for edge, kind in lattice.edges_to_kind.items()
        ]
    )
    x_nodes = np.random.uniform(-1, 1, size=(lattice.number_spins, 2))
    x_edges = np.zeros((all_edges.shape[1], 1))
    x_edges[real_edges_idxs, 0] = np.random.uniform(-1, 1, size=(len(real_edges_idxs)))
    return Data(
        edge_index=all_edges,
        x_nodes=x_nodes,
        x_edges=x_edges,
        pbc=False,
    )


# %%
lattice = SquareLatticeNoDiag(4, 4, enumerate_along="x")
data = random_sample_coeffs(lattice)

# %%
system = SpinSystem(
    lattice=lattice,
    hamiltonian=ls.Operator(
        expression=data.to_hamiltonian_expression(),
        basis=ls.SpinBasis(number_spins=lattice.number_spins),
    ),
)

# %%
system.get_eigenstates(1)

# %%
# SpinSystem(lattice, ls.Operator(ls.SpinBasis(
#             number_spins=(expr.lattice.number_spins),
#             hamming_weight=(
#                 (expr.lattice.number_spins // 2)
#                 if hamming_weight == "half"
#                 else hamming_weight
#             ),
#             spin_inversion=spin_inversion,
#             symmetries=symmetries_factory(expr),
#         ))

# %%
