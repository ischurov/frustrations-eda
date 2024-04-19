# %%
import itertools
from hashlib import md5
from math import factorial
from pathlib import Path

import lattice_symmetries as ls
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import minimize
from sympy.combinatorics import Permutation

from determinant_2023_04_07 import configuration_to_tensor
from spin_systems import HeisenbergJ1J2
from misc_utils import make_unpacked_configurations
from spin_lattices import KagomeLattice, SquareLattice1Diag, TriangularLattice


# %%
# FROM: GPT-4
def configurations_to_tensors(configurations: torch.Tensor | np.ndarray, up=1, down=0):
    # Ensure the input is a torch.Tensor
    if isinstance(configurations, np.ndarray):
        configurations = torch.from_numpy(configurations)

    # Get the indices of up and down elements
    up_indices = (configurations == up).nonzero(as_tuple=True)
    down_indices = (configurations == down).nonzero(as_tuple=True)

    # Split the indices into batch-wise arrays and create a 3D tensor
    up_indices_split = up_indices[1].view(-1, configurations.shape[1] // 2)
    down_indices_split = down_indices[1].view(-1, configurations.shape[1] // 2)

    result = torch.stack([up_indices_split, down_indices_split], dim=1)
    return result


# END FROM


def torch_hash(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    return md5(x.tobytes()).hexdigest() + str(x.shape) + str(x.dtype)


# %%
class SlaterDeterminant(nn.Module):
    def __init__(self, basis: ls.Basis):
        super().__init__()
        self.basis = basis
        self.n_sites = basis.number_bits
        self.configurations = configurations_to_tensors(
            make_unpacked_configurations(basis.states, number_spins=self.n_sites).astype("int64")
        )
        self.f = nn.Parameter(torch.randn(self.n_sites, self.n_sites, dtype=torch.float64))
        signs = []
        logger.debug("Finding signs...")
        for configuration in self.configurations:
            perm_initial = list(itertools.chain(*zip(configuration[0, :], configuration[1, :])))
            signs.append(Permutation(perm_initial).signature())
        logger.debug("Ready")
        self.signs = torch.tensor(signs)

    def forward(self, x: torch.Tensor):
        configs = self.configurations[x.view(-1)]
        signs = self.signs[x.view(-1)]
        # x[..., 0, :] is the spin up part
        # x[..., 1, :] is the spin down part
        # x[..., 0, i] is the i-th spin up site
        # x[..., 1, i] is the i-th spin down site
        # it is expected that x[..., i] < x[..., j] for i < j
        rows = configs[:, 0, :]
        columns = configs[:, 1, :]

        row_indices = rows.unsqueeze(-1).expand(rows.shape[0], rows.shape[1], columns.shape[1])
        col_indices = columns.unsqueeze(1).expand(rows.shape[0], rows.shape[1], columns.shape[1])

        matrices = self.f[row_indices, col_indices]

        return (torch.linalg.det(matrices) * signs).reshape(
            x.shape
        )  # * factorial(self.n_sites // 2)


# %%
from determinant_2023_04_07 import SlaterDeterminant as SlaterDeterminantReference

# %%
lattice = SquareLattice1Diag(3, 3, boundary_conditions="open")
system = HeisenbergJ1J2(
    lattice,
    J1=1,
    J2=1,
    use_symmetries=False,
    spin_inversion=None,
    ground_state_cache_dir=Path("groundstates"),
)
det = SlaterDeterminant(system.basis)
# det_ref = SlaterDeterminantReference(system.number_spins)
# det_ref.f = nn.Parameter(det.f.detach().clone())

# %%
# ref = det_ref(configurations_to_tensors(make_unpacked_configurations(system.basis.states, number_spins=system.number_spins).astype('int64')))

# %%
# assert (det(torch.arange(len(system.basis.states)).long()) == ref).all().item()

# %%
system.get_eigenstates(1)

# %%
eps_train = 1.0

# train_set = torch.Tensor(np.random.choice(len(system.basis.states), size=int(len(system.basis.states) * eps_train), replace=False)).long()
train_set = torch.arange(len(system.basis.states)).long()
gs_numpy = system.get_eigenstates(1)[1][:, 0]
# print(f"{gs_numpy.sum()=}")
ground_state = torch.from_numpy(gs_numpy)
# print(f"{ground_state[train_set]=}")
# print(f"{ground_state[train_set].shape=}")
# print(f"{ground_state=}")
# print(f"{ground_state.sum()}")

target = (ground_state[train_set] < 0).double()
number_spins = system.number_spins


def compute_det(f):
    if not isinstance(f, torch.Tensor):
        f = torch.from_numpy(f)
    with torch.no_grad():
        det.f.copy_(f.reshape((system.number_spins, system.number_spins)))
    # print(f"{torch_hash(f)=}")
    return det(train_set)


def loss_fn(f: torch.Tensor) -> torch.Tensor:
    det_output = compute_det(f)
    # print(f"{torch_hash(det_output)=}")
    probs = torch.sigmoid(det_output)
    # print(f"{torch_hash(probs)=}")
    # print(f"{torch_hash(target)=}")
    loss = torch.nn.functional.binary_cross_entropy(probs, target, reduction="none")
    loss = torch.dot(loss, ground_state[train_set] ** 2)
    # overlap = torch.dot(ground_state, signs * torch.abs(ground_state)) ** 2
    # assert overlap <= 1
    # return 1 - overlap
    return loss


@torch.no_grad()
def loss_fn_no_grad(f):
    value = loss_fn(f).item()
    print(f"{torch_hash(f)=}, {md5(repr(value).encode('utf8')).hexdigest()=}")
    return value


# np.random.seed(1)
# f = 1 - 2 * np.random.rand(system.number_spins * system.number_spins).reshape(
#     (system.number_spins, system.number_spins)
# )
# print(f[:4, :4])
# print(loss_fn_no_grad(f))
# 1 / 0

# FROM: Tom


def unpack_spin_configurations(xs, number_spins: int) -> torch.Tensor:
    unpacked = np.unpackbits(xs.reshape(-1, 1).view(np.uint8), axis=1, bitorder="little")[
        :, :number_spins
    ]
    return torch.from_numpy(unpacked).to(dtype=torch.float32)


def prepare_fermionic_configurations(representatives, number_spins):
    representatives = unpack_spin_configurations(representatives, number_spins)

    confs = torch.stack([configuration_to_tensor(c) for c in representatives])
    signs = []
    for i in range(confs.shape[0]):
        perm_initial = itertools.chain(*zip(confs[i, 0, :], confs[i, 1, :]))
        signs.append(Permutation(list(perm_initial)).signature())
    rows = confs[..., 0, :]
    columns = confs[..., 1, :]
    row_indices = rows.unsqueeze(-1).expand(*rows.shape, columns.shape[1])
    col_indices = columns.unsqueeze(1).expand(*rows.shape, columns.shape[1])
    return (torch.tensor(signs), row_indices, col_indices)


representatives = system.basis.states
correction, row_indices, col_indices = prepare_fermionic_configurations(
    representatives, system.basis.number_bits
)


def compute_signs(f):
    if not isinstance(f, torch.Tensor):
        f = torch.from_numpy(f)
    f = f.view([number_spins, number_spins])
    # print(f"{torch_hash(f)=}")

    matrices = f[row_indices, col_indices]
    # print(f"{matrices.dtype=}")
    return torch.linalg.det(matrices) * correction


def loss_fn_tom(f: torch.Tensor) -> torch.Tensor:
    # signs = torch.tanh(compute_signs(f))
    # signs = signs / torch.max(torch.abs(signs))
    signs = (1 + torch.tanh(compute_signs(f))) / 2
    target = (1 - torch.sign(ground_state)) // 2
    # loss = torch.nn.functional.binary_cross_entropy(signs, target, reduction="none")
    # loss = torch.dot(loss, ground_state**2)
    # overlap = torch.dot(ground_state, signs * torch.abs(ground_state)) ** 2
    # assert overlap <= 1
    # return 1 - overlap
    det_output = compute_signs(f)
    probs = torch.sigmoid(det_output)
    loss = torch.nn.functional.binary_cross_entropy(probs, target, reduction="none")
    loss = torch.dot(loss, ground_state**2)
    return loss


@torch.no_grad()
def loss_fn_no_grad_tom(f):
    value = loss_fn_tom(f).item()
    print(f"{torch_hash(f)=}, {md5(repr(value).encode('utf8')).hexdigest()=}")

    return value


def grad_fn_tom(f):
    print(f"{f.sum()=}")
    p = nn.Parameter(torch.from_numpy(f))
    p.requires_grad_(True)
    loss = loss_fn_tom(p)
    loss.backward()
    grad = p.grad.numpy()
    print(f"{f[:10]=}, {torch_hash(grad)=}")
    return grad


def grad_fn(f):
    print(f"{f.sum()=}")
    p = nn.Parameter(torch.from_numpy(f).reshape((system.number_spins, system.number_spins)))
    if det.f.grad is not None:
        det.f.grad.data.zero_()
    loss = loss_fn(p)
    loss.backward()
    grad = det.f.grad.numpy().reshape(-1)
    print(f"{torch_hash(f)=}, {torch_hash(grad)=}")
    return grad


# np.random.seed(1)
# f = 1 - 2 * np.random.rand(system.number_spins * system.number_spins)
# #    print(f.reshape([system.number_spins, system.number_spins])[:4, :4])
# print(f"{torch_hash(grad_fn(f))=}")
# print(f"{torch_hash(loss_fn(f))=}")

# f = 1 - 2 * np.random.rand(system.number_spins * system.number_spins)
# #    print(f.reshape([system.number_spins, system.number_spins])[:4, :4])
# print(f"{torch_hash(grad_fn(f))=}")
# print(f"{torch_hash(loss_fn(f))=}")

# f = 1 - 2 * np.random.rand(system.number_spins * system.number_spins)
# #    print(f.reshape([system.number_spins, system.number_spins])[:4, :4])
# print(f"{torch_hash(grad_fn(f))=}")
# print(f"{torch_hash(loss_fn(f))=}")

# 1 / 0


@torch.no_grad()
def callback_fn(x, f=None, context=None):
    signs = torch.sign(compute_det(x))
    overlap = torch.dot(ground_state[train_set], signs * torch.abs(ground_state[train_set]))
    # print(ground_state[train_set].sum())
    # print(f"{torch_hash(x)=}, {torch_hash(signs)=}, {torch_hash(ground_state[train_set])=}, {torch_hash(overlap)=}")
    print(f"{overlap.item()=}")
    return


logger.debug("Starting minimize")
np.random.seed(1)
x0 = 1 - 2 * np.random.rand(system.number_spins**2)
for i in range(10):
    res = minimize(
        fun=loss_fn_no_grad_tom,
        x0=x0,
        method="BFGS",
        jac=grad_fn,
        options={"maxiter": 1000},
        callback=callback_fn,
    )
    print(res)
