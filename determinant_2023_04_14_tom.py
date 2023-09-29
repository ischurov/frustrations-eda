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
from heisenberg_hamiltonians import HeisenbergJ1J2
from misc_utils import make_unpacked_configurations
from spin_lattices import KagomeLattice, SquareLattice1Diag, TriangularLattice

lattice = SquareLattice1Diag(3, 3, boundary_conditions="open")
system = HeisenbergJ1J2(
    lattice,
    J1=1,
    J2=1,
    use_symmetries=False,
    spin_inversion=None,
    ground_state_cache_dir=Path("groundstates"),
)


def torch_hash(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    return md5(x.tobytes()).hexdigest() + str(x.shape) + str(x.dtype)


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


if __name__ == "__main__":
    (_, ground_state) = system.get_eigenstates(1)
    # print(f"{ground_state.sum()} (numpy)")
    # with open("ground_state.pkl", "rb") as f:
    #     ground_state = pickle.load(f)
    #     # pickle.dump(ground_state, f)
    ground_state = torch.from_numpy(ground_state).squeeze(dim=1)
    # print(f"{ground_state=}")
    # print(f"{ground_state.shape=}")
    # print(f"{ground_state=}")
    # print(f"{ground_state.sum()}")
    representatives = system.basis.states
    correction, row_indices, col_indices = prepare_fermionic_configurations(
        representatives, system.basis.number_bits
    )

    number_spins = system.number_spins

    def compute_signs(f):
        if not isinstance(f, torch.Tensor):
            f = torch.from_numpy(f)
        f = f.view([number_spins, number_spins])
        # print(f"{torch_hash(f)=}")

        matrices = f[row_indices, col_indices]
        # print(f"{matrices.dtype=}")
        return torch.linalg.det(matrices) * correction

    def loss_fn(f: torch.Tensor) -> torch.Tensor:
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
    def loss_fn_no_grad(f):
        value = loss_fn(f).item()
        print(f"{torch_hash(f)=}, {md5(repr(value).encode('utf8')).hexdigest()=}")

        return value

    def grad_fn(f):
        print(f"grad_fn: {f[:10]=}")
        p = nn.Parameter(torch.from_numpy(f))
        p.requires_grad_(True)
        loss = loss_fn(p)
        loss.backward()
        grad = p.grad.numpy()
        # print(f"{torch_hash(f)=}, {torch_hash(grad)=}")
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

    @torch.no_grad()
    def callback_fn(x, f=None, context=None):
        signs = torch.sign(compute_signs(x))
        overlap = torch.dot(ground_state, signs * torch.abs(ground_state))
        # print(ground_state.sum())
        # print(
        #     f"{torch_hash(x)=}, {torch_hash(signs)=}, {torch_hash(ground_state)=}, {torch_hash(overlap)=}"
        # )
        print(f"{overlap.item()=}")
        return
        # assert overlap <= 1
        # return 1 - overlap
        if True:  # overlap.item() > 0.1:
            res = minimize(
                fun=loss_fn_no_grad,
                x0=x,
                method="BFGS",
                jac=grad_fn,
                options={"maxiter": 100},
            )
            signs_new = torch.sign(compute_signs(res.x))
            overlap_new = torch.dot(ground_state, signs_new * torch.abs(ground_state)) ** 2
        else:
            res = None
            overlap_new = None

        print(f, overlap.item(), overlap_new, context)

    logger.debug("Starting minimize")
    np.random.seed(1)
    x0 = 1 - 2 * np.random.rand(system.number_spins * system.number_spins)
    for i in range(10):
        res = minimize(
            fun=loss_fn_no_grad,
            x0=x0,
            method="BFGS",
            jac=grad_fn,
            options={"maxiter": 1000},
            callback=callback_fn,
        )
        print(res)
    exit(0)
