import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from sympy.combinatorics import Permutation, PermutationGroup

from spin_systems import SpinSystem
from misc_utils import make_packed_configurations, make_unpacked_configurations, one
from spin_lattices import FactorLattice, KagomeLattice, SpinLattice


def sample_spin_configs(
    number_spins: int, size: int, hamming_weight: int | None = None
):
    from combinadics import Combination

    def unique_sample(N, k):
        result = set()
        while len(result) < k:
            sample = np.random.choice(N, size=k, replace=True)
            result.update(sample)
        return list(result)[:k]

    if hamming_weight is None:
        hamming_weight = number_spins // 2
    n_combinations = int(math.comb(number_spins, hamming_weight))
    if n_combinations < 10_000_000:
        idxs = np.random.choice(n_combinations, size, replace=False)
    elif size / n_combinations < 0.1:
        idxs = unique_sample(n_combinations, size)
    else:
        raise ValueError(
            "number_spins is too large and size is too close to the number of combinations"
        )

    combinations = [
        Combination(number_spins, hamming_weight).Element(idx).data for idx in idxs
    ]
    bitstrings = np.array([np.eye(number_spins)[c].sum(axis=0) for c in combinations])
    return bitstrings


def get_hamming_weight(system: SpinSystem):
    if not system.basis.has_fixed_hamming_weight:
        raise ValueError("system does not have fixed hamming weight")
    return int(
        make_unpacked_configurations(
            system.basis.states[0], number_spins=system.number_spins
        ).sum()
    )


def get_X_Y(system: SpinSystem, sample: int | None = None):
    if sample is None:
        states = system.basis.states
        states_unpacked = make_unpacked_configurations(
            states, system.number_spins
        ).astype(np.float32)
    else:
        states_unpacked = sample_spin_configs(
            number_spins=system.number_spins,
            size=sample,
            hamming_weight=get_hamming_weight(system),
        )
        states = make_packed_configurations(states_unpacked, system.number_spins)

    X = torch.from_numpy(states_unpacked).float()
    Y = torch.log(
        torch.from_numpy(np.real_if_close(system.get_ground_state_coeffs(states)))
        .float()
        .abs()
    )
    return X, Y


def is_inside_starry_region(site, center, starry_region):
    # get polar coordinates of site - center:
    r, phi = np.linalg.norm(site - center), np.arctan2(
        site[1] - center[1], site[0] - center[0]
    )
    return r <= starry_region(phi)


def is_inside_starry_region_row(row, center, starry_region):
    return is_inside_starry_region(
        row[["emb_x", "emb_y"]].to_numpy(), center, starry_region
    )


def circle_starry_region(radius):
    return lambda phi: radius


def hexagon_with_ear_starry_region(radius):
    def wrapper(phi):
        phi = phi % (np.pi / 3) - np.pi / 6
        return radius * (1 + 0.3 * (phi > 0)) / np.cos(np.abs(phi))

    return wrapper


def hexagon_with_notch_starry_region(radius):
    def wrapper(phi):
        phi_ = phi % (np.pi / 3) - np.pi / 6
        return (
            radius
            * (1 - 0.3 * (np.abs((phi + np.pi / 12) % (2 * np.pi / 3)) < 0.5))
            / np.cos(np.abs(phi_))
        )

    return wrapper


def get_fundamental_domain_elements(
    lat: SpinLattice, center: npt.NDArray, region: Callable, scale: float = 1.7
):
    u, v = lat.lattice_basis.T
    fundamental_domain = (
        lat.sites_df.query("is_canonical")
        .apply(
            is_inside_starry_region_row,
            axis=1,
            #    args=(hexagonal_center, hexagon_with_notch_starry_region(np.linalg.norm(u + v) * 1.7)),
            args=(center, region(np.linalg.norm(u + v) * scale)),
        )
        .to_numpy()
    )
    fundamental_domain_elements = list(np.argwhere(fundamental_domain).flatten())
    return fundamental_domain_elements


# plt.figure(figsize=(24, 24))
# lat.plot(spins=fundamental_domain, ax=plt.gca(), show_numbers=False)


# plt.plot(*hexagonal_center, 'o')
def get_factor_lattice(
    initial_lattice: SpinLattice,
    fundamental_domain_elements: list[int],
    shifts: list[Permutation],
):
    group_elements = [shift**i for shift in shifts for i in [1, -1]]
    factor_lattice = FactorLattice(
        initial_lattice=initial_lattice,
        group_elements=group_elements,
        fundamental_domain=fundamental_domain_elements,
    )
    return factor_lattice


def apply(g: Permutation, elements: list[int]):
    return [g(x) for x in elements]


def filter_permutations(group_elements, preimage, image):
    return [g for g in group_elements if apply(g, preimage) == image]


def overlap(x, y):
    return torch.sum(x * y) / torch.sqrt(torch.sum(x**2) * torch.sum(y**2))


def get_generators_dict(
    lattice: SpinLattice, conditions: dict[str, tuple[list[int], list[int]]]
):
    automorphisms = set(Permutation(g) for g in lattice.get_automorphisms())
    generators = {
        name: one(filter_permutations(automorphisms, preimage, image))
        for name, (preimage, image) in conditions.items()
    }
    return generators


# generators = dict(
#     tx=one(filter_permutations(automorphisms, [5, 14, 6, 4], [15, 24, 16, 14])),
#     ty=one(filter_permutations(automorphisms, [3, 5, 14, 6], [6, 8, 17, 9])),
#     rotation=one(filter_permutations(automorphisms, [6, 14, 15], [14, 15, 16])),
#     flip=one(filter_permutations(automorphisms, [0, 6, 4], [0, 6, 1])),
# #    nonplanar4=one(filter_permutations(automorphisms, [0, 21, 16, 11], [11, 16, 21, 0]))
# )


def get_kagome27() -> tuple[SpinLattice, dict[str, Permutation]]:
    kagome12x12 = KagomeLattice(12, 12, isotropic=True, enumerate_along=None)
    u, v = kagome12x12.lattice_basis.T
    hexagonal_center = (
        kagome12x12.sites_df[["emb_x", "emb_y"]].mean().to_numpy() - u - v
    )

    tx = Permutation(kagome12x12.get_translation("x"))
    ty = Permutation(kagome12x12.get_translation("y"))

    right = tx**3
    topright = ty**3
    bottomright = tx**3 * ty ** (-3)

    kagome27 = get_factor_lattice(
        initial_lattice=kagome12x12,
        fundamental_domain_elements=get_fundamental_domain_elements(
            kagome12x12, hexagonal_center, hexagon_with_notch_starry_region
        ),
        shifts=[right, topright, bottomright],
    )

    generators_kagome27 = get_generators_dict(
        lattice=kagome27,
        conditions=dict(
            tx=([5, 14, 6, 4], [15, 24, 16, 14]),
            ty=([3, 5, 14, 6], [6, 8, 17, 9]),
            rotation=([6, 14, 15], [14, 15, 16]),
            flip=([0, 6, 4], [0, 6, 1]),
            nonplanar4=([0, 21, 16, 11], [11, 16, 21, 0]),
        ),
    )
    return kagome27, generators_kagome27


def get_kagome36() -> tuple[SpinLattice, dict[str, Permutation]]:
    kagome12x12 = KagomeLattice(12, 12, isotropic=True, enumerate_along=None)
    u, v = kagome12x12.lattice_basis.T
    hexagonal_center = (
        kagome12x12.sites_df[["emb_x", "emb_y"]].mean().to_numpy() - u - v
    )

    tx = Permutation(kagome12x12.get_translation("x"))
    ty = Permutation(kagome12x12.get_translation("y"))

    righttop = tx**2 * ty**2
    rightbottom = tx**4 * ty ** (-2)
    bottom = tx ** (2) * ty ** (-4)

    kagome36 = get_factor_lattice(
        initial_lattice=kagome12x12,
        fundamental_domain_elements=get_fundamental_domain_elements(
            kagome12x12, hexagonal_center, hexagon_with_ear_starry_region
        ),
        shifts=[righttop, rightbottom, bottom],
    )

    generators_kagome36 = get_generators_dict(
        lattice=kagome36,
        conditions=dict(
            tx=([8, 9, 19, 10], [19, 20, 31, 21]),
            ty=([9, 19, 10, 7], [12, 22, 13, 10]),
            rotation=([10, 19, 20], [19, 20, 21]),
            flip=([0, 10, 21, 9], [0, 10, 21, 11]),
        ),
    )
    return kagome36, generators_kagome36
