import numpy as np
import numpy.typing as npt
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from misc_utils import one, make_packed_configurations, make_unpacked_configurations

import torch
import math


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
