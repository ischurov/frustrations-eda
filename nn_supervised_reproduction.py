from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np
import numpy.typing as npt
import torch
from jsonlines import jsonlines
from loguru import logger
from scipy.special import comb
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from conv2d_circular import InvariantSpinCNNRegression
from fourier_supervised_cleanroom import fit_fourier_series, mk_train_test, sign_signal
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from heisenberg_hamiltonians import (
    HeisenbergJ1J2,
    SpinSystem,
    heisenberg_expr,
    heisenberg_expr_hadamard,
)
from misc_utils import keep_serializable, make_unpacked_configurations
from parity import parity, popcount
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name

default_config = {
    "J2s": np.linspace(0, 1, 11),
    "eps_train": [0.01, 0.001, 0.0001],
    "n_test": 50000,
    "sampling_power_train": 2.0,
    "architecture": "dense",
    "n_hidden": 512,
    "hidden_layers": 1,
    "epochs": 1000,
    "write_each_epoch": 1,
    "lr": 1e-3,
    "batch_size": 64,
    "shuffle": True,
    "runs": 1,
    "dilations": None,
    "use_symmetries": False,  # should be True for CNNs and other invariant models
    "skip_symmetries_whitelist": False,
    "spin_inversion": None,
    "sample_repr_then_apply_random_symmetry": False,
    "sample_with_replacement": False,
    "hadamard_basis": False,
    "n_train_from_full_space": True,
    "one_dimensonal_output": False,
    "last_layer_bias": True,
}

# System without symmetries, any network -> usual sampling
# System with symmetries, network is not invariant ->
#   sample representatives, then act with random symmetry from system symmetries
# System with symmetries, network is invariant w.r.t system's symmetries ->
#   sample representatives
configs = {
    0: {
        "lattice": "kagome2x4",
        "sampling_power_train": 2.0,
    },
    1: {
        "lattice": "kagome2x4",
        "sampling_power_train": 1.0,
    },
    2: {
        "lattice": "kagome2x4",
        "sampling_power_train": 0.5,
    },
    3: {
        "lattice": "kagome2x4",
        "sampling_power_train": 0.01,
    },
    4: {
        "lattice": "kagome2x4",
        "sampling_power_train": 4,
    },
    5: {
        "lattice": "kagome2x4",
        "sampling_power_train": 6,
    },
    6: {
        "lattice": "kagome2x4",
        "sampling_power_train": 8,
    },
    7: {
        "lattice": "kagome2x4",
        "sampling_power_train": 10,
    },
    8: {
        "lattice": "kagome2x4",
        "sampling_power_train": 20,
    },
    9: {
        "lattice": "square5x4",
        "J2s": np.linspace(0, 1, 21),
    },
    10: {
        "lattice": "square5x5",
        "J2s": np.linspace(0, 1, 21),
    },
    11: {
        "lattice": "triangular5x5",
        "J2s": np.linspace(0, 1.4, 29),
    },
    12: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 1,
        "xor_strategy": "uniform",
        "epochs": 300,
        "runs": 10,
    },
    13: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 1,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 2,
        "epochs": 300,
        "runs": 10,
    },
    14: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 1,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 2,
        "epochs": 300,
        "runs": 10,
    },
    15: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 1,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 12,
        "epochs": 300,
        "runs": 10,
    },
    16: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 2,
        "xor_strategy": "uniform",
        "epochs": 300,
        "runs": 10,
    },
    17: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 2,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 2,
        "epochs": 300,
        "runs": 10,
    },
    18: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 2,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 2,
        "epochs": 300,
        "runs": 10,
    },
    19: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 2,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 12,
        "epochs": 300,
        "runs": 10,
    },
    20: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "uniform",
        "epochs": 300,
        "runs": 10,
    },
    21: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 2,
        "epochs": 300,
        "runs": 10,
    },
    22: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 2,
        "epochs": 300,
        "runs": 10,
    },
    23: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 12,
        "epochs": 300,
        "runs": 10,
    },
    24: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "uniform",
        "epochs": 300,
        "runs": 10,
    },
    25: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 2,
        "epochs": 300,
        "runs": 10,
    },
    26: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 2,
        "epochs": 300,
        "runs": 10,
    },
    27: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 12,
        "epochs": 300,
        "runs": 10,
    },
    28: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 1,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 8,
        "epochs": 300,
        "runs": 10,
    },
    29: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 2,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 8,
        "epochs": 300,
        "runs": 10,
    },
    30: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 8,
        "epochs": 300,
        "runs": 10,
    },
    31: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "hamming_weight",
        "xor_hamming_weight": 8,
        "epochs": 300,
        "runs": 10,
    },
    32: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 4,
        "epochs": 300,
        "runs": 10,
    },
    33: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 8,
        "epochs": 300,
        "runs": 10,
    },
    34: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 4,
        "epochs": 300,
        "runs": 10,
    },
    35: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 8,
        "epochs": 300,
        "runs": 10,
    },
    36: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 4,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 16,
        "epochs": 300,
        "runs": 10,
    },
    37: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 8,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 16,
        "epochs": 300,
        "runs": 10,
    },
    38: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 16,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 16,
        "epochs": 300,
        "runs": 10,
    },
    39: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 16,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 8,
        "epochs": 300,
        "runs": 10,
    },
    40: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense+xor",
        "n_xors": 16,
        "xor_strategy": "fourier_weight",
        "xor_sampling_power": 4,
        "epochs": 300,
        "runs": 10,
    },
    41: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
    },
    42: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [1, 2, 3],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
    },
    43: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
    },
    44: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "spin_inversion": 1,
        "n_test": 50000,
        "skip_symmetries_whitelist": True,
        "lr": 1e-3,
    },
    45: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "dense",
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 1,
        "eps_train": [0.01],
        "epochs": 200,
        "n_test": 50000,
        "skip_symmetries_whitelist": True,
        "sample_repr_then_apply_random_symmetry": True,
        "sample_with_replacement": True,
    },
    46: {
        "lattice": "square4x6",
        "J2s": [0.5],
        "architecture": "dense",
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 1,
        "eps_train": [0.01],
        "epochs": 200,
        "n_test": 50000,
        "skip_symmetries_whitelist": True,
        "sample_repr_then_apply_random_symmetry": True,
        "sample_with_replacement": True,
    },
    47: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "dense",
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 1,
        "eps_train": [0.01],
        "epochs": 200,
        "n_test": 50000,
        "skip_symmetries_whitelist": True,
        "sample_repr_then_apply_random_symmetry": False,
    },
    48: {
        "lattice": "triangular4x4",
        "J2s": [1.3],
        "architecture": "dense",
        "eps_train": [0.01],
        "hadamard_basis": True,
        "use_symmetries": False,
        "spin_inversion": None,
    },
    49: {
        "lattice": "triangular4x4",
        "J2s": [1.3],
        "architecture": "dense",
        "eps_train": [0.01],
        "n_test": 10000,
        "hadamard_basis": False,
        "use_symmetries": False,
        "spin_inversion": None,
    },
    50: {
        "lattice": "triangular6x4",
        "J2s": [1.3],
        "architecture": "dense",
        "eps_train": [0.005, 0.01],
        "hadamard_basis": True,
        "use_symmetries": False,
        "spin_inversion": None,
    },
    51: {
        "lattice": "triangular6x4",
        "J2s": [1.3],
        "architecture": "dense",
        "eps_train": [0.01, 0.05],
        "n_test": 10000,
        "hadamard_basis": False,
        "use_symmetries": False,
        "spin_inversion": None,
    },
    52: {
        "lattice": "triangular6x4",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "skip_symmetries_whitelist": True,
    },
    53: {
        "lattice": "triangular6x4",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "skip_symmetries_whitelist": True,
    },
    54: {
        "lattice": "triangular6x4",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "dilations": [1, 2, 3],
        "skip_symmetries_whitelist": True,
    },
    55: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense",
        "epochs": 200,
        "eps_train": [0.01],
        "use_symmetries": True,
        "spin_inversion": None,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
    },
    56: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "architecture": "dense",
        "epochs": 200,
        "eps_train": [0.01],
        "use_symmetries": False,
        "spin_inversion": None,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": False,
    },
    57: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "dense",
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 1,
        "eps_train": [0.01],
        "epochs": 200,
        "n_test": 50000,
        "skip_symmetries_whitelist": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
    },
    58: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
    },
    59: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [1, 2, 3],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
    },
    60: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
    },
    61: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "one_dimensonal_output": True,
    },
    62: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [1, 2, 3],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "one_dimensonal_output": True,
    },
    63: {
        "lattice": "square6x4",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "runs": 10,
        "eps_train": [0.05],
        "epochs": 200,
        "use_symmetries": True,
        "n_test": 5000,
        "one_dimensonal_output": True,
    },
    64: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
    },
    65: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [1, 2, 3],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
    },
    66: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
    },
    67: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [3, 2, 1],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    68: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilations": [1, 2, 3],
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    69: {
        "lattice": "square6x6",
        "J2s": [0.5],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    70: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "dilations": [1, 2, 3],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    71: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 10,
        "eps_train": [0.001, 0.005, 0.01],
        "dilations": [3, 2, 1],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    72: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.001, 0.005, 0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    73: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.005],
        "dilations": [1, 2, 3],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    74: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.01],
        "dilations": [1, 2, 3],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    75: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.001],
        "dilations": [3, 2, 1],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    76: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.005],
        "dilations": [3, 2, 1],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    77: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.01],
        "dilations": [3, 2, 1],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    78: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.001],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    79: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.005],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
    80: {
        "lattice": "triangular6x6",
        "J2s": [1.3],
        "architecture": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 3,
        "use_symmetries": True,
        "spin_inversion": None,
        "runs": 5,
        "eps_train": [0.01],
        "skip_symmetries_whitelist": True,
        "epochs": 500,
        "n_test": 10000,
        "sample_with_replacement": True,
        "sample_repr_then_apply_random_symmetry": True,
        "n_train_from_full_space": False,
        "last_layer_bias": False,
        "batch_size": 8192,
    },
}


def get_config(task_id: int):
    return default_config | configs[task_id]


class SignDenseNet(nn.Module):
    def __init__(
        self,
        system: SpinSystem,
        n_hidden: int = 100,
        hidden_layers=1,
        output_dim=2,
        last_layer_bias=True,
    ):
        super().__init__()
        self.system = system
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(system.number_spins, n_hidden), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, output_dim, bias=last_layer_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(
            torch.from_numpy(
                make_unpacked_configurations(
                    x.detach().cpu().numpy(), self.system.number_spins
                ).astype(np.float32)
            ).to(x.device)
        )


class SignDenseNetXor(nn.Module):
    def __init__(
        self,
        system: SpinSystem,
        n_hidden: int = 100,
        hidden_layers=1,
        xor_masks: npt.NDArray[np.uint64] | None = None,
        output_dim=2,
        last_layer_bias=True,
    ):
        super().__init__()
        self.system = system
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        if xor_masks is None:
            xor_masks = np.array([], dtype=np.uint64)
        self.xor_masks = xor_masks

        layers = [
            nn.Linear(system.number_spins + xor_masks.shape[0], n_hidden),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, output_dim, bias=last_layer_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor | npt.NDArray) -> Tensor:
        unpacked_configurations = make_unpacked_configurations(
            x, self.system.number_spins
        ).astype(np.float32)
        if isinstance(x, Tensor):
            x = x.detach().numpy()

        x = x.astype(np.uint64)
        xor_values = parity(x.reshape(-1, 1) & self.xor_masks).astype(np.float32)
        return self.net(
            torch.from_numpy(np.hstack([unpacked_configurations, xor_values]))
        )


def sample_xors(probs: npt.NDArray, size=None):
    probs = probs / probs.sum()
    return np.random.choice(np.arange(len(probs), dtype=np.uint64), size=size, p=probs)


def sample_xors_fourier_weight(power=2):
    def wrapper(
        system: SpinSystem, signal, size: int | None = None
    ) -> npt.NDArray[np.uint64]:
        signal_fn = signal(system)
        series = fit_fourier_series(
            system.canonical_basis.states, signal_fn, system.number_spins
        )
        return sample_xors(np.abs(series) ** power, size=size)

    return wrapper


def sample_xors_uniform():
    def wrapper(
        system: SpinSystem, signal, size: int | None = None
    ) -> npt.NDArray[np.uint64]:
        return sample_xors(np.ones(2**system.number_spins), size=size)

    return wrapper


def sample_xors_hamming_weight(hamming_weight: int):
    def wrapper(system, signal, size: int | None = None) -> npt.NDArray[np.uint64]:
        return sample_xors(
            popcount(np.arange(2**system.number_spins, dtype=np.uint64))
            == hamming_weight,
            size=size,
        )

    return wrapper


def get_sample_xor_strategy(config: dict[str, Any]):
    if config["xor_strategy"] == "uniform":
        return sample_xors_uniform()
    elif config["xor_strategy"] == "fourier_weight":
        return sample_xors_fourier_weight(power=config["xor_sampling_power"])
    elif config["xor_strategy"] == "hamming_weight":
        return sample_xors_hamming_weight(hamming_weight=config["xor_hamming_weight"])
    else:
        raise ValueError(f"Unknown xor strategy {config['xor_strategy']}")


def get_network(config: dict[str, Any], system: SpinSystem, signal) -> nn.Module:
    output_dim = 1 if config["one_dimensonal_output"] else 2
    if config["architecture"] == "dense":
        return SignDenseNet(
            system,
            n_hidden=config["n_hidden"],
            hidden_layers=config["hidden_layers"],
            output_dim=output_dim,
            last_layer_bias=config["last_layer_bias"],
        )
    elif config["architecture"] == "dense+xor":
        xor_masks = get_sample_xor_strategy(config)(
            system, signal, size=config["n_xors"]
        )
        net = SignDenseNetXor(
            system=system,
            n_hidden=config["n_hidden"],
            hidden_layers=config["hidden_layers"],
            xor_masks=xor_masks,
            output_dim=output_dim,
            last_layer_bias=config["last_layer_bias"],
        )
        return net
    elif config["architecture"] == "invariant_cnn":
        assert config[
            "use_symmetries"
        ], "CNNs require symmetries for correct evaluation"
        return InvariantSpinCNNRegression(
            lattice=get_lattice(config["lattice"]),
            hidden_channels=config["hidden_channels"],
            dilations=config["dilations"],
            kernel_size=config["kernel_size"],
            out_dim=output_dim,
            last_layer_bias=config["last_layer_bias"],
        )
    else:
        raise ValueError(f"Unknown architecture {config['architecture']}")


def train(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def get_predicted_signs(states: npt.NDArray, sign_net: nn.Module, device: torch.device):
    outputs = sign_net(torch.from_numpy(states.astype(np.int64)).to(device))
    return (1 - 2 * torch.argmax(outputs, dim=1)).detach().cpu().numpy()


def sign_overlap(system: SpinSystem):
    def wrapper(states: npt.NDArray, sign_net: nn.Module, device: torch.device):
        true_signs = np.sign(system.get_ground_state_coeffs(states))
        probs = np.abs(system.get_ground_state_coeffs(states)) ** 2
        predicted_signs = get_predicted_signs(states, sign_net, device)

        return np.sum(true_signs * predicted_signs * probs) / np.sum(probs)

    return wrapper


def accuracy(system: SpinSystem):
    def wrapper(states: npt.NDArray, sign_net: nn.Module, device: torch.device):
        true_signs = np.sign(system.get_ground_state_coeffs(states))
        predicted_signs = get_predicted_signs(states, sign_net, device)
        mask = (true_signs != 0) & (predicted_signs != 0)
        return np.mean(true_signs[mask] == predicted_signs[mask])

    return wrapper


def main(task_id: int):
    config = get_config(task_id)
    lattice = get_lattice(config["lattice"])
    J2s = config["J2s"]

    (output_dir / str(task_id)).mkdir(parents=True, exist_ok=True)
    signal_factory = sign_signal

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for run in range(config["runs"]):
        start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

        for J2 in J2s:
            logger.debug(f"Running {task_id=} {J2=}. Creating system...")
            system = HeisenbergJ1J2(
                lattice=lattice,
                J1=1,
                J2=J2,
                use_symmetries=config["use_symmetries"],
                spin_inversion=config["spin_inversion"],
                skip_symmetries_whitelist=config["skip_symmetries_whitelist"],
                hamming_weight=None if config["hadamard_basis"] else "half",
                expr_str=heisenberg_expr_hadamard
                if config["hadamard_basis"]
                else heisenberg_expr,
            )
            system.get_eigenstates(1)

            signal_fn = signal_factory(system)
            sign_overlap_fn = sign_overlap(system)
            accuracy_fn = accuracy(system)

            for eps_train in config["eps_train"]:
                logger.debug(f"{eps_train=}. Making train and test states...")
                if (
                    config["sample_repr_then_apply_random_symmetry"]
                    and config["n_train_from_full_space"]
                ):
                    full_sample_space_size = comb(
                        system.number_spins, system.number_spins // 2
                    )
                else:
                    full_sample_space_size = system.basis.states.shape[0]

                logger.debug(f"{full_sample_space_size=}")
                n_train = int(full_sample_space_size * eps_train)
                logger.debug(f"{n_train=}")
                n_test = config["n_test"]
                train_states, test_states = mk_train_test(
                    system,
                    n_train=n_train,
                    n_test=n_test,
                    sampling_power_train=config["sampling_power_train"],
                    apply_random_symmetries=config[
                        "sample_repr_then_apply_random_symmetry"
                    ],
                    replace=config["sample_with_replacement"],
                )
                np.save(
                    output_dir
                    / str(task_id)
                    / f"test_states_{run}_{J2}_{eps_train}.npy",
                    test_states,
                )
                logger.debug("Creating network, criterion, optimizer...")

                net = get_network(config, system, signal=signal_factory)
                net.to(device)

                if config["one_dimensonal_output"]:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
                logger.debug("Creating TensorDataset and DataLoader...")
                # Create a TensorDataset from your inputs X and Y
                target = torch.from_numpy(signal_fn(train_states) == -1).to(torch.long)
                if config["one_dimensonal_output"]:
                    target = target.view(-1, 1).to(torch.float32)
                dataset = TensorDataset(
                    torch.from_numpy(train_states.astype(np.int64)),
                    target,
                )

                dataloader = DataLoader(
                    dataset, batch_size=config["batch_size"], shuffle=config["shuffle"]
                )

                for epoch in range(config["epochs"]):
                    logger.debug("Training")
                    train_loss = train(net, dataloader, criterion, optimizer, device)
                    if epoch % config["write_each_epoch"] == 0:
                        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                        logger.info(f"Epoch {epoch}, train loss: {train_loss:.8f}")
                        np.save(
                            output_dir
                            / str(task_id)
                            / f"prediction_{run}_{J2}_{eps_train}_{epoch}.npy",
                            net(
                                torch.from_numpy(test_states.astype(np.int64)).to(
                                    device
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                        test_overlap = sign_overlap_fn(test_states, net, device)
                        test_accuracy = accuracy_fn(test_states, net, device)
                        logger.info(f"Overlap: {test_overlap:.4f}")
                        with jsonlines.open(
                            output_dir / str(task_id) / f"results.jsonl", mode="a"
                        ) as writer:
                            writer.write(
                                keep_serializable(config, scalar_only=False)
                                | {
                                    "test_overlap": test_overlap,
                                    "test_accuracy": test_accuracy,
                                    "train_loss": train_loss,
                                    "epoch": epoch,
                                    "J2": J2,
                                    "eps_train": eps_train,
                                    "start_timestamp": start_timestamp,
                                    "current_timestamp": current_timestamp,
                                    "run": run,
                                    "task_id": task_id,
                                    "n_train": n_train,
                                }
                            )


if __name__ == "__main__":
    fire.Fire(main)
