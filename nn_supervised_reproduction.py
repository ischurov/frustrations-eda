from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np
import numpy.typing as npt
import torch
from jsonlines import jsonlines
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from conv2d_circular import InvariantSpinCNNRegression
from fourier_supervised_cleanroom import fit_fourier_series, mk_train_test, sign_signal
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
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
}

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
}


def get_config(task_id: int):
    return default_config | configs[task_id]


class SignDenseNet(nn.Module):
    def __init__(self, system: SpinSystem, n_hidden: int = 100, hidden_layers=1):
        super().__init__()
        self.system = system
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(system.number_spins, n_hidden), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(
            torch.from_numpy(
                make_unpacked_configurations(x, self.system.number_spins).astype(np.float32)
            )
        )


class SignDenseNetXor(nn.Module):
    def __init__(
        self,
        system: SpinSystem,
        n_hidden: int = 100,
        hidden_layers=1,
        xor_masks: npt.NDArray[np.uint64] | None = None,
    ):
        super().__init__()
        self.system = system
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        if xor_masks is None:
            xor_masks = np.array([], dtype=np.uint64)
        self.xor_masks = xor_masks

        layers = [nn.Linear(system.number_spins + xor_masks.shape[0], n_hidden), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor | npt.NDArray) -> Tensor:
        unpacked_configurations = make_unpacked_configurations(x, self.system.number_spins).astype(
            np.float32
        )
        if isinstance(x, Tensor):
            x = x.detach().numpy()

        x = x.astype(np.uint64)
        xor_values = parity(x.reshape(-1, 1) & self.xor_masks).astype(np.float32)
        return self.net(torch.from_numpy(np.hstack([unpacked_configurations, xor_values])))


def sample_xors(probs: npt.NDArray, size=None):
    probs = probs / probs.sum()
    return np.random.choice(np.arange(len(probs), dtype=np.uint64), size=size, p=probs)


def sample_xors_fourier_weight(power=2):
    def wrapper(system: SpinSystem, signal, size: int | None = None) -> npt.NDArray[np.uint64]:
        signal_fn = signal(system)
        series = fit_fourier_series(system.canonical_basis.states, signal_fn, system.number_spins)
        return sample_xors(np.abs(series) ** power, size=size)

    return wrapper


def sample_xors_uniform():
    def wrapper(system: SpinSystem, signal, size: int | None = None) -> npt.NDArray[np.uint64]:
        return sample_xors(np.ones(2**system.number_spins), size=size)

    return wrapper


def sample_xors_hamming_weight(hamming_weight: int):
    def wrapper(system, signal, size: int | None = None) -> npt.NDArray[np.uint64]:
        return sample_xors(
            popcount(np.arange(2**system.number_spins, dtype=np.uint64)) == hamming_weight,
            size=size,
        )

    return wrapper


def get_sample_strategy(config: dict[str, Any]):
    if config["xor_strategy"] == "uniform":
        return sample_xors_uniform()
    elif config["xor_strategy"] == "fourier_weight":
        return sample_xors_fourier_weight(power=config["xor_sampling_power"])
    elif config["xor_strategy"] == "hamming_weight":
        return sample_xors_hamming_weight(hamming_weight=config["xor_hamming_weight"])
    else:
        raise ValueError(f"Unknown xor strategy {config['xor_strategy']}")


def get_network(config: dict[str, Any], system: SpinSystem, signal) -> nn.Module:
    if config["architecture"] == "dense":
        return SignDenseNet(
            system, n_hidden=config["n_hidden"], hidden_layers=config["hidden_layers"]
        )
    elif config["architecture"] == "dense+xor":
        xor_masks = get_sample_strategy(config)(system, signal, size=config["n_xors"])
        net = SignDenseNetXor(
            system=system,
            n_hidden=config["n_hidden"],
            hidden_layers=config["hidden_layers"],
            xor_masks=xor_masks,
        )
        return net
    elif config["architecture"] == "invariant_cnn":
        assert config["use_symmetries"], "CNNs require symmetries for correct evaluation"
        return InvariantSpinCNNRegression(
            lattice=get_lattice(config["lattice"]),
            hidden_channels=config["hidden_channels"],
            dilations=config["dilations"],
            kernel_size=config['kernel_size']
            out_dim=2,
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
    outputs = sign_net(states).to(device)
    return (1 - 2 * torch.argmax(outputs, dim=1)).detach().numpy()


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
            )
            system.get_eigenstates(1)

            signal_fn = signal_factory(system)
            sign_overlap_fn = sign_overlap(system)
            accuracy_fn = accuracy(system)

            for eps_train in config["eps_train"]:
                logger.debug(f"{eps_train=}. Making train and test states...")
                n_train = int(system.basis.states.shape[0] * eps_train)
                n_test = config["n_test"]
                train_states, test_states = mk_train_test(
                    system,
                    n_train=n_train,
                    n_test=n_test,
                    sampling_power_train=config["sampling_power_train"],
                )
                net = get_network(config, system, signal=signal_factory)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

                # Create a TensorDataset from your inputs X and Y
                dataset = TensorDataset(
                    torch.from_numpy(train_states.astype(np.int64)),
                    torch.from_numpy(signal_fn(train_states) == -1).to(torch.long),
                )

                dataloader = DataLoader(
                    dataset, batch_size=config["batch_size"], shuffle=config["shuffle"]
                )

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                net.to(device)

                for epoch in range(config["epochs"]):
                    train_loss = train(net, dataloader, criterion, optimizer, device)
                    if epoch % config["write_each_epoch"] == 0:
                        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                        logger.info(f"Epoch {epoch}, train loss: {train_loss:.4f}")
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
                                }
                            )


if __name__ == "__main__":
    fire.Fire(main)
