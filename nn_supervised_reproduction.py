from pathlib import Path

import fire
import numpy as np
import numpy.typing as npt
import torch
from jsonlines import jsonlines
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fourier_supervised_cleanroom import mk_train_test, sign_signal
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from misc_utils import keep_serializable, make_unpacked_configurations
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
}


def get_config(task_id: int):
    return default_config | configs[task_id % len(configs)]


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


def get_network(config, system):
    if config["architecture"] == "dense":
        return SignDenseNet(
            system, n_hidden=config["n_hidden"], hidden_layers=config["hidden_layers"]
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


def sign_overlap(system: SpinSystem):
    def wrapper(states: npt.NDArray, sign_net: nn.Module, device: torch.device):
        true_signs = np.sign(system.get_ground_state_coeffs(states))
        probs = np.abs(system.get_ground_state_coeffs(states)) ** 2
        outputs = sign_net(states).to(device)
        predicted_signs = (1 - 2 * torch.argmax(outputs, dim=1)).detach().numpy()

        return np.sum(true_signs * predicted_signs * probs) / np.sum(probs)

    return wrapper


def main(task_id: int):
    config = get_config(task_id)
    lattice = get_lattice(config["lattice"])
    J2s = config["J2s"]

    (output_dir / str(task_id)).mkdir(parents=True, exist_ok=True)

    for J2 in J2s:
        logger.debug(f"Running {task_id=} {J2=}. Creating system...")
        system = HeisenbergJ1J2(
            lattice=lattice, J1=1, J2=J2, use_symmetries=False, spin_inversion=None
        )
        system.get_eigenstates(1)
        signal_fn = sign_signal(system)
        sign_overlap_fn = sign_overlap(system)

        for eps_train in config["eps_train"]:
            logger.debug(f"{eps_train=}. Making train and test states...")
            n_train = int(system.canonical_basis.states.shape[0] * eps_train)
            n_test = config["n_test"]
            train_states, test_states = mk_train_test(
                system,
                n_train=n_train,
                n_test=n_test,
                sampling_power_train=config["sampling_power_train"],
            )
            net = get_network(config, system)
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
                    logger.info(f"Epoch {epoch}, train loss: {train_loss:.4f}")
                    test_overlap = sign_overlap_fn(test_states, net, device)
                    logger.info(f"Overlap: {test_overlap:.4f}")
                    with jsonlines.open(
                        output_dir / str(task_id) / f"results.jsonl", mode="a"
                    ) as writer:
                        writer.write(
                            keep_serializable(config)
                            | {
                                "test_overlap": test_overlap,
                                "train_loss": train_loss,
                                "epoch": epoch,
                                "J2": J2,
                                "eps_train": eps_train,
                            }
                        )


if __name__ == "__main__":
    fire.Fire(main)
