from spin_lattices import ParallelogramSpinLattice
import numpy as np
import numpy.typing as npt
from nn_xors_2023_07_18 import FourierSeries
from nn_supervised_reproduction import train, SpinDataset, get_predicted_signs
from conv2d_circular import InvariantSpinCNNRegression, EquivariantConv2d
from heisenberg_hamiltonians import HeisenbergJ1J2
from fourier_supervised_cleanroom import mk_train_test, thresholded_sign
import torch
from pathlib import Path
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from loguru import logger
from datetime import datetime
from torch import nn
import fire
import jsonlines
from misc_utils import keep_serializable
from typing import Callable
from dilated_nns_xors import resolve_config_inheritance, accuracy

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name

default_config = {
    "eps_train": [0.01, 0.001, 0.0001],
    "n_test": 5000,
    "hidden_channels": [32, 32, 32],
    "dilations": None,
    "epochs": 200,
    "runs": 10,
    "write_each_epoch": 1,
    "lr": 1e-3,
    "batch_size": 4192,
    "shuffle": True,
    "n_train_from_full_space": True,
    "sample_repr_then_apply_random_symmetry": False,
    "sample_with_replacement": False,
    "kernel_size": 3,
    "target_kernel_size": 5,
    "target_hidden_channels": [32, 32, 32],
    "lattice": "square5x5",
}

configs = {
    0: {"kernel_size": 5},
    1: {},
    2: {"dilations": [1, 2, 3]},
    3: {"dilations": [3, 2, 1]},
    4: {"dilations": [2, 2, 2]},
    5: {"lattice": "square5x6", "eps_train": [1e-5, 1e-6]},
    6: {
        "_inherit": 5,
        "dilations": [1, 2, 3],
    },
    7: {"_inherit": 5, "dilations": [3, 2, 1]},
    8: {
        "_inherit": 3,
        "dilations": [2, 2, 2],
    },
    9: {"target_hidden_channels": [64, 64, 64, 64]},
    10: {"target_hidden_channels": [64, 64, 64, 64], "dilations": [1, 2, 3]},
    11: {"target_hidden_channels": [64, 64, 64, 64], "dilations": [3, 2, 1]},
    12: {"_inherit": 9, "eps_train": [2e-5]},
}


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)


def main(task_id: int):
    config = get_config(task_id)
    lattice = get_lattice(config["lattice"])

    (output_dir / str(task_id)).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Torch will use device: {device}")
    system = HeisenbergJ1J2(
        lattice,
        use_symmetries=True,
        skip_symmetries_whitelist=True,
        hamming_weight=None,
    )
    # we don't actually need Heisenberg model here
    # this is just a convenient way to make other helper functions work
    # e.g. get dataset

    for run in range(config["runs"]):
        start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        for eps_train in config["eps_train"]:
            logger.debug(f"{eps_train=}. Making train and test states...")
            if (
                config["sample_repr_then_apply_random_symmetry"]
                and config["n_train_from_full_space"]
            ):
                full_sample_space_size = 2**system.number_spins
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
                sampling_power_train=0,
                sampling_power_test=0,
                apply_random_symmetries=config[
                    "sample_repr_then_apply_random_symmetry"
                ],
                replace=config["sample_with_replacement"],
            )
            train_states_torch = torch.from_numpy(train_states.astype(np.int64)).to(
                device
            )
            np.save(
                output_dir / str(task_id) / f"test_states_{run}_{eps_train}.npy",
                test_states,
            )
            logger.debug("Creating network, criterion, optimizer...")

            net = InvariantSpinCNNRegression(
                lattice=lattice,
                hidden_channels=config["hidden_channels"],
                dilations=config["dilations"],
                kernel_size=config["kernel_size"],
                out_dim=2,
                last_layer_bias=False,
            )

            net.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
            logger.debug("Creating TensorDataset and DataLoader...")

            target_net = InvariantSpinCNNRegression(
                lattice=lattice,
                hidden_channels=config["target_hidden_channels"],
                kernel_size=config["target_kernel_size"],
                out_dim=1,
                last_layer_bias=True,
            )
            target_net.to(device)
            target_net.fc.bias.data = torch.tensor([0.0]).to(device)
            bias = target_net(train_states_torch).median()
            target_net.fc.bias.data = torch.tensor([-bias]).to(device)

            target = (target_net(train_states_torch) < 0).to(torch.long).view(-1)

            accuracy_fn = accuracy(
                lambda x: target_net(torch.from_numpy(x.astype(np.int64)).to(device))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            dataset = SpinDataset(
                train_states,
                target,
                batch_size=config["batch_size"],
                shuffle=config["shuffle"],
                device=device,
            )

            for epoch in range(config["epochs"]):
                logger.debug("Training")

                train_loss = train(
                    net, dataset, criterion, optimizer, device  # , profiler=p
                )
                if epoch % config["write_each_epoch"] == 0:
                    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                    logger.info(f"Epoch {epoch}, train loss: {train_loss:.8f}")

                    logger.info("Evaluating test overlap and accuracy")

                    test_accuracy = accuracy_fn(test_states, net, device)

                    with jsonlines.open(
                        output_dir / str(task_id) / f"results.jsonl", mode="a"
                    ) as writer:
                        writer.write(
                            keep_serializable(config, scalar_only=False)
                            | {
                                "test_accuracy": test_accuracy,
                                "train_loss": train_loss,
                                "epoch": epoch,
                                "eps_train": eps_train,
                                "start_timestamp": start_timestamp,
                                "current_timestamp": current_timestamp,
                                "run": run,
                                "task_id": task_id,
                                "n_train": n_train,
                                "target_mean": target.to(torch.float).mean().item(),
                            }
                        )


if __name__ == "__main__":
    fire.Fire(main)
