print("Running dilated_nns_xors.py")

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
from typing import Callable, Any

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name

default_config = {
    "eps_train": [0.1, 0.05],
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
    "last_layer_bias": False,
    "threshold_sign_tol": 1e-14,
    "kernel_size": 3,
}

configs = {
    0: {
        "lattice": "square6x4",
        "xor_physical_size": 3,
        "xor_num_ones": 7,
        "xor_num_terms": 3,
    },
    1: {
        "_inherit": 0,
        "dilations": [1, 2, 3],
    },
    2: {
        "_inherit": 0,
        "dilations": [3, 2, 1],
    },
    3: {
        "_inherit": 0,
        "xor_physical_size": 4,
    },
    4: {
        "_inherit": 3,
        "dilations": [1, 2, 3],
    },
    5: {
        "_inherit": 3,
        "dilations": [3, 2, 1],
    },
    6: {
        "lattice": "square5x5",
        "xor_physical_size": 3,
        "xor_num_ones": 7,
        "xor_num_terms": 3,
        "runs": 20,
    },
    7: {
        "_inherit": 6,
        "dilations": [1, 2, 3],
    },
    8: {
        "_inherit": 6,
        "dilations": [3, 2, 1],
    },
    9: {
        "_inherit": 6,
        "xor_physical_size": 4,
    },
    10: {
        "_inherit": 9,
        "dilations": [1, 2, 3],
    },
    11: {
        "_inherit": 9,
        "dilations": [3, 2, 1],
    },
    12: {
        "_inherit": 6,
        "xor_physical_size": 5,
    },
    13: {
        "_inherit": 12,
        "dilations": [1, 2, 3],
    },
    14: {
        "_inherit": 12,
        "dilations": [3, 2, 1],
    },
    15: {
        "_inherit": 6,
        "xor_num_terms": 1,
    },
    16: {
        "_inherit": 15,
        "dilations": [1, 2, 3],
    },
    17: {
        "_inherit": 15,
        "dilations": [3, 2, 1],
    },
    18: {
        "_inherit": 15,
        "xor_physical_size": 4,
    },
    19: {
        "_inherit": 18,
        "dilations": [1, 2, 3],
    },
    20: {
        "_inherit": 18,
        "dilations": [3, 2, 1],
    },
    21: {
        "_inherit": 15,
        "xor_physical_size": 5,
    },
    22: {
        "_inherit": 21,
        "dilations": [1, 2, 3],
    },
    23: {
        "_inherit": 21,
        "dilations": [3, 2, 1],
    },
    24: {
        "_inherit": 19,
        "hidden_channels": [64, 64, 64],
        "epochs": 500,
        "runs": 5,
    },
    25: {
        "_inherit": 18,
        "hidden_channels": [32, 32, 32, 32],
        "dilations": None,
        "epochs": 500,
        "runs": 5,
    },
    26: {
        "_inherit": 25,
        "dilations": [1, 2, 3, 3],
    },
    27: {
        "_inherit": 25,
        "dilations": [1, 2, 2, 3],
    },
}


def count_ones_in_window(matrix: npt.NDArray, window_size: int) -> npt.NDArray:
    conv2d = EquivariantConv2d(1, 1, window_size, 1)
    conv2d.conv.weight.data = torch.nn.Parameter(
        torch.ones_like(conv2d.conv.weight.data), requires_grad=False
    )
    conv2d.conv.bias.data = torch.nn.Parameter(
        torch.zeros_like(conv2d.conv.bias.data), requires_grad=False
    )
    input_tensor = torch.from_numpy(matrix.reshape(1, 1, *matrix.shape)).float()
    return conv2d(input_tensor).detach().numpy().reshape(*matrix.shape)


def make_random_xor(
    lattice: ParallelogramSpinLattice, size: int, num_ones: int, attempts=1000
):
    if num_ones > size**2:
        raise ValueError("num_ones must be less than or equal to size ** 2")
    if num_ones < 2:
        raise ValueError("num_ones must be at least 2")
    sequence = np.zeros(size**2)
    sequence[:num_ones] = 1
    for _ in range(attempts):
        np.random.shuffle(sequence)
        patch = sequence.reshape(size, size)
        matrix = np.zeros((lattice.width, lattice.height))
        matrix[:size, :size] = patch

        # make sure that there are no smaller windows
        # that contain all the ones
        if count_ones_in_window(matrix, size - 1).max() < num_ones:
            break
    else:
        raise ValueError(
            "Could not construct the random xor with given size; probably, it's impossible. Try increasing num_ones"
        )
    assert count_ones_in_window(matrix, size).max() == num_ones

    spin_config = np.zeros(lattice.number_spins)
    spin_config[lattice.num_tensor_order] = matrix.reshape(-1)
    return spin_config


def make_random_fourier_series(
    lattice: ParallelogramSpinLattice,
    physical_size: int,
    num_ones: int,
    coeffs: npt.NDArray[np.float64],
    make_symmetric: bool = False,
) -> FourierSeries:
    xors = []
    coeffs_duplicated = []
    for coeff in coeffs:
        random_xors = make_random_xor(lattice, physical_size, num_ones).reshape(1, -1)
        if make_symmetric:
            random_xors = random_xors[0][
                np.array(lattice.get_automorphisms()).reshape(-1)
            ].reshape(-1, lattice.number_spins)
        xor_idxs = lattice.pack_configurations(random_xors)
        xors.append(xor_idxs)
        coeffs_duplicated.append(np.repeat(coeff, len(xor_idxs)))

    return FourierSeries(
        xors=np.concatenate(xors), coeffs=np.concatenate(coeffs_duplicated)
    )


def resolve_config_inheritance(task_id: int, configs: dict[int, dict[str, Any]]):
    """
    Get the config for the task_id, recursively resolving inheritance if necessary.
    """
    config = configs[task_id]
    visited = set([task_id])
    while "_inherit" in config:
        inherited = config.pop("_inherit")
        if inherited in visited:
            raise ValueError(f"Circular inheritance detected: {visited}")
        visited.add(inherited)

        inherited_config = configs[inherited]
        config = inherited_config | config

    return config


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)


def get_series(config: dict) -> FourierSeries:
    coeffs = np.random.uniform(-1, 1, size=config["xor_num_terms"])
    series = make_random_fourier_series(
        lattice=get_lattice(config["lattice"]),
        physical_size=config["xor_physical_size"],
        num_ones=config["xor_num_ones"],
        coeffs=coeffs,
        make_symmetric=True,
    )
    return series


def accuracy(
    ground_truth: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    tol: float = 0.0,
):
    def wrapper(states: npt.NDArray, sign_net: nn.Module, device: torch.device):
        true_signs = thresholded_sign(ground_truth(states), tol=tol)
        predicted_signs = get_predicted_signs(states, sign_net, device)
        mask = (true_signs != 0) & (predicted_signs != 0)
        return np.mean(true_signs[mask] == predicted_signs[mask])

    return wrapper


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
                last_layer_bias=config["last_layer_bias"],
            )

            net.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
            logger.debug("Creating TensorDataset and DataLoader...")
            # Create a TensorDataset from your inputs X and Y

            series = get_series(config)

            target = torch.from_numpy(series(train_states) < 0).to(torch.long)
            accuracy_fn = accuracy(series, tol=config["threshold_sign_tol"])
            # dataset = TensorDataset(
            #     torch.from_numpy(train_states.astype(np.int64)).to(device),
            #     target.to(device),
            # )

            # dataloader = DataLoader(
            #     dataset, batch_size=config["batch_size"], shuffle=config["shuffle"]
            # )

            dataset = SpinDataset(
                train_states,
                target,
                batch_size=config["batch_size"],
                shuffle=config["shuffle"],
                device=device,
            )

            for epoch in range(config["epochs"]):
                logger.debug("Training")
                # with torch.profiler.profile(
                #     activities=[
                #         torch.profiler.ProfilerActivity.CPU,
                #         torch.profiler.ProfilerActivity.CUDA,
                #     ],
                #     schedule=torch.profiler.schedule(wait=10, warmup=10, active=10),
                #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
                #         dir_name=output_dir / str(task_id) / "logs",
                #     ),
                #     record_shapes=True,
                #     profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                #     with_stack=True,
                # ) as p:
                train_loss = train(
                    net, dataset, criterion, optimizer, device  # , profiler=p
                )
                if epoch % config["write_each_epoch"] == 0:
                    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                    logger.info(f"Epoch {epoch}, train loss: {train_loss:.8f}")
                    # logger.info("Writing test predictions")
                    # np.save(
                    #     output_dir
                    #     / str(task_id)
                    #     / f"prediction_{run}_{J2}_{eps_train}_{epoch}.npy",
                    #     net(torch.from_numpy(test_states.astype(np.int64)).to(device))
                    #     .detach()
                    #     .cpu()
                    #     .numpy(),
                    # )
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
                            }
                        )


if __name__ == "__main__":
    fire.Fire(main)
