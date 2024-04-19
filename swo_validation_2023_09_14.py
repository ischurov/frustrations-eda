from itertools import product
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from jsonlines import jsonlines
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gcnn_naive import SplitGroupResConvNet
from spin_systems import HeisenbergJ1J2, SpinSystem
from kagome_round import get_kagome27, get_kagome36
from misc_utils import torch_overlap as overlap
from my_stopwatch import stopwatch
from nqs_playground_helpers import SamplingOptions, sample_exactly
from spin_lattices import KagomeLattice
from swo import generate_training_set, generate_training_set_lanczos
from vmc_amplitude import (
    LogProbDenseNet,
    LogProbFn,
    almost_true_relsigns,
    random_relsigns,
)

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

default_config = {
    "n_samples": 5000,
    "energy_baseline": 10.0,
    "lattice_name": "kagome27",
    "epochs": 100,
    "reset_network": False,
    "net": "LogProbDenseNet",
    "use_symmetries": True,
    "power_iterations": 50,
    "early_stop": False,
    "resample_every": 1,
    "snapshot_each": None,
    "n_hidden": 512,
    "hidden_layers": 1,
    "sampling_power": 2.0,
    "batch_size": 64,
    "lr": 1e-3,
    "sign": "true",
}

configs = {
    0: {"lattice_name": "kagome36"},
    1: {"energy_baseline": 250.0, "lattice_name": "kagome36"},
    2: {},
    3: {"energy_baseline": 250.0},
    4: {"net": "SplitGroupResConvNet"},
    5: {"energy_baseline": 250.0, "net": "SplitGroupResConvNet"},
    6: {"epochs": 10, "net": "SplitGroupResConvNet"},
    7: {"energy_baseline": 250.0, "epochs": 10, "net": "SplitGroupResConvNet"},
    8: {"lattice_name": "kagome2x4"},
    9: {"energy_baseline": 250.0, "lattice_name": "kagome2x4"},
    10: {"epochs": 10, "net": "SplitGroupResConvNet", "use_symmetries": False},
    11: {
        "energy_baseline": 250.0,
        "epochs": 10,
        "net": "SplitGroupResConvNet",
        "use_symmetries": False,
    },
    12: {"epochs": 10, "use_symmetries": False},
    13: {"energy_baseline": 250.0, "epochs": 10, "use_symmetries": False},
    14: {"lattice_name": "kagome2x4", "epochs": 10, "use_symmetries": False},
    15: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x4",
        "epochs": 10,
        "use_symmetries": False,
    },
    16: {"lattice_name": "kagome2x5", "epochs": 10, "use_symmetries": False},
    17: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x5",
        "epochs": 10,
        "use_symmetries": False,
    },
    18: {"lattice_name": "kagome2x4", "use_symmetries": False},
    19: {"energy_baseline": 250.0, "lattice_name": "kagome2x4", "use_symmetries": False},
    20: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "snapshot_each": 1,
    },
    21: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 1,
        "power_iterations": 500,
        "snapshot_each": 1,
    },
    22: {
        "energy_baseline": 1000.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "snapshot_each": 1,
    },
    23: {
        "energy_baseline": 1000.0,
        "lattice_name": "kagome36",
        "epochs": 1000,
        "early_stop": True,
        "power_iterations": 500,
        "snapshot_each": 1,
    },
    24: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 1000,
        "early_stop": True,
        "power_iterations": 500,
        "snapshot_each": 1,
    },
    25: {
        "energy_baseline": 1000.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25000,
        "snapshot_each": 1,
    },
    26: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25000,
    },
    27: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 5000,
        "resample_every": 10,
    },
    28: {  # copy of 20
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
    },
    29: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 50,
        "power_iterations": 500,
    },
    30: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome27",
        "epochs": 10,
        "power_iterations": 50,
        "snapshot_each": 1,
    },
    31: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x4",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 10_000,
        "snapshot_each": 1,
        "sampling_power": 1.0,
    },
    32: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 100_000,
        "snapshot_each": 1,
    },
    33: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 100_000,
        "snapshot_each": 1,
        "sampling_power": 1.0,
    },
    34: {
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 100_000,
        "snapshot_each": 1,
        "sampling_power": 0.5,
    },
    35: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 5_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 2,
    },
    36: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 5_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 1,
        "n_hidden": 2048,
    },
    37: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 5_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 2,
        "n_hidden": 128,
    },
    38: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 2,
    },
    39: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 1,
        "n_hidden": 2048,
    },
    40: {  # architecture search
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25_000,
        "snapshot_each": 1,
        "sampling_power": 1,
        "hidden_layers": 2,
        "n_hidden": 128,
    },
    41: {  # obtaining data for ising
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x3",
        "epochs": 10,
        "power_iterations": 50,
        "snapshot_each": 1,
        "n_samples": 1000,
    },
    42: {  # obtaining data for ising
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x3",
        "epochs": 10,
        "power_iterations": 50,
        "snapshot_each": 1,
        "n_samples": 1000,
        "use_symmetries": False,
    },
    43: {  # obtaining data for ising
        "energy_baseline": 250.0,
        "lattice_name": "kagome2x3",
        "epochs": 10,
        "power_iterations": 250,
        "snapshot_each": 1,
        "n_samples": 1000,
        "use_symmetries": False,
    },
    44: {  # architecture search (baseline 25_000 samples)
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "power_iterations": 500,
        "n_samples": 25_000,
        "snapshot_each": 1,
        "sampling_power": 1,
    },
    45: {  # architecture search (baseline 5_000 samples)
        "energy_baseline": 250.0,
        "lattice_name": "kagome36",
        "epochs": 10,
        "n_samples": 5_000,
        "snapshot_each": 1,
        "sampling_power": 1,
    },
    46: {
        "lattice_name": "kagome2x3",
        "energy_baseline": 250.0,
        "epochs": 10,
        "power_iterations": 250,
        "n_samples": 5000,
        "sign": "random",
    },
    47: {
        "lattice_name": "kagome2x4",
        "energy_baseline": 250.0,
        "epochs": 10,
        "power_iterations": 250,
        "n_samples": 5000,
        "sign": "random",
    },
}


def true_amplitudes(system: SpinSystem, states):
    _, eigenstates = system.get_eigenstates(1)
    ground_state = eigenstates[:, 0].reshape(-1)

    return torch.from_numpy(np.abs(ground_state[system.basis.index(states)])).float()


def get_config(task_id: int):
    return default_config | configs[task_id % len(configs)]


def get_setup(task_id: int):
    config = get_config(task_id)
    lattice_name = config["lattice_name"]
    use_symmetries = config["use_symmetries"]
    net = config["net"]
    n_hidden = config["n_hidden"]
    hidden_layers = config["hidden_layers"]

    generators = None
    filter1_sites = None

    if lattice_name == "kagome36":
        lattice, generators = get_kagome36()
        filter1_sites = [10, 19, 20, 21, 22, 12, 13, 11, 9, 18, 31, 23]
    elif lattice_name == "kagome27":
        lattice, generators = get_kagome27()
        filter1_sites = [14, 15, 16, 17, 8, 6, 13, 24, 18, 9, 7, 5]
    elif lattice_name == "kagome2x3":
        lattice = KagomeLattice(2, 3)
    elif lattice_name == "kagome2x4":
        lattice = KagomeLattice(2, 4)
    elif lattice_name == "kagome2x5":
        lattice = KagomeLattice(2, 5)
    else:
        raise ValueError(f"Unknown lattice {lattice_name}")

    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1.0,
        J2=1.0,
        use_symmetries=use_symmetries,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )

    if net == "LogProbDenseNet":
        net_factory = lambda n_hidden=n_hidden, hidden_layers=hidden_layers: LogProbDenseNet(
            system, n_hidden=n_hidden, hidden_layers=hidden_layers
        )
    elif net == "SplitGroupResConvNet":
        additional_generators = ["rotation", "flip"]
        extend_filter1 = (1, 1)
        filter_size = (2, 2)
        channels = 16
        blocks = 4

        def net_factory() -> nn.Module:
            model = LogProbFn(
                system,
                SplitGroupResConvNet(
                    tx=generators["tx"],
                    ty=generators["ty"],
                    filter1_sites=filter1_sites,
                    additional_generators=[generators[gen] for gen in additional_generators],
                    extend_filter1=extend_filter1,
                    filter_size=filter_size,
                    channels=channels,
                    blocks=blocks,
                ),
            )
            return model

    else:
        raise ValueError(f"Unknown net {net}")

    return lattice, system, net_factory


def main(task_id: int):
    # Get specific configuration by task_id
    config = get_config(task_id)

    lattice, system, net_factory = get_setup(task_id)

    n_samples = config["n_samples"]
    energy_baseline = config["energy_baseline"]
    epochs = config["epochs"]
    reset_network = config["reset_network"]
    power_iterations = config["power_iterations"]
    early_stop = config["early_stop"]
    resample_every = config["resample_every"]
    snapshot_each = config["snapshot_each"]
    sampling_power = config["sampling_power"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    run = task_id // len(configs)

    (output_dir / str(task_id)).mkdir(exist_ok=True)

    if energy_baseline is None:
        mode = "lanczos"
    else:
        mode = "power"

    test_samples = 50000

    params_dict = config | {
        "run": run,
        "mode": mode,
        "lattice": lattice.get_cache_id(),
        "n_spins": lattice.number_spins,
        "task_id": task_id,
    }

    # lattice = KagomeLattice(2, 3)
    # system = HeisenbergJ1J2(lattice, J2=1, use_symmetries=False, spin_inversion=None)
    # eigenvalues, _ = system.get_eigenstates(1)
    # logger.info(f"Ground state energy: {eigenvalues[0]}")
    # lattice = KagomeLattice(3, 4, isotropic=True)
    # assert lattice.number_spins == 36

    # system = HeisenbergJ1J2(
    #     lattice=lattice,
    #     J1=1.0,
    #     use_symmetries=True,
    #     spin_inversion=1,
    #     ground_state_cache_dir=Path("groundstates"),
    #     skip_symmetries_whitelist=True,
    # )

    system.get_eigenstates(1)

    stopwatch.reset()

    if config["sign"] == "true":
        relsigns_fn = almost_true_relsigns(system, eps=0.0)
    elif config["sign"] == "random":
        relsigns_fn = random_relsigns(system)
    else:
        raise ValueError(f"Unknown sign {config['sign']}")

    logger.info(f"Running {mode=} with run {run}")

    log_prob_fn = net_factory()
    previous_log_prob_fn = net_factory()

    optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    states_test_uniform = torch.from_numpy(
        np.random.choice(system.basis.states, test_samples, replace=True).astype(np.int64)
    ).long()

    true_amplitudes_test = true_amplitudes(system, states_test_uniform)
    # true_amplitudes_test /= torch.sqrt(torch.sum(true_amplitudes_test**2))

    for iteration in range(power_iterations):
        if reset_network:
            logger.info("Resetting network")
            log_prob_fn = net_factory()
            optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)

        logger.info(f"Running iteration {iteration}")
        if iteration % resample_every == 0:
            logger.info(f"Sampling")
            all_states, _, all_probs = sample_exactly(
                lambda s: log_prob_fn(s) * sampling_power * 0.5,
                system.basis,
                SamplingOptions(
                    number_samples=n_samples + test_samples,
                    number_chains=1,
                    mode="exact",
                    sweep_size=1,
                    number_discarded=0,
                ),
                return_all_probs=True,
            )
            logger.info("Sampling successful")
            all_states = all_states.view(-1)
            states = all_states[:n_samples]
            states_validation = all_states[n_samples:]

        # #            all_states = np.concatenate([states.detach().numpy(), states_test.detach().numpy()])
        # logger.info("Test states sampled")
        if mode == "power":
            logger.info("Generating new train set with power method")
            all_target, local_energies, new_psi = generate_training_set(
                hamiltonian=system.hamiltonian,
                states=all_states.detach().numpy(),
                log_prob_fn=lambda s: log_prob_fn(s).detach().numpy().reshape(-1),
                relsigns_fn=relsigns_fn,
                energy_baseline=energy_baseline,
            )
            logger.info("Done")
            alpha = None
        else:
            raise ValueError(f"Unknown mode {mode}")
        logger.info("Making TensorDataset")
        target = torch.from_numpy(all_target[:n_samples]).float()
        target_validation = torch.from_numpy(all_target[n_samples:]).float()

        # Create a TensorDataset from your inputs X and Y
        dataset = TensorDataset(states, target)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # log_prob_fn = LogProbDenseNet(system, n_hidden=n_hidden)

        # optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)

        # writer = SummaryWriter(
        #     log_dir=(
        #         f"experiments/{datetime.now().strftime('%Y_%m_%d')}_dense_xor_supervised/{datetime.now().strftime('%H_%M_%S')}"
        #         #        f"xor={xor_masks_idxs}_ch=_{channels}_{lr=}_{blocks=}"
        #     )
        # )

        # initial_ground_state_overlap_full = overlap(
        #     torch.exp(log_prob_fn(system.basis.states).view(-1) * 0.5),
        #     true_amplitudes(system, system.basis.states),
        # )

        # initial_ground_state_overlap_train = overlap(
        #     torch.exp(log_prob_fn(states).view(-1) * 0.5),
        #     true_amplitudes(system, states),
        # )
        logger.info("Finding initial ground state overlap test")
        initial_ground_state_overlap_test = overlap(
            torch.exp(log_prob_fn(states_test_uniform).view(-1) * 0.5),
            true_amplitudes(system, states_test_uniform),
        )

        # target_ground_state_overlap_train = overlap(
        #     torch.exp(target.view(-1) * 0.5), true_amplitudes(system, states)
        # )

        # target_ground_state_overlap_test = overlap(
        #     torch.exp(target_test.view(-1) * 0.5), true_amplitudes[system.basis.index(states_test)]
        # )
        logger.info("Starting training")
        loss_validation = None
        epoch = -1
        for epoch in range(epochs):
            logger.info(f"Running epoch {epoch}")
            running_loss = 0.0

            previous_log_prob_fn.load_state_dict(log_prob_fn.state_dict())

            for action_index, data in enumerate(dataloader, 0):
                # Get the inputs and move them to the specified device
                inputs, log_probs = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = log_prob_fn(inputs)

                # weights = torch.exp(2 * log_amplitudes)
                # weights = weights / torch.sum(weights)

                # Compute loss
                loss = criterion(
                    outputs, log_probs.view(-1, 1)
                )  # Reshape labels to match output shape
                # loss = 1 - overlap(
                #     differentiable_safe_exp(outputs.view(-1) * 0.5),
                #     differentiable_safe_exp(log_probs * 0.5),
                # )
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Collect loss
                running_loss += loss.item()
            logger.info(f"Loss: {running_loss / len(dataloader)}")
            previous_loss_validation = loss_validation
            loss_validation = criterion(
                log_prob_fn(states_validation), target_validation.view(-1, 1)
            ).item()

            with jsonlines.open(output_dir / str(task_id) / "losses.jsonl", mode="a") as writer:
                writer.write(
                    {
                        "iteration": iteration,
                        "epoch": epoch,
                        "global_epoch": iteration * epochs + epoch,
                        "loss_train": running_loss / len(dataloader),
                        "loss_validation": loss_validation,
                    }
                    | params_dict
                )

            if (
                early_stop
                and previous_loss_validation is not None
                and loss_validation > previous_loss_validation
            ):
                logger.info("Early stopping")
                # Roll back to previous model
                log_prob_fn.load_state_dict(previous_log_prob_fn.state_dict())
                break

        # new_ground_state_overlap_full = overlap(
        #     torch.exp(log_prob_fn(system.basis.states).view(-1) * 0.5),
        #     true_amplitudes(system, system.basis.states),
        # )

        logger.info("Finished training step")

        # new_ground_state_overlap_train = overlap(
        #     torch.exp(log_prob_fn(states).view(-1) * 0.5),
        #     true_amplitudes(system, states),
        # )
        predicted_amplitudes_test = torch.exp(log_prob_fn(states_test_uniform).view(-1) * 0.5)
        predicted_amplitudes_test /= torch.sqrt(torch.sum(predicted_amplitudes_test**2))

        new_ground_state_overlap_test = overlap(
            predicted_amplitudes_test,
            true_amplitudes_test,
        )

        previous_predicted_amplitudes_test = np.sqrt(
            all_probs[system.basis.index(states_test_uniform)]
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(true_amplitudes_test, previous_predicted_amplitudes_test, s=0.1)
        ax.plot([0, 1], [0, 1], ls="--", color="C1")
        # make picture log-log
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-9, 1)
        ax.set_ylim(1e-9, 1)
        ax.set_xlabel("True amplitude")
        ax.set_ylabel("Predicted amplitude")

        fig.savefig(str(output_dir / str(task_id) / f"amplitudes_{iteration:04d}.png"))
        plt.close(fig)

        ipr = (
            torch.sum(predicted_amplitudes_test**4)
            / torch.sum(predicted_amplitudes_test**2) ** 2
        ).item()

        if local_energies is not None:
            energy_data = {
                "energy": float(np.mean(local_energies).real),
                "loc_energy_std": float(np.std(local_energies).real),
            }
        else:
            energy_data = {}

        with jsonlines.open(output_dir / str(task_id) / "overlaps.jsonl", mode="a") as writer:
            writer.write(
                {
                    "iteration": iteration,
                    # "initial_ground_state_overlap_full": initial_ground_state_overlap_full.item(),
                    # "initial_ground_state_overlap_train": initial_ground_state_overlap_train.item(),
                    "initial_ground_state_overlap_test": initial_ground_state_overlap_test.item(),
                    # "target_ground_state_overlap_train": target_ground_state_overlap_train.item(),
                    #            "target_ground_state_overlap_test": target_ground_state_overlap_test.item(),
                    # "new_ground_state_overlap_full": new_ground_state_overlap_full.item(),
                    # "new_ground_state_overlap_train": new_ground_state_overlap_train.item(),
                    "new_ground_state_overlap_test": new_ground_state_overlap_test.item(),
                    "alpha": alpha.real if alpha is not None else None,
                    "epoch": epoch,
                    "ipr": float(ipr),
                }
                | params_dict
                | energy_data
            )

        if snapshot_each is not None and iteration % snapshot_each == 0:
            logger.info("Saving snapshot")
            torch.save(
                log_prob_fn.state_dict(),
                output_dir / str(task_id) / f"log_prob_fn_{iteration}.pt",
            )

        # # Add average loss per epoch to TensorBoard
        # writer.add_scalar("Training Loss", running_loss / len(dataloader), epoch)

        # Calculate overlaps and add them to TensorBoard
        # overlap_train = overlap(
        #     torch.exp(log_prob_fn(states).view(-1) * 0.5), torch.exp(target * 0.5)
        # )
        # logger.info(f"Overlap with target (train): {overlap_train.item()}")

        # overlap_test = overlap(
        #     torch.exp(log_prob_fn(states_test).view(-1) * 0.5),
        #     torch.exp(target_test * 0.5),
        # )
        # logger.info(f"Overlap with target (test): {overlap_test.item()}")

        # writer.add_scalar("train/overlap", overlap_train, epoch)
        # # writer.add_scalar('Overlap Test', overlap_test, epoch)
        # writer.add_scalar("test/overlap", overlap_test, epoch)

        # print("Finished Training")


if __name__ == "__main__":
    fire.Fire(main)
