from itertools import product
from pathlib import Path

import fire
import numpy as np
import torch
from jsonlines import jsonlines
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gcnn_naive import SplitGroupResConvNet
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from kagome_round import get_kagome27, get_kagome36
from misc_utils import torch_overlap as overlap
from my_stopwatch import stopwatch
from nqs_playground_helpers import SamplingOptions, sample_exactly
from spin_lattices import KagomeLattice
from swo import generate_training_set, generate_training_set_lanczos
from vmc_amplitude import LogProbDenseNet, LogProbFn, almost_true_relsigns

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)


def true_amplitudes(system: SpinSystem, states):
    _, eigenstates = system.get_eigenstates(1)
    ground_state = eigenstates[:, 0].reshape(-1)

    return torch.from_numpy(np.abs(ground_state[system.basis.index(states)])).float()


def main(task_id: int):
    # fmt: off
    configs = [
        (5000, 10.0, "kagome36", 100, False, "LogProbDenseNet", True),          # 0
        (5000, 250.0, "kagome36", 100, False, "LogProbDenseNet", True),         # 1
        (5000, 10.0, "kagome27", 100, False, "LogProbDenseNet", True),          # 2
        (5000, 250.0, "kagome27", 100, False, "LogProbDenseNet", True),         # 3
        (5000, 10.0, "kagome27", 100, False, "SplitGroupResConvNet", True),     # 4
        (5000, 250.0, "kagome27", 100, False, "SplitGroupResConvNet", True),    # 5
        (5000, 10.0, "kagome27", 10, False, "SplitGroupResConvNet", True),      # 6
        (5000, 250.0, "kagome27", 10, False, "SplitGroupResConvNet", True),     # 7
        (5000, 10.0, "kagome2x4", 100, False, "LogProbDenseNet", True),         # 8
        (5000, 250.0, "kagome2x4", 100, False, "LogProbDenseNet", True),        # 9
        (5000, 10.0, "kagome27", 10, False, "SplitGroupResConvNet", False),     # 10
        (5000, 250.0, "kagome27", 10, False, "SplitGroupResConvNet", False),    # 11
        (5000, 10.0, "kagome27", 10, False, "LogProbDenseNet", False),          # 12
        (5000, 250.0, "kagome27", 10, False, "LogProbDenseNet", False),         # 13
    ]
    # fmt: on

    n_samples, energy_baseline, lattice_name, epochs, reset_network, net, use_symmetries = configs[
        task_id % len(configs)
    ]

    if lattice_name == "kagome36":
        lattice, generators = get_kagome36()
        filter1_sites = [10, 19, 20, 21, 22, 12, 13, 11, 9, 18, 31, 23]
    elif lattice_name == "kagome27":
        lattice, generators = get_kagome27()
        filter1_sites = [14, 15, 16, 17, 8, 6, 13, 24, 18, 9, 7, 5]
    elif lattice_name == "kagome2x4":
        lattice = KagomeLattice(2, 4)
    else:
        raise ValueError(f"Unknown lattice {lattice_name}")
    run = task_id // len(configs)

    (output_dir / str(task_id)).mkdir(exist_ok=True)

    if energy_baseline is None:
        mode = "lanczos"
    else:
        mode = "power"

    test_samples = 50000
    power_iterations = 50
    lr = 1e-3
    n_hidden = 512
    batch_size = 64

    params_dict = {
        "run": run,
        "mode": mode,
        "n_samples": n_samples,
        "lattice": lattice.get_cache_id(),
        "energy_baseline": energy_baseline,
        "n_spins": lattice.number_spins,
        "epochs": epochs,
        "reset_network": reset_network,
        "net": net,
        "use_symmetries": use_symmetries,
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

    system = HeisenbergJ1J2(
        lattice=lattice,
        J1=1.0,
        J2=1.0,
        use_symmetries=use_symmetries,
        spin_inversion=None,
        skip_symmetries_whitelist=True,
    )

    if net == "LogProbDenseNet":
        net_factory = lambda: LogProbDenseNet(system, n_hidden=n_hidden)
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

    system.get_eigenstates(1)

    stopwatch.reset()

    relsigns_fn = almost_true_relsigns(system, eps=0.0)

    logger.info(f"Running {mode=} with run {run}")
    log_prob_fn = net_factory()
    optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for iteration in range(power_iterations):
        if params_dict["reset_network"]:
            log_prob_fn = net_factory()
            optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)

        logger.info(f"Running iteration {iteration}")
        logger.info(f"Sampling")
        all_states, _ = sample_exactly(
            log_prob_fn,
            system.basis,
            SamplingOptions(
                number_samples=n_samples + test_samples,
                number_chains=1,
                mode="exact",
                sweep_size=1,
                number_discarded=0,
            ),
            return_all_probs=False,
        )
        logger.info("Sampling successful")
        all_states = all_states.view(-1)
        states = all_states[:n_samples]
        states_validation = all_states[n_samples:]

        remaining_states = np.delete(
            system.basis.states,
            system.basis.index(all_states.detach().numpy()),
        )
        states_test_uniform = torch.from_numpy(
            np.random.choice(remaining_states, test_samples, replace=True).astype(np.int64)
        ).long()
        # #            all_states = np.concatenate([states.detach().numpy(), states_test.detach().numpy()])
        # logger.info("Test states sampled")
        if mode == "power":
            logger.info("Generating new train set with power method")
            all_target = generate_training_set(
                hamiltonian=system.hamiltonian,
                states=all_states.detach().numpy(),
                log_prob_fn=lambda s: log_prob_fn(s).detach().numpy().reshape(-1),
                relsigns_fn=relsigns_fn,
                energy_baseline=energy_baseline,
            )
            logger.info("Done")
            alpha = None
        elif mode == "lanczos":
            logger.info("Generating new train set with Lanczos")
            all_target, alpha = generate_training_set_lanczos(
                hamiltonian=system.hamiltonian,
                states=all_states.detach().numpy(),
                log_prob_fn=lambda s: log_prob_fn(s).detach().numpy().reshape(-1),
                relsigns_fn=relsigns_fn,
            )
            logger.info(f"Done, alpha={alpha}")
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
        for epoch in range(epochs):
            logger.info(f"Running epoch {epoch}")
            running_loss = 0.0
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

        # new_ground_state_overlap_full = overlap(
        #     torch.exp(log_prob_fn(system.basis.states).view(-1) * 0.5),
        #     true_amplitudes(system, system.basis.states),
        # )

        logger.info("Finished training step")

        # new_ground_state_overlap_train = overlap(
        #     torch.exp(log_prob_fn(states).view(-1) * 0.5),
        #     true_amplitudes(system, states),
        # )

        new_ground_state_overlap_test = overlap(
            torch.exp(log_prob_fn(states_test_uniform).view(-1) * 0.5),
            true_amplitudes(system, states_validation),
        )

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
                }
                | params_dict
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
