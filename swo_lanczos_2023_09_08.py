from itertools import product
from pathlib import Path

import fire
import numpy as np
import torch
from jsonlines import jsonlines
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from heisenberg_hamiltonians import HeisenbergJ1J2
from misc_utils import torch_overlap as overlap
from my_stopwatch import stopwatch
from nqs_playground_helpers import SamplingOptions, sample_exactly
from spin_lattices import KagomeLattice
from swo import generate_training_set, generate_training_set_lanczos
from vmc_amplitude import LogProbDenseNet, almost_true_relsigns

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)


def true_amplitudes(system, states):
    return torch.from_numpy(np.abs(system.get_ground_state_coeffs(states))).float()


def main(task_id: int):
    n_samples_list = [1000, 5000, 25000]
    energy_baseline_list = [None, 0.0, 10.0, 50.0, 250.0]
    lattice_list = [KagomeLattice(2, 3), KagomeLattice(2, 4), KagomeLattice(2, 5)]

    configs = list(product(n_samples_list, energy_baseline_list, lattice_list))
    n_samples, energy_baseline, lattice = configs[task_id]
    run = task_id // len(configs)

    (output_dir / str(task_id)).mkdir(exist_ok=True)

    if energy_baseline is None:
        mode = "lanczos"
    else:
        mode = "power"

    test_samples = 50000
    power_iterations = 500
    epochs = 100
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
        use_symmetries=False,
        spin_inversion=None,
    )
    system_nosym = system

    system.get_eigenstates(1)

    stopwatch.reset()

    relsigns_fn = almost_true_relsigns(system, eps=0.0)

    logger.info(f"Running {mode=} with run {run}")
    log_prob_fn = LogProbDenseNet(system_nosym, n_hidden=n_hidden)

    optimizer = torch.optim.Adam(log_prob_fn.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for iteration in range(power_iterations):
        logger.info(f"Running iteration {iteration}")
        logger.info(f"Sampling")
        states, _ = sample_exactly(
            log_prob_fn,
            system.canonical_basis,
            SamplingOptions(
                number_samples=n_samples,
                number_chains=1,
                mode="exact",
                sweep_size=1,
                number_discarded=0,
            ),
            return_all_probs=False,
        )
        logger.info("Sampling successful")
        states = states.view(-1)
        remaining_states = np.delete(
            system.canonical_basis.states,
            system.canonical_basis.index(states.detach().numpy()),
        )
        states_test = torch.from_numpy(
            np.random.choice(remaining_states, test_samples, replace=True).astype(np.int64)
        ).long()
        #            all_states = np.concatenate([states.detach().numpy(), states_test.detach().numpy()])
        logger.info("Test states sampled")
        if mode == "power":
            logger.info("Generating new train set with power method")
            target = generate_training_set(
                hamiltonian=system_nosym.hamiltonian,
                states=states.detach().numpy(),
                log_prob_fn=lambda s: log_prob_fn(s).detach().numpy().reshape(-1),
                relsigns_fn=relsigns_fn,
                energy_baseline=energy_baseline,
            )
            logger.info("Done")
            alpha = None
        elif mode == "lanczos":
            logger.info("Generating new train set with Lanczos")
            target, alpha = generate_training_set_lanczos(
                hamiltonian=system_nosym.hamiltonian,
                states=states.detach().numpy(),
                log_prob_fn=lambda s: log_prob_fn(s).detach().numpy().reshape(-1),
                relsigns_fn=relsigns_fn,
            )
            logger.info(f"Done, alpha={alpha}")
        else:
            raise ValueError(f"Unknown mode {mode}")

        target = torch.from_numpy(target).float()

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
        #     torch.exp(log_prob_fn(system.basis.states).view(-1) * 0.5), true_amplitudes
        # )

        initial_ground_state_overlap_train = overlap(
            torch.exp(log_prob_fn(states).view(-1) * 0.5),
            true_amplitudes(system, states),
        )

        initial_ground_state_overlap_test = overlap(
            torch.exp(log_prob_fn(states_test).view(-1) * 0.5),
            true_amplitudes(system, states_test),
        )

        target_ground_state_overlap_train = overlap(
            torch.exp(target.view(-1) * 0.5), true_amplitudes(system, states)
        )

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
            with jsonlines.open(output_dir / str(task_id) / "losses.jsonl", mode="a") as writer:
                writer.write(
                    {
                        "iteration": iteration,
                        "epoch": epoch,
                        "loss": running_loss / len(dataloader),
                    }
                    | params_dict
                )

        # new_ground_state_overlap_full = overlap(
        #     torch.exp(log_prob_fn(system.basis.states).view(-1) * 0.5), true_amplitudes
        # )

        logger.info("Finished training step")

        new_ground_state_overlap_train = overlap(
            torch.exp(log_prob_fn(states).view(-1) * 0.5),
            true_amplitudes(system, states),
        )

        new_ground_state_overlap_test = overlap(
            torch.exp(log_prob_fn(states_test).view(-1) * 0.5),
            true_amplitudes(system, states_test),
        )

        with jsonlines.open(output_dir / str(task_id) / "overlaps.jsonl", mode="a") as writer:
            writer.write(
                {
                    "iteration": iteration,
                    # "initial_ground_state_overlap_full": initial_ground_state_overlap_full.item(),
                    "initial_ground_state_overlap_train": initial_ground_state_overlap_train.item(),
                    "initial_ground_state_overlap_test": initial_ground_state_overlap_test.item(),
                    "target_ground_state_overlap_train": target_ground_state_overlap_train.item(),
                    #            "target_ground_state_overlap_test": target_ground_state_overlap_test.item(),
                    # "new_ground_state_overlap_full": new_ground_state_overlap_full.item(),
                    "new_ground_state_overlap_train": new_ground_state_overlap_train.item(),
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
