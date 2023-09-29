from pathlib import Path

import fire
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image

from nqs_playground_helpers import (
    SamplingOptions,
    forward_with_batches,
    safe_exp,
    sample_exactly,
)
from swo_validation_2023_09_14 import configs, default_config, get_setup, output_dir


def main(task_id: int, step: int = 1):
    config = default_config | configs[task_id]
    logger.info(f"Running task {task_id} with config {config}")
    logger.info("Loading setup...")
    lattice, system, net_factory = get_setup(task_id)
    logger.info("Getting ground state...")
    energy, ground_state = system.get_eigenstates(1)
    ground_state = ground_state[:, 0]
    log_prob_fn = net_factory()

    np.save(output_dir / str(task_id) / "ground_state.npy", ground_state)

    for i in range(0, config["power_iterations"], step):
        snapshot = output_dir / str(task_id) / f"log_prob_fn_{i}.pt"
        logger.info(f"Processing {snapshot}")
        filename = snapshot.name
        log_prob_fn.load_state_dict(torch.load(snapshot))
        logger.info(f"Applying log_prob_fn to all states...")
        # _, _, predicted_probs = sample_exactly(
        #     log_prob_fn,
        #     system.basis,
        #     SamplingOptions(
        #         number_samples=1,
        #         number_chains=1,
        #         mode="exact",
        #         sweep_size=1,
        #         number_discarded=0,
        #     ),
        #     return_all_probs=True,
        # )
        with torch.no_grad():
            log_prob = forward_with_batches(
                log_prob_fn, torch.from_numpy(system.basis.states.view(np.int64)), batch_size=8192
            )
            logger.info("Computing predicted probs...")
            predicted_probs = safe_exp(log_prob.view(-1)).detach().numpy()

        # predicted_probs = safe_exp(forward_with_batches(log_prob_fn, system.basis.states, 8096))
        # predicted_probs = predicted_probs.detach().cpu().numpy()

        logger.info(f"Saving outputs for {snapshot}")
        # write predicted probs
        predicted_probs_path = (
            output_dir / str(task_id) / filename.replace(".pt", "_predicted_probs.npy")
        )
        np.save(predicted_probs_path, predicted_probs)


if __name__ == "__main__":
    fire.Fire(main)
