from parity import calculate_fourier_transform_matrix, popcount
import numpy as np
from utils import make_unpacked_configurations, make_packed_configurations
import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim import Adam
from torch.nn import functional as F

# binomial coefficients:
from scipy.special import comb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from loguru import logger
import itertools
import os
from pathlib import Path
import jsonlines
from nn_xors_2023_07_18 import (
    FourierSeries,
    similar_distance_network,
    MLPBinaryClassifier,
    make_dataset,
    train,
)

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)
logger.add(output_dir / "log_{time}.log")


def main():
    n_spins = 24
    eps_test = 0.03
    all_states = np.arange(2**n_spins, dtype=np.uint64)
    all_states = all_states[popcount(all_states) == n_spins // 2]
    eps_train = 0.008 * comb(n_spins, n_spins // 2) / len(all_states)
    batch_size = 64
    n_hidden = 512
    runs = 20
    n_epochs = 1000
    n_super_epochs = 2
    break_on_loss = 1e-04
    for n_xors, xor_hamming_weight, distance in [
        (8, 8, 2),
        (8, 8, 4),
        (8, 8, 6),
        (8, 8, 10),
        (8, 8, 12),
        (1, 10, 2),
        (18, 10, 2),
        (18, 10, 4),
        (18, 10, 6),
        (18, 10, 8),
        (18, 10, 10),
        (18, 10, 12),
        (3, 8, 2),
        (3, 8, 4),
        (3, 8, 6),
        (3, 8, 10),
        (3, 8, 12),
        (3, 8, 14),
        (3, 8, 16),
    ]:
        for run in range(runs):
            logger.debug(f"Starting {n_xors=}, {xor_hamming_weight=}, {distance=}, {run=}")

            writer = SummaryWriter(
                log_dir=(
                    f"experiments/2023_07_18_2/{datetime.now().strftime('%H_%M_%S')}"
                    f"_{n_xors=}_{xor_hamming_weight=}_{distance=}_{run=}"
                )
            )

            sample_states = np.random.choice(
                all_states,
                size=int(len(all_states) * (eps_train + eps_test)),
                replace=False,
            )

            all_xors = np.arange(2**n_spins, dtype=np.uint64)
            all_xors = all_xors[popcount(all_xors) == xor_hamming_weight]

            for _ in range(200):
                selected_xors = similar_distance_network(
                    all_xors, distance_min=distance, distance_max=distance + 2, maxitems=n_xors
                )
                if len(selected_xors) == n_xors:
                    break
            else:
                logger.debug(f"Could not find enough xors: {len(selected_xors)}")
                continue

            series = FourierSeries(selected_xors, 2 * np.random.rand(len(selected_xors)) - 1)

            dataset = make_dataset(series, sample_states, n_spins)
            train_dataset, test_dataset = random_split(
                dataset, [eps_train / (eps_train + eps_test), eps_test / (eps_train + eps_test)]
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            net = MLPBinaryClassifier(n_spins, n_hidden)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

            for super_epoch in range(n_super_epochs):
                output = train(
                    net,
                    criterion,
                    optimizer,
                    train_loader,
                    test_dataset=test_dataset,
                    n_epochs=n_epochs,
                    writer=writer,
                    break_on_loss=break_on_loss,
                )

                output["epoch"] += super_epoch * n_epochs
                logger.debug(f"Super epoch {super_epoch} finished: {output}")

                with jsonlines.open(output_dir / "results.jsonl", mode="a") as f:
                    f.write(
                        {
                            "n_xors": n_xors,
                            "xor_hamming_weight": xor_hamming_weight,
                            "distance": distance,
                            "run": run,
                            "eps_train": eps_train,
                            "batch_size": batch_size,
                            "n_hidden": n_hidden,
                            "model": net.__class__.__name__,
                            **output,
                        }
                    )

                if output["loss"] < break_on_loss:
                    logger.debug(f"Breaking on loss {output['loss']}")
                    break


if __name__ == "__main__":
    main()
