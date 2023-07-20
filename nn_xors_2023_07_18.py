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

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)
logger.add(output_dir / "log_{time}.log")


class FourierSeries:
    def __init__(self, xors: npt.NDArray[np.uint64], coeffs: npt.NDArray[np.float64]):
        self.xors = xors
        self.coeffs = coeffs
        assert len(xors) == len(coeffs)

    def __call__(self, x: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
        return calculate_fourier_transform_matrix(x, self.xors).astype(np.float64) @ self.coeffs


def similar_distance_network(
    all_xors, distance_min, distance_max, maxitems=10, n_possible_new_xors=100
):
    new_xors = []
    for _ in range(maxitems):
        possible_new_xors = np.unique(
            np.random.choice(all_xors, size=n_possible_new_xors, replace=True)
        )
        distances = popcount(all_xors.reshape(-1, 1) ^ possible_new_xors.reshape(1, -1))
        good_elements = ((distance_min <= distances) & (distances <= distance_max)).sum(axis=0)

        new_xor = possible_new_xors[np.argmax(good_elements)]
        new_xors.append(new_xor)

        new_distances = popcount(all_xors ^ new_xor)
        all_xors = all_xors[(distance_min <= new_distances) & (new_distances <= distance_max)]
        if len(all_xors) == 0:
            break
    return np.array(new_xors)


class MLPBinaryClassifier(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def make_dataset(series: FourierSeries, states: npt.NDArray[np.uint64], n_spins) -> TensorDataset:
    x = torch.tensor(
        make_unpacked_configurations(states, n_spins).astype(np.float32), dtype=torch.float32
    )
    y = torch.tensor(series(states) > 0, dtype=torch.long)
    return TensorDataset(x, y)


def train(
    net,
    criterion,
    optimizer,
    train_loader,
    test_dataset,
    n_epochs,
    writer: SummaryWriter,
    break_on_loss: None | float = None,
):
    for epoch in range(n_epochs):
        hits = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = net(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            hits += (yhat.argmax(dim=1) == y).float().sum()
        train_accuracy = hits / len(train_loader.dataset)

        writer.add_scalar("train/loss", loss.item(), epoch)
        writer.add_scalar("train/accuracy", train_accuracy, epoch)
        with torch.no_grad():
            x, y = test_dataset[:]
            yhat = net(x)
            # find accuracy
            test_accuracy = (yhat.argmax(dim=1) == y).float().mean()
            writer.add_scalar("test/accuracy", test_accuracy, epoch)
            logger.debug(
                f"Epoch\t{epoch}\tloss\t{loss.item():.4f}"
                f"\taccuracy\t{train_accuracy:.4f}\tAccuracy (test)\t{test_accuracy:.4f}"
            )
        if break_on_loss is not None and loss < break_on_loss:
            break
    return {
        "loss": loss.item(),
        "train_accuracy": train_accuracy.item(),
        "test_accuracy": test_accuracy.item(),
        "epoch": epoch,
    }


def main():
    n_spins = 24
    eps_test = 0.03
    all_states = np.arange(2**n_spins, dtype=np.uint64)
    all_states = all_states[popcount(all_states) == n_spins // 2]
    eps_train = 0.003 * comb(n_spins, n_spins // 2) / len(all_states)
    batch_size = 64
    n_hidden = 512
    runs = 20
    for n_xors, xor_hamming_weight, distance in [
        (1, 10, 2),
        (2, 10, 2),
        #    (2, 10, 4),
        (2, 10, 6),
        #    (2, 10, 8),
        (2, 10, 10),
        #    (2, 10, 12),
        (2, 10, 14),
        #    (2, 10, 16),
        (2, 10, 18),
        (2, 10, 20),
        (3, 8, 2),
        (3, 8, 4),
        (3, 8, 6),
        (3, 8, 10),
        (3, 8, 12),
        (3, 8, 14),
        (3, 8, 16),
        (8, 8, 2),
        (8, 8, 4),
        (8, 8, 6),
        (8, 8, 10),
        (8, 8, 12),
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

            for _ in range(2):
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

            output = train(
                net,
                criterion,
                optimizer,
                train_loader,
                test_dataset=test_dataset,
                n_epochs=500,
                writer=writer,
                break_on_loss=1e-04,
            )
            logger.debug(f"Finished training, {output=}")

            with jsonlines.open(output_dir / "results.jsonl", mode="a") as writer:
                writer.write(
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


if __name__ == "__main__":
    main()
