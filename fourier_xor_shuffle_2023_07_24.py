from datetime import datetime
from itertools import product
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from fast_boolean_analysis import FourierSeries, fourier_expand, keep_largest_n
from spin_systems import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem
from misc_utils import (
    groupby_shuffle,
    make_packed_configurations,
    make_unpacked_configurations,
)
from nn_xors_2023_07_18 import MLPBinaryClassifier, make_dataset
from parity import popcount
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

logger.add(output_dir / "log.log")

eps_train = 0.01
eps_test = 0.1
batch_size = 64
n_hidden = 512
epochs = 300
runs = 20

system_specs = [
    (SquareLattice(6, 4), 0.5),
    (KagomeLattice(2, 4), 1.0),
    (TriangularLattice(6, 4), 0.9),
    (TriangularLattice(6, 4), 1.3),
]

keep_xors_list = [2, 3, 4, 10]
break_on_loss = 1e-2
lr = 1e-3


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
        running_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = net(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            hits += (yhat.argmax(dim=1) == y).float().sum()
        train_loss = running_loss / n_epochs
        train_accuracy = hits / len(train_loader.dataset)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_accuracy, epoch)
        with torch.no_grad():
            x, y = test_dataset[:]
            yhat_test = net(x)
            # find accuracy
            test_accuracy = (yhat_test.argmax(dim=1) == y).float().mean()
            writer.add_scalar("test/accuracy", test_accuracy, epoch)
            logger.debug(
                f"Epoch\t{epoch}\tloss\t{train_loss:.4f}"
                f"\taccuracy\t{train_accuracy:.4f}\tAccuracy (test)\t{test_accuracy:.4f}"
            )
        if break_on_loss is not None and train_loss < break_on_loss:
            break
    return {
        "loss": train_loss,
        "train_accuracy": train_accuracy.item(),
        "test_accuracy": test_accuracy.item(),
        "epoch": epoch,
    }


def shuffle_xors(series: FourierSeries) -> FourierSeries:
    return FourierSeries(series.signal, groupby_shuffle(series.coeffs, series.coeffs != 0))


def replace_xors_with_random(series: FourierSeries) -> FourierSeries:
    all_xors = np.arange(len(series.coeffs), dtype=np.uint64)
    all_popcounts = popcount(all_xors)
    new_coeffs = groupby_shuffle(series.coeffs, all_popcounts)
    return FourierSeries(series.signal, new_coeffs)


def identity(series: FourierSeries) -> FourierSeries:
    return series


def main():
    for lattice, J2 in system_specs:
        system = HeisenbergJ1J2(lattice, J1=1, J2=J2, ground_state_cache_dir=Path("groundstates"))
        system.get_eigenstates(1)
        signal = LBFFromSpinSystem(system)
        series = fourier_expand(signal)

        n_spins = system.number_spins
        for keep_xors, transform, run in product(
            keep_xors_list, [identity, replace_xors_with_random, shuffle_xors], range(runs)
        ):
            writer = SummaryWriter(
                log_dir=(
                    f"experiments/{self_name}.tb/{datetime.now().strftime('%H_%M_%S')}"
                    f"_{system.get_cache_id()}"
                    f"_keep_{keep_xors}"
                    f"_{transform.__name__}"
                )
            )
            all_states = system.canonical_basis.states
            sample_states = np.random.choice(
                all_states,
                size=int(len(all_states) * (eps_train + eps_test)),
                replace=False,
            )

            if keep_xors is not None:
                truncated_series = series.truncate_orbitwise(keep_largest_n(keep_xors))
            else:
                truncated_series = series

            truncated_series = transform(truncated_series)

            dataset = make_dataset(truncated_series, sample_states, n_spins)
            train_dataset, test_dataset = random_split(
                dataset, [eps_train / (eps_train + eps_test), eps_test / (eps_train + eps_test)]
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            net = MLPBinaryClassifier(n_spins, n_hidden)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            output = train(
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                test_dataset=test_dataset,
                n_epochs=epochs,
                writer=writer,
                break_on_loss=break_on_loss,
            )

            outfile = output_dir / f"output.jsonl"
            with jsonlines.open(outfile, mode="a") as writer:
                writer.write(
                    output
                    | {
                        "keep": keep_xors,
                        "transform": transform.__name__,
                        "run": run,
                        "non_zero_coeffs": int((truncated_series.coeffs != 0).sum()),
                        "system": system.get_cache_id(),
                    }
                )

            logger.debug(f"Finished training, {output=}")


if __name__ == "__main__":
    main()
