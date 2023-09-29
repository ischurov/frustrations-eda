import json
from collections.abc import Callable
from datetime import datetime
from itertools import product
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from fast_boolean_analysis import FourierSeries, fourier_expand, keep_largest_n
from fourier_xor_shuffle_2023_07_24 import (
    identity,
    replace_xors_with_random,
    shuffle_xors,
    train,
)
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem
from misc_utils import Compose
from nn_xors_2023_07_18 import MLPBinaryClassifier, make_dataset
from spin_lattices import KagomeLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

eps_train = 0.1
eps_test = 0.1

batch_size = 256
n_hidden = 512
epochs = 3000
target_rel_weight = 0.2
runs = 20
break_on_loss = 1e-2
lr = 1e-3


def truncate(series: FourierSeries) -> FourierSeries:
    return series.truncate(
        keep_largest_n(series.how_many_terms_to_achieve_relative_weight(target_rel_weight))
    )


system_specs = [
    (KagomeLattice(2, 4), 1.0),
]

transformations = [
    Compose(identity),
    Compose(truncate),
    Compose(truncate, replace_xors_with_random),
    Compose(truncate, shuffle_xors),
]


def main(task_id: int | None = None):
    run = task_id

    for (lattice, J2), transform in product(system_specs, transformations):
        system = HeisenbergJ1J2(lattice, J1=1, J2=J2, ground_state_cache_dir=Path("groundstates"))
        system.get_eigenstates(1)

        task_name = f"{system.get_cache_id()}_{transform}_{run}"
        n_spins = system.number_spins

        writer = SummaryWriter(
            log_dir=(
                f"experiments/{self_name}.tb/{datetime.now().strftime('%H_%M_%S')}" f"_{task_name}"
            )
        )

        all_states = system.canonical_basis.states
        sample_states = np.random.choice(
            all_states,
            size=int(len(all_states) * (eps_train + eps_test)),
            replace=False,
            p=system.get_ground_state_in_canonical_basis().astype(np.float64) ** 2,
        )

        signal = LBFFromSpinSystem(system)
        series = fourier_expand(signal)

        transformed_series = transform(series)
        dataset = make_dataset(transformed_series, sample_states, n_spins)

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

        Path(output_dir / f"{task_name}.json").write_text(
            json.dumps(
                output
                | {
                    "transform": repr(transform),
                    "run": run,
                    "non_zero_coeffs": int((transformed_series.coeffs != 0).sum()),
                    "system": system.get_cache_id(),
                    "lattice": lattice.get_cache_id(),
                }
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
