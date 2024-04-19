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
from spin_systems import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem
from nn_xors_2023_07_18 import MLPBinaryClassifier, make_dataset
from spin_lattices import KagomeLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

eps_train = 0.05
eps_test = 0.1

batch_size = 64
n_hidden = 512
epochs = 300
break_on_loss = 1e-2
lr = 1e-3


system_specs = [
    (TriangularLattice(6, 4), 1.3),
    (KagomeLattice(2, 4), 1.0),
]


def main(task_id: int | None = None, splits=10):
    # task_list = list(product(system_specs, range(runs)))
    # if task_id is None:
    #     print("You have to specify task_id. Available tasks:")
    #     for i, task in enumerate(task_list):
    #         print(i, task)
    #     return
    run = task_id

    for (lattice, J2), alpha in product(system_specs, [0.001, 1.0]):
        system = HeisenbergJ1J2(lattice, J1=1, J2=J2, ground_state_cache_dir=Path("groundstates"))
        system.get_eigenstates(1)
        n_spins = system.number_spins

        signal = LBFFromSpinSystem(system)
        series = fourier_expand(signal)

        for split in range(splits):
            all_terms_log = np.log(len(series.coeffs))
            keep_terms = int(np.exp(all_terms_log * (split + 1) / splits))

            transformed_series = series.truncate(keep_largest_n(keep_terms))

            task_name = f"{system.get_cache_id()}_{split}_{run}_{alpha}"

            writer = SummaryWriter(
                log_dir=(
                    f"experiments/{self_name}.tb/{datetime.now().strftime('%H_%M_%S')}"
                    f"_{task_name}"
                )
            )
            probs = (
                np.abs(system.get_ground_state_in_canonical_basis().astype(np.float64)) ** alpha
            )
            probs /= probs.sum()
            all_states = system.canonical_basis.states
            sample_states = np.random.choice(
                all_states,
                size=int(len(all_states) * (eps_train + eps_test)),
                replace=False,
                p=probs,
            )

            dataset = make_dataset(transformed_series, sample_states, n_spins)

            train_dataset, test_dataset = random_split(
                dataset,
                [eps_train / (eps_train + eps_test), eps_test / (eps_train + eps_test)],
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
                        "run": run,
                        "non_zero_coeffs": int((transformed_series.coeffs != 0).sum()),
                        "system": system.get_cache_id(),
                        "lattice": lattice.get_cache_id(),
                        "J2": J2,
                        "keep_terms": keep_terms,
                        "split": split,
                        "alpha": alpha,
                    }
                )
            )


if __name__ == "__main__":
    fire.Fire(main)
