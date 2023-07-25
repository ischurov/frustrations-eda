from fourier_xor_shuffle_2023_07_24 import train, replace_xors_with_random, shuffle_xors, identity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from itertools import product
from pathlib import Path
from loguru import logger
from nn_xors_2023_07_18 import MLPBinaryClassifier, make_dataset
from spin_lattices import TriangleLattice, SquareLattice, KagomeLattice
from heisenberg_hamiltonians import HeisenbergJ1J2
import numpy as np
from lattice_boolean_analysis import LBFFromSpinSystem
from fast_boolean_analysis import FourierSeries, fourier_expand, keep_largest_n
from collections.abc import Callable
import fire
import json

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

eps_train = 0.01
eps_test = 0.1

batch_size = 64
n_hidden = 512
epochs = 300
target_rel_weight = 0.2
runs = 20
break_on_loss = 1e-2
lr = 1e-3


def main(task_id: int | None = None, runs: int = 20):
    run = task_id

    lattice_triangle = TriangleLattice(6, 4)
    system_triangle = HeisenbergJ1J2(
        lattice_triangle, J1=1, J2=1.2, ground_state_cache_dir=Path("groundstates")
    )
    system_triangle.get_eigenstates(1)
    series_triangle = fourier_expand(LBFFromSpinSystem(system_triangle))

    lattice_kagome = KagomeLattice(2, 4)
    system_kagome = HeisenbergJ1J2(
        lattice_kagome, J1=1, J2=1, ground_state_cache_dir=Path("groundstates")
    )
    system_kagome.get_eigenstates(1)
    series_kagome = fourier_expand(LBFFromSpinSystem(system_kagome))

    n_terms = series_triangle.how_many_terms_to_achieve_relative_weight(target_rel_weight)

    series_triangle_truncated = series_triangle.truncate(keep_largest_n(n_terms))
    series_kagome_truncated = series_kagome.truncate(keep_largest_n(n_terms))

    for system, series, transform in [
        (system_triangle, series_triangle_truncated, identity),
        (system_triangle, series_triangle_truncated, shuffle_xors),
        (system_kagome, series_kagome_truncated, identity),
    ]:
        task_name = f"{system.get_cache_id()}_{transform.__name__}_{run}"
        n_spins = system.number_spins

        writer = SummaryWriter(
            log_dir=(
                f"experiments/{self_name}.tb/{datetime.now().strftime('%H_%M_%S')}_{task_name}"
            )
        )

        all_states = system.canonical_basis.states
        sample_states = np.random.choice(
            all_states,
            size=int(len(all_states) * (eps_train + eps_test)),
            replace=False,
            p=system.get_ground_state_in_canonical_basis().astype(np.float64) ** 2,
        ).astype(np.uint64)

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
                    "run": run,
                    "non_zero_coeffs": int((transformed_series.coeffs != 0).sum()),
                    "system": system.get_cache_id(),
                    "lattice": system.lattice.get_cache_id(),
                    "transform": transform.__name__,
                }
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
