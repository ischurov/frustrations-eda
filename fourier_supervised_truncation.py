import re
from pathlib import Path

import fire
import jsonlines
import numpy as np
from loguru import logger

from fourier_supervised_cleanroom import (
    fit_fourier_series,
    hadamard_transform,
    keep_largest_n,
    mk_train_test,
    sign_signal,
)
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from heisenberg_hamiltonians import HeisenbergJ1J2
from misc_utils import keep_serializable
from spin_lattices import KagomeLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

default_config = {
    "J2": 1,
    "eps_train": 0.05,
    "n_test": 50000,
    "splits": 10,
}

configs = {
    0: {"lattice": "triangular6x4", "J2": 1.3},
}


def get_config(task_id: int):
    return default_config | configs[task_id % len(configs)]


def main(task_id: int):
    config = get_config(task_id)
    lattice = get_lattice(config["lattice"])
    J2 = config["J2"]
    splits = config["splits"]

    logger.debug(f"Running {task_id=} {J2=}. Creating system...")
    system = HeisenbergJ1J2(
        lattice=lattice, J1=1, J2=J2, use_symmetries=False, spin_inversion=None
    )
    system.get_eigenstates(1)
    initial_signal_fn = sign_signal(system)
    initial_signal_fourier_expansion = fit_fourier_series(
        system.canonical_basis.states, initial_signal_fn, system.number_spins
    )
    eps_train = config["eps_train"]

    for split in range(splits):
        all_terms_log = system.number_spins
        initial_keep_terms = int(np.exp(all_terms_log * (split + 1) / splits))
        initial_signal_fourier_expansion_truncated = keep_largest_n(
            initial_signal_fourier_expansion, initial_keep_terms
        )
        truncated_signal = np.sign(
            hadamard_transform(initial_signal_fourier_expansion_truncated, inplace=True)
        )

        def truncated_signal_fn(s):
            return truncated_signal[s]

        logger.debug(f"{eps_train=}. Making train and test states...")
        n_train = int(system.canonical_basis.states.shape[0] * eps_train)
        n_test = config["n_test"]
        train_states, test_states = mk_train_test(
            system,
            n_train=n_train,
            n_test=n_test,
            sampling_power_train=0,
        )

        logger.debug(f"Fitting Fourier series...")
        series = fit_fourier_series(
            train_states,
            truncated_signal_fn,
            system.number_spins,
        )
        ground_truth = truncated_signal_fn(test_states)
        probs_test = np.abs(system.get_ground_state_coeffs(test_states)) ** 2

        for keep_terms in [2**k for k in range(1, system.number_spins)]:
            logger.debug(f"Keeping {keep_terms=} terms...")

            series_truncated = keep_largest_n(series, keep_terms)
            weight_kept = (series_truncated**2).sum() / (series**2).sum()
            reconstructed_signal = hadamard_transform(series_truncated, inplace=True)
            prediction = np.sign(reconstructed_signal[test_states])
            accuracy = np.mean(prediction == ground_truth)
            overlap = (prediction * ground_truth * probs_test).sum() / probs_test.sum()
            logger.debug(f"{accuracy=}, {overlap=}, {weight_kept=}")

            (output_dir / str(task_id)).mkdir(exist_ok=True)

            np.save(
                (output_dir / str(task_id) / f"prediction_{J2=}_{eps_train=}_{keep_terms=}"),
                prediction,
            )
            np.save(
                (output_dir / str(task_id) / f"ground_truth_{J2=}_{eps_train=}_{keep_terms=}"),
                ground_truth,
            )
            np.save(
                (output_dir / str(task_id) / f"probs_test_{J2=}_{eps_train=}_{keep_terms=}"),
                probs_test,
            )

            # save train states and test states
            np.save(
                (output_dir / str(task_id) / f"train_states_{J2=}_{eps_train=}_{keep_terms=}"),
                train_states,
            )
            np.save(
                (output_dir / str(task_id) / f"test_states_{J2=}_{eps_train=}_{keep_terms=}"),
                test_states,
            )

            with jsonlines.open((output_dir / str(task_id)) / "results.jsonl", mode="a") as writer:
                writer.write(
                    keep_serializable(config)
                    | {
                        "initial_keep_terms": int(initial_keep_terms),
                        "keep_terms": int(keep_terms),
                        "accuracy": float(accuracy),
                        "overlap": float(overlap),
                        "weight_kept": float(weight_kept),
                        "eps_train": float(eps_train),
                        "J2": float(J2),
                        "n_spins": int(system.number_spins),
                        "task_id": int(task_id),
                    },
                )


if __name__ == "__main__":
    fire.Fire(main)
