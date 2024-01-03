import re
from datetime import datetime
from pathlib import Path

import fire
import jsonlines
import numpy as np
from loguru import logger

from fourier_supervised_cleanroom import (
    fit_fourier_series,
    ground_state_signal,
    hadamard_transform,
    keep_largest_n,
    mk_train_test,
    sign_signal,
)
from heisenberg_hamiltonians import HeisenbergJ1J2, heisenberg_expr
from misc_utils import keep_serializable
from spin_lattices import KagomeLattice, SquareLattice, TriangularLattice

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True, parents=True)

default_config = {
    "J2s": np.linspace(0, 1, 11),
    "eps_train": [0.01, 0.001, 0.0001, 0.00001],
    "n_test": 50000,
    "sampling_power_train": 2.0,
    "runs": 1,
    "hamming_weight": "half",
    "expr": heisenberg_expr,
    "signal_train": "sign",
    "signal_test": "sign",
}

configs = {
    0: {"lattice": "kagome2x4", "runs": 10},
    1: {"lattice": "kagome3x3"},
    2: {"lattice": "square6x4", "J2s": np.linspace(0, 1, 41), "runs": 10},
    3: {"lattice": "triangle6x4", "J2s": np.linspace(0, 1.4, 15), "runs": 10},
    4: {"lattice": "square7x4", "runs": 10},
    5: {"lattice": "triangle7x4", "J2s": np.linspace(0, 1.4, 15), "runs": 10},
    6: {"lattice": "square6x5"},
    7: {"lattice": "triangle6x5", "J2s": np.linspace(0, 1.4, 15)},
    8: {"lattice": "kagome2x4", "sampling_power_train": 1.0, "runs": 10},
    9: {"lattice": "kagome2x4", "sampling_power_train": 0.5, "runs": 10},
    10: {"lattice": "kagome2x4", "sampling_power_train": 0.01, "runs": 10},
    11: {"lattice": "kagome2x4", "sampling_power_train": 4.0, "runs": 10},
    12: {"lattice": "kagome2x4", "sampling_power_train": 6.0, "runs": 10},
    13: {"lattice": "kagome2x4", "sampling_power_train": 8.0, "runs": 10},
    14: {"lattice": "kagome2x4", "sampling_power_train": 10.0, "runs": 10},
    15: {"lattice": "kagome2x4", "sampling_power_train": 20.0, "runs": 10},
    16: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "sampling_power_train": 2,
        "runs": 10,
        "expr": "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + 2 σᶻ₀ σᶻ₁",
        "hamming_weight": None,
    },
    17: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "sampling_power_train": 2,
        "runs": 10,
        "expr": "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀)",
        "hamming_weight": None,
    },
    18: {
        "lattice": "kagome2x4",
        "J2s": [1.0],
        "sampling_power_train": 0,
        "runs": 10,
        "signal_train": "groundstate",
        "signal_test": "sign",
    },
}


def get_config(task_id: int):
    return default_config | configs[task_id]


def get_lattice(lattice_name):
    re_match = re.match(r"([a-z]+)(\d+)x(\d+)", lattice_name)
    if re_match is None:
        raise ValueError(f"Unknown lattice name {lattice_name}")
    lattice_type, n_rows, n_cols = re_match.groups()
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    if lattice_type == "kagome":
        return KagomeLattice(n_rows, n_cols)
    elif lattice_type == "square":
        return SquareLattice(n_rows, n_cols)
    elif lattice_type in ("triangle", "triangular"):
        return TriangularLattice(n_rows, n_cols)
    else:
        raise ValueError(f"Unknown lattice name {lattice_name}")


def get_signal(signal_id):
    if signal_id == "sign":
        return sign_signal
    elif signal_id == "groundstate":
        return ground_state_signal
    else:
        raise ValueError(f"Unknown signal_id {signal_id}")


def main(task_id: int):
    config = get_config(task_id)
    lattice = get_lattice(config["lattice"])
    J2s = config["J2s"]
    for run in range(config["runs"]):
        start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

        for J2 in J2s:
            logger.debug(f"Running {task_id=} {J2=}. Creating system...")
            system = HeisenbergJ1J2(
                lattice=lattice,
                J1=1,
                J2=J2,
                use_symmetries=False,
                spin_inversion=None,
                hamming_weight=config["hamming_weight"],
                expr_str=config["expr"],
            )
            system.get_eigenstates(1)
            # if config["signal_train"] == "groundstate":
            #     signal_fn = ground_state_signal(system)
            # elif config["signal_train"] == "sign":
            #     signal_fn = sign_signal(system)
            # else:
            #     raise ValueError(f"Unknown signal_train {config['signal_train']}")
            signal_train = get_signal(config["signal_train"])(system)
            signal_test = get_signal(config["signal_test"])(system)

            for eps_train in config["eps_train"]:
                logger.debug(f"{eps_train=}. Making train and test states...")
                n_train = int(system.canonical_basis.states.shape[0] * eps_train)
                n_test = config["n_test"]
                train_states, test_states = mk_train_test(
                    system,
                    n_train=n_train,
                    n_test=n_test,
                    sampling_power_train=config["sampling_power_train"],
                )
                logger.debug(f"Fitting Fourier series...")
                series = fit_fourier_series(
                    train_states,
                    signal_train,
                    system.number_spins,
                )
                if config["signal_test"] != "sign":
                    raise NotImplementedError

                ground_truth = signal_test(test_states)
                ground_truth_amplitude = np.abs(system.get_ground_state_coeffs(test_states))
                probs_test = ground_truth_amplitude**2

                previous_prediction = None

                for keep_terms in [2**k for k in range(1, system.number_spins)]:
                    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                    logger.debug(f"Keeping {keep_terms=} terms...")

                    series_truncated = keep_largest_n(series, keep_terms)
                    weight_kept = (series_truncated**2).sum() / (series**2).sum()
                    reconstructed_signal = hadamard_transform(series_truncated, inplace=True)
                    prediction = np.sign(reconstructed_signal[test_states])
                    # prediction_amplitude = np.abs(reconstructed_signal[test_states]) ** (
                    #     1 / config["sampling_power_train"]
                    # )
                    accuracy = np.mean(prediction == ground_truth)
                    overlap = (prediction * ground_truth * probs_test).sum() / probs_test.sum()
                    # overlap_amplitude = (
                    #     prediction_amplitude
                    #     @ ground_truth_amplitude
                    #     / np.linalg.norm(prediction_amplitude)
                    #     / np.linalg.norm(ground_truth_amplitude)
                    # )
                    logger.debug(f"{accuracy=}, {overlap=}, {weight_kept=}")

                    if previous_prediction is not None:
                        accuracy_predicted_vs_previous = np.mean(previous_prediction == prediction)
                        overlap_predicted_vs_previous = (
                            previous_prediction * prediction * probs_test
                        ).sum() / probs_test.sum()
                    else:
                        accuracy_predicted_vs_previous = np.nan
                        overlap_predicted_vs_previous = np.nan

                    previous_prediction = prediction

                    (output_dir / str(task_id)).mkdir(exist_ok=True)

                    np.save(
                        (
                            output_dir
                            / str(task_id)
                            / f"prediction_{J2=}_{eps_train=}_{keep_terms=}"
                        ),
                        prediction,
                    )
                    np.save(
                        (
                            output_dir
                            / str(task_id)
                            / f"ground_truth_{J2=}_{eps_train=}_{keep_terms=}"
                        ),
                        ground_truth,
                    )
                    np.save(
                        (
                            output_dir
                            / str(task_id)
                            / f"probs_test_{J2=}_{eps_train=}_{keep_terms=}"
                        ),
                        probs_test,
                    )

                    # save train states and test states
                    np.save(
                        (
                            output_dir
                            / str(task_id)
                            / f"train_states_{J2=}_{eps_train=}_{keep_terms=}"
                        ),
                        train_states,
                    )
                    np.save(
                        (
                            output_dir
                            / str(task_id)
                            / f"test_states_{J2=}_{eps_train=}_{keep_terms=}"
                        ),
                        test_states,
                    )

                    with jsonlines.open(
                        (output_dir / str(task_id)) / "results.jsonl", mode="a"
                    ) as writer:
                        writer.write(
                            keep_serializable(config)
                            | {
                                "keep_terms": int(keep_terms),
                                "accuracy": float(accuracy),
                                "overlap": float(overlap),
                                "weight_kept": float(weight_kept),
                                "eps_train": float(eps_train),
                                "J2": float(J2),
                                "n_spins": int(system.number_spins),
                                "task_id": int(task_id),
                                "accuracy_predicted_vs_previous": float(
                                    accuracy_predicted_vs_previous
                                ),
                                "overlap_predicted_vs_previous": float(
                                    overlap_predicted_vs_previous
                                ),
                                # "overlap_amplitude": float(overlap_amplitude),
                                "start_timestamp": start_timestamp,
                                "current_timestamp": current_timestamp,
                                "run": run,
                            },
                        )


if __name__ == "__main__":
    fire.Fire(main)
