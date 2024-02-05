from itertools import product
from pathlib import Path

import fire
import jsonlines
import numpy as np
from loguru import logger

from fourier_paper_reproduction import (
    accuracy as thresholded_accuracy,
    get_ipr,
    how_many_terms_to_achieve,
    how_many_terms_to_achieve_relative_weight,
    rel_fourier_weight_in_largest_terms,
    sign_overlap,
)
from fourier_supervised_cleanroom import (
    amplitude_median_bin_signal as thresholded_amplitude_median_bin_signal,
    amplitude_prob_median_bin_signal,
    amplitude_signal,
    fit_fourier_series,
    ground_state_signal,
    hadamard_transform,
    keep_largest_n,
    sign_signal as thresholded_sign_signal,
)
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name
output_dir.mkdir(exist_ok=True)

tol = 1e-14


def sign_signal(system: SpinSystem):
    return thresholded_sign_signal(system, tol=tol)


def amplitude_median_bin_signal(system: SpinSystem):
    return thresholded_amplitude_median_bin_signal(system, tol=tol)


def accuracy(system: SpinSystem, signal_fn, states=None):
    return thresholded_accuracy(system, signal_fn, states=states, tol=tol)


default_config = {
    "J2s": np.linspace(0, 1, 21),
    "signals": [
        sign_signal,
        ground_state_signal,
        amplitude_median_bin_signal,
        amplitude_prob_median_bin_signal,
        amplitude_signal,
    ],
    "scorers": [sign_overlap, accuracy],
    "thresholds": [0.02, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
    "run_id": "",
    "sign_tol": tol,
}

configs = {
    0: {"lattice": "kagome2x2"},
    1: {"lattice": "kagome2x3"},
    2: {"lattice": "kagome2x4"},
    3: {"lattice": "kagome3x3"},
    4: {"lattice": "square4x3"},
    5: {"lattice": "square4x4"},
    6: {"lattice": "square5x4"},
    7: {"lattice": "square6x4"},
    8: {"lattice": "square7x4"},
    9: {"lattice": "square5x5"},
    10: {"lattice": "triangular4x3", "J2s": np.linspace(0, 1.4, 29)},
    11: {"lattice": "triangular4x4", "J2s": np.linspace(0, 1.4, 29)},
    12: {"lattice": "triangular5x4", "J2s": np.linspace(0, 1.4, 29)},
    13: {"lattice": "triangular6x4", "J2s": np.linspace(0, 1.4, 29)},
    14: {
        "lattice": "triangular7x4",
        "J2s": np.linspace(0, 1.4, 29),
        "scorers": [accuracy],
        "thresholds": [0.2, 0.8],
        "signals": [sign_signal, amplitude_median_bin_signal],
    },
    15: {"lattice": "triangular5x5", "J2s": np.linspace(0, 1.4, 29)},
    16: {
        "lattice": "kagome3x3",
        "scorers": [accuracy],
        "thresholds": [0.2, 0.8],
        "signals": [sign_signal, amplitude_median_bin_signal],
    },
    17: {
        "lattice": "square7x4",
        "scorers": [accuracy],
        "thresholds": [0.2, 0.8],
        "signals": [sign_signal, amplitude_median_bin_signal],
    },
    18: {"lattice": "kagome3x2"},
    19: {"lattice": "kagome4x2"},
    20: {"lattice": "triangular6x4", "J2s": np.linspace(0, 1.4, 29)[:10]}, # the same as 13
    21: {"lattice": "triangular6x4", "J2s": np.linspace(0, 1.4, 29)[10:20]}, # the same as 13
    22: {"lattice": "triangular6x4", "J2s": np.linspace(0, 1.4, 29)[20:]}, # the same as 13
}


def get_config(task_id: int):
    return default_config | configs[task_id]


def main(task_id: int, run_id: int | str | None = None):
    config = get_config(task_id)
    if run_id is not None:
        config["run_id"] = run_id

    logger.info(f"Running task {task_id} with config {config}")

    lattice = get_lattice(config["lattice"])
    J2s = config["J2s"]
    signals = config["signals"]
    scorers = config["scorers"]
    thresholds = config["thresholds"]

    (output_dir / str(task_id)).mkdir(exist_ok=True)

    for J2, signal_factory, scorer_factory, threshold in product(
        J2s, signals, scorers, thresholds
    ):
        logger.info(
            f"Running J2={J2}, signal={signal_factory.__name__}, "
            f"scorer={scorer_factory.__name__}, "
            f"threshold={threshold}"
        )
        system = HeisenbergJ1J2(
            lattice, J1=1, J2=J2, use_symmetries=False, spin_inversion=None
        )
        system.get_eigenstates(1)

        signal_fn = signal_factory(system)
        scorer_fn = scorer_factory(system, signal_fn=signal_fn)

        series_coeffs = fit_fourier_series(
            system.canonical_basis.states,
            signal_fn=signal_fn,
            n_bits=system.number_spins,
        )

        terms_score = how_many_terms_to_achieve(series_coeffs, threshold, scorer_fn)
        achieved_sign_overlap_for_terms_score = sign_overlap(
            system, signal_fn=signal_fn
        )(keep_largest_n(series_coeffs, terms_score))

        with jsonlines.open(output_dir / str(task_id) / f"results.jsonl", "a") as f:
            f.write(
                {
                    "J2": J2,
                    "lattice": lattice.get_cache_id(),
                    "n_spins": system.number_spins,
                    "terms_score_threshold": terms_score,
                    "relweight_corresponding_to_terms": rel_fourier_weight_in_largest_terms(
                        series_coeffs, terms_score
                    ),
                    "terms_relweight_threshold": how_many_terms_to_achieve_relative_weight(
                        series_coeffs, threshold
                    ),
                    "achieved_sign_overlap_for_terms_score": achieved_sign_overlap_for_terms_score,
                    "iipr": 1.0 / get_ipr(series_coeffs),
                    "scorer": scorer_factory.__name__,
                    "signal": signal_factory.__name__,
                    "threshold": threshold,
                }
            )


if __name__ == "__main__":
    fire.Fire(main)
