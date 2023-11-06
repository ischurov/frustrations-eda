from time import perf_counter
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from fourier_paper_reproduction import how_many_terms_to_achieve
from fourier_supervised_cleanroom import (
    fit_fourier_series,
    hadamard_transform,
    keep_largest_n,
    mk_train_test,
    sign_signal,
)
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from parity import calculate_fourier_transform_matrix
from spin_lattices import KagomeLattice


def naive_fourier_predict(
    coeffs: npt.NDArray,
    states: npt.NDArray[np.uint64],
    subsets: npt.NDArray[np.uint64] | None = None,
    max_batch_size=256,
):
    logger.debug("Making sure that coeffs are of dtype float64")
    coeffs = np.asarray(coeffs, dtype="float64")

    if subsets is None:
        assert np.allclose(
            np.log2(len(coeffs)) % 1, 0
        ), "The number of coefficients must be a power of 2."
        subsets = np.arange(len(coeffs), dtype="uint64")

    predictions = []
    if max_batch_size is None or max_batch_size > len(states):
        max_batch_size = len(states)
    for batch, x_batch in enumerate(np.array_split(states, len(states) // max_batch_size)):
        logger.debug(f"Processing batch {batch} of size {len(x_batch)} out of {len(states)}")
        logger.debug("Calculating Fourier transform matrix")
        transform_matrix = calculate_fourier_transform_matrix(
            states=x_batch, subsets=subsets, out_dtype="float64"
        )
        logger.debug(
            f"{transform_matrix.shape=}, {transform_matrix.size * transform_matrix.itemsize=}"
        )
        # logger.debug("Converting to float64")
        # transform_matrix = np.asarray(transform_matrix, dtype=np.float64)
        logger.debug("Finding product")
        prediction = transform_matrix @ coeffs
        logger.debug("Converting to float64")
        prediction = np.asarray(
            prediction,
            dtype="float64",
        )
        logger.debug("Appending to predictions")

        predictions.append(prediction)

    prediction = np.concatenate(predictions)
    return prediction


def hadamard_predict(coeffs, states):
    return hadamard_transform(coeffs)[states]


def sign_overlap(
    system: SpinSystem,
    signal_fn: Callable[[npt.NDArray[np.uint64]], npt.NDArray[np.float64]],
    states: npt.NDArray[np.uint64] | None = None,
    prediction_fn: Callable[
        [npt.NDArray, npt.NDArray[np.uint64]], npt.NDArray
    ] = naive_fourier_predict,
):
    if states is None:
        states = system.canonical_basis.states

    ground_truth = np.sign(signal_fn(states))
    probs = system.get_ground_state_coeffs(states) ** 2

    def wrapper(fourier_series: npt.NDArray[np.float64]):
        predictions = np.sign(prediction_fn(fourier_series, states))
        return np.sum(ground_truth * predictions * probs) / np.sum(probs)

    return wrapper


n_train = 10000
n_runs = 10
lattice = KagomeLattice(2, 4)
system = HeisenbergJ1J2(lattice, J2=1, use_symmetries=False, spin_inversion=None)

test_states = np.random.choice(system.canonical_basis.states, size=n_train, replace=False)
system.get_eigenstates(1)
signal = sign_signal(system)
scorer = sign_overlap(system, signal_fn=signal, states=test_states)
logger.debug("Fitting Fourier series")
out = []
for n_threads in [1, 2, 4, 8, 16]:
    logger.debug(f"Setting number of threads to {n_threads}")
    torch.set_num_threads(n_threads)

    for _ in range(n_runs):
        tick = perf_counter()
        series = fit_fourier_series(
            system.canonical_basis.states, signal_fn=signal, n_bits=system.number_spins
        )
        tock = perf_counter()
        out.append({"n_threads": n_threads, "time": tock - tick})

    logger.debug("Done fitting Fourier series")
    logger.debug(f"Average time: {(tock - tick) / n_runs}")

print(out)
# n_terms = how_many_terms_to_achieve(series, 0.8, scorer)
# print(n_terms)
