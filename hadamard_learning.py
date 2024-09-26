import numpy as np
import lattice_symmetries as ls
import numpy.typing as npt
from parity import calculate_fourier_transform_matrix


def evaluate_hadamard_learning_test_set(
    wavefunction: npt.NDArray[np.float64],
    basis: ls.Basis,
    sample_power: float,
    test_size: int,
    sample_size: int | None = None,
    eps_train: float | None = None,
):
    wavefunction_tolerance = 1e-12

    non_zero_terms = (np.abs(wavefunction) > wavefunction_tolerance).sum()

    if (sample_size is None) + (eps_train is None) != 1:
        raise ValueError("Exactly one of sample_size and eps_train must be provided")

    if sample_size is None:
        sample_size = int(eps_train * non_zero_terms)
    else:
        eps_train = sample_size / non_zero_terms

    assert sample_size >= 1

    if sample_power == np.inf:
        sample = np.argsort(np.abs(wavefunction))[-sample_size:]
    else:
        sample = np.random.choice(
            np.arange(len(wavefunction)),
            size=sample_size,
            replace=False,
            p=np.abs(wavefunction) ** sample_power
            / (np.abs(wavefunction) ** sample_power).sum(),
        )

    test_sample = np.random.choice(
        np.setdiff1d(np.where(non_zero_terms), sample),
        size=test_size,
        replace=False,
    )
    test_wavefunction = wavefunction[test_sample]

    transform_matrix = calculate_fourier_transform_matrix(
        basis.states[test_sample], basis.states[sample], out_dtype="float64"
    )

    predictions = transform_matrix @ np.sign(wavefunction[sample])

    return {
        "sample_size": sample_size,
        "eps_train": eps_train,
        "sample_power": sample_power,
        "test_size": test_size,
        "accuracy_test": (
            (np.sign(test_wavefunction) * np.sign(predictions)).mean() + 1
        )
        / 2,
        # "overlap_test": np.abs(gs_truncated_test @ gs_truncated_hadamard)
        # / np.linalg.norm(gs_truncated_test)
        # / np.linalg.norm(gs_truncated_hadamard),
        "sign_overlap_test": (
            np.sign(predictions) * np.sign(test_wavefunction) * test_wavefunction**2
        ).sum()
        / (test_wavefunction**2).sum(),
        # "sign_overlap_with_random": (
        #     np.sign(predictions)
        #     * np.sign(np.random.choice(np.sign(test_wavefunction), size=test_size))
        #     * test_wavefunction**2
        # ).sum()
        # / (test_wavefunction**2).sum(),
    }
