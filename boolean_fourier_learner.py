import lzma
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from parity import calculate_fourier_transform_matrix
from tqdm import tqdm


class BooleanFourierLearner:
    def __init__(self, number_spins: int, subsets: Optional[np.ndarray] = None):
        self.number_spins = number_spins
        if subsets is not None:
            self.subsets = subsets
        else:
            self.subsets = np.arange(2**number_spins, dtype="uint64")
        self.coeffs_: np.ndarray
        self.x_: np.ndarray
        self.y_: np.ndarray

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[npt.NDArray[np.float64]] = None,
        batch_size=None,
        stochastic_iterations=None,
        pickle_progress_to=None,
        pickle_each=1,
        show_progress=False,
    ):
        if len(x) != len(y):
            raise ValueError("Lengths of x and y should coincide")

        if weights is None:
            weights = np.ones_like(x)

        if len(x) != len(weights):
            raise ValueError("Lengths of x and weights should coincide")

        sample_size = len(x)

        if not (
            not pickle_progress_to
            or stochastic_iterations
            or batch_size is None
            or sample_size % batch_size == 0
        ):
            raise NotImplementedError(
                "If pickle_progress_to is specified, either stochastic_iteration should be True, "
                "or sample size should be a factor of batch_size"
            )

        if y.dtype == "uint8" or y.dtype == "int8":
            print("Warning! Possible overfulls due to small dtype of y. Converting to int64")
            y = y.astype("int64")

        if pickle_progress_to is not None:
            Path(pickle_progress_to).parent.mkdir(parents=True, exist_ok=True)

        if batch_size is not None and not stochastic_iterations:
            n_batches = sample_size // batch_size + (sample_size % batch_size != 0)
        elif not stochastic_iterations:
            n_batches = 1
            batch_size = sample_size
        else:
            n_batches = stochastic_iterations

        sumprod = np.zeros(len(self.subsets), dtype="float64")
        columns = 0

        equal_batches = stochastic_iterations or sample_size % batch_size == 0  # type: ignore
        # we know that batch_size is not None if stochastic_iterations is False

        for i in [lambda _: _, tqdm][show_progress](range(n_batches)):
            if not stochastic_iterations:
                x_batch = x[i * batch_size : (i + 1) * batch_size]
                y_batch = y[i * batch_size : (i + 1) * batch_size]
                weights_batch = weights[i * batch_size : (i + 1) * batch_size]
            else:
                indicies = np.random.choice(np.arange(sample_size), batch_size, replace=False)
                x_batch = x[indicies]
                y_batch = y[indicies]
                weights_batch = weights[indicies]

            fourier_transform_matrix = np.asanyarray(
                calculate_fourier_transform_matrix(x_batch, self.subsets).T, order="C"
            )

            sumprod_delta = (fourier_transform_matrix @ (weights_batch * y_batch)).astype(
                "float64"
            )

            if equal_batches:
                sumprod += sumprod_delta / batch_size / n_batches
                self.coeffs_ = sumprod
                if pickle_progress_to and (i % pickle_each) == (pickle_each - 1):
                    with lzma.open(pickle_progress_to.format(i=i + 1), "wb") as f:
                        pickle.dump(self, f)
            else:
                sumprod += sumprod_delta
                columns += len(x_batch)

        if not equal_batches:
            self.coeffs_ = sumprod / columns
        self.x_ = x
        self.y_ = y

    def _ensure_fitted(self):
        if not hasattr(self, "coeffs_"):
            raise ValueError("Model is not fitted; please, run .fit(x, y, ...) first")

    def get_coeffs_ser(self) -> pd.Series:
        """
        Returns a pandas Series with coefficients of the Fourier expansion.
        The index is the subset of sites, the value is the coefficient.
        Sorted by absolute value of the coefficient.

        Returns
        -------
        coeffs_ser : pd.Series
        """
        self._ensure_fitted()

        if not hasattr(self, "coeffs_ser_"):
            self.coeffs_ser_ = (
                pd.DataFrame(dict(coeff=self.coeffs_), index=self.subsets)
                .assign(abs_coeff=lambda x: np.abs(x["coeff"]))
                .sort_values("abs_coeff", ascending=False)["coeff"]
            )

        return self.coeffs_ser_
