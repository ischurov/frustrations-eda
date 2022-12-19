import numpy as np
from boolean_analysis import calculate_fourier_transform_matrix, parity_of_1s
import pandas as pd
from tqdm import tqdm
import pickle
import lzma


class BooleanFourierLearner:
    def __init__(self, number_spins: int, subsets: np.ndarray = None):
        self.number_spins = number_spins
        if subsets is not None:
            self.subsets = subsets
        else:
            self.subsets = np.arange(2**number_spins, dtype="uint64")
        self.coeffs_ = None
        self.x_ = None
        self.y_ = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size=None,
        stochastic_iterations=None,
        pickle_progress_to=None,
        pickle_each=1,
    ):
        if len(x) != len(y):
            raise ValueError("Lengths of x and y should coincide")

        if stochastic_iterations is None and pickle_progress_to is not None:
            raise NotImplementedError(
                "pickle_progress_to is supported only in stochastic mode"
            )

        if y.dtype == "uint8" or y.dtype == "int8":
            print(
                "Warning! Possible overfulls due to small dtype of y. Converting to float64"
            )
            y = y.astype("float64")

        sample_size = len(x)
        if batch_size and not stochastic_iterations:
            n_batches = sample_size // batch_size + (sample_size % batch_size != 0)
        elif not stochastic_iterations:
            n_batches = 1
            batch_size = sample_size
        else:
            n_batches = stochastic_iterations

        sumprod = np.zeros(len(self.subsets), dtype="float64")
        columns = 0

        for i in tqdm(range(n_batches)):
            if not stochastic_iterations:
                x_batch = x[i * batch_size : (i + 1) * batch_size]
                y_batch = y[i * batch_size : (i + 1) * batch_size]
            else:
                indicies = np.random.choice(
                    np.arange(sample_size), batch_size, replace=False
                )
                x_batch = x[indicies]
                y_batch = y[indicies]

            fourier_transform_matrix = calculate_fourier_transform_matrix(
                x_batch,
                self.subsets,
                self.number_spins,
                show_progress=True,
            )

            self.fourier_transform_matrix = fourier_transform_matrix

            sumprod_delta = (fourier_transform_matrix.T @ y_batch).astype("float64")

            if stochastic_iterations:
                sumprod += sumprod_delta / batch_size / n_batches
                self.coeffs_ = sumprod
                if pickle_progress_to and i % pickle_each == 0:
                    with lzma.open(pickle_progress_to.format(i=i), "wb") as f:
                        pickle.dump(self, f)

            else:
                sumprod += sumprod_delta
                columns += len(x_batch)

        if not stochastic_iterations:
            self.coeffs_ = sumprod / columns
        self.x_ = x
        self.y_ = y

    def _assure_fitted(self):
        if self.coeffs_ is None:
            raise ValueError("Model is not fitted; please, run .fit(x, y, ...) first")

    def get_coeffs_df(self, abs_largest_first=True):
        self._assure_fitted()
        df = pd.DataFrame(dict(coeff=self.coeffs_), index=self.subsets)
        if abs_largest_first:
            return (
                df.assign(abs_coeff=lambda x: np.abs(x["coeff"]))
                .sort_values("abs_coeff", ascending=False)
                .drop("abs_coeff", axis=1)
            )
        else:
            return df
