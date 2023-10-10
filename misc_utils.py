from collections.abc import Callable, Iterable
from math import ceil, sqrt
from pathlib import Path

import jsonlines
import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from loguru import logger

### BASED ON: https://github.com/amitport/hadamard-transform
### MIT LICENSE


@torch.jit.script  # type: ignore
def hadamard_transform_pytorch_inplace(x: torch.Tensor, chunks: int = 8):
    """Fast Walsh–Hadamard transform

    The hadamard transform is not numerically stable by nature (lots of subtractions),
    it is recommended to use with float64 when possible

    :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
              Each vector's length is expected to be a power of 2! (or each row if it is batched)

    :param chunks: The number of chunks to split the Hadamard transform into.
                   This is done to avoid memory issues when the input is too large.

    :return: The normalized Hadamard transform of each vector in x
    """
    with torch.no_grad():
        original_shape = x.shape
        assert 1 <= len(original_shape) <= 2, "input's dimension must be either 1 or 2"
        if len(original_shape) == 1:
            # add fake 1 batch dimension
            # for making the code a follow a single (batched) path
            x = x.unsqueeze(0)
        batch_dim, d = x.shape

        h = 2
        while h <= d:
            hf = h // 2
            d_over_h = d // h
            x = x.view(batch_dim, d_over_h, h)

            chunk_size = (d_over_h + chunks - 1) // chunks
            #        logger.debug(f"iteration {np.log2(h)} of {np.log2(d)}")

            for i in range(chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, d_over_h)
                chunk_slice = slice(chunk_start, chunk_end)
                half_1 = x[:, chunk_slice, :hf].clone()
                x[:, chunk_slice, :hf] += x[:, chunk_slice, hf:]
                x[:, chunk_slice, hf:] *= -1
                x[:, chunk_slice, hf:] += half_1

            h *= 2

        x /= torch.sqrt(torch.scalar_tensor(d))

        return x.view(original_shape)


### END BASED


def make_unpacked_configurations(states: npt.ArrayLike, number_spins: int):
    initial_shape = np.shape(states)
    return (
        (
            np.asarray(states, dtype="uint64").reshape(-1, 1)
            >> np.arange(number_spins, dtype="uint64")
        )
        & 1
    ).reshape(initial_shape + (number_spins,))


def make_packed_configurations(states: npt.ArrayLike, number_spins: int):
    return (np.asarray(states, dtype="uint64") << np.arange(number_spins, dtype="uint64")).sum(
        axis=1
    )


def batched_state_info_df(basis: ls.SpinBasis, states: npt.NDArray[np.uint64]):
    """
    Parameters
    ----------
    basis : ls.SpinBasis
        Basis to use for the state info
    states : npt.NDArray[np.uint64]
        States to get the info for

    Returns
    -------
    pd.DataFrame

    Returns a DataFrame with index states and the following columns:

    - representative: representative of the group trajectory containing the state
    - character: character of the group element that takes the representative to the state
    - norm: normalizing coefficient
    """
    representative, eigenvalue, norm = basis.state_info(states)

    return pd.DataFrame(
        dict(representative=representative, character=eigenvalue, norm=norm),
        index=states,
    )


### BASED ON: https://github.com/amitport/hadamard-transform
### MIT LICENSE
def hadamard_transform(x: npt.NDArray[np.float64], verbose: bool = False):
    """Fast Walsh–Hadamard transform

    The hadamard transform is not numerically stable by nature (lots of subtractions),
    it is recommended to use with float64 when possible

    :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
              Each vector's length is expected to be a power of 2! (or each row if it is batched)
    :return: The normalized Hadamard transform of each vector in x
    """
    original_shape = x.shape
    assert 1 <= len(original_shape) <= 2, "input's dimension must be either 1 or 2"
    if len(original_shape) == 1:
        # add fake 1 batch dimension
        # for making the code a follow a single (batched) path
        x = x[None, :]
    batch_dim, d = x.shape

    h = 2
    while h <= d:
        logger.debug(f"iteration {np.log2(h)} of {np.log2(d)}")
        if verbose:
            print(f"iteration {np.log2(h)} of {np.log2(d)}")

        hf = h // 2

        x = x.view()
        x.shape = (batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = np.concatenate((half_1 + half_2, half_1 - half_2), axis=-1)

        h *= 2

    return (x / np.sqrt(d)).reshape(*original_shape)


# END BASED


def read_jsonl_to_df(file: str | Path | Iterable[str | Path]):
    if isinstance(file, (str, Path)):
        file = [file]
    dataframes = []
    for f in file:
        with jsonlines.open(f) as reader:
            dataframes.append(pd.DataFrame(reader))
    return pd.concat(dataframes).reset_index(drop=True)


def read_json_dir_to_df(dir: str | Path):
    return pd.concat(read_jsonl_to_df(file) for file in Path(dir).glob("*.json")).reset_index(
        drop=True
    )


def get_abslargest_terms(
    coeffs: npt.NDArray, n: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray]:
    """Get the n largest terms in absolute value."""
    abs_values = np.abs(coeffs)
    indices = np.asarray(np.argpartition(abs_values, -n)[-n:], dtype=np.uint64)
    sorted_indices = indices[np.argsort(-abs_values[indices])]
    return sorted_indices, coeffs[sorted_indices]


def ensure_newfile(path: Path):
    if path.exists():
        logger.debug(f"Warning! {path} already exists")
    return path


def one(iterable: Iterable):
    it = iter(iterable)
    x = next(it)
    try:
        next(it)
    except StopIteration:
        return x
    raise ValueError("More than one element in iterable")


def groupby_shuffle(values, groups):
    # find unique groups and their indices
    unique_groups = np.unique(groups)

    # construct the shuffled values array
    shuffled_values = np.empty_like(values)
    for group in zip(unique_groups):
        shuffled_values[groups == group] = np.random.permutation(values[groups == group])

    return shuffled_values


class Compose:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x

    def __repr__(self):
        return "∘".join(f.__name__ for f in self.funcs[::-1])


# @torch.no_grad()
def torch_overlap(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))


def differentiable_safe_exp(x: torch.Tensor, normalise: bool = True) -> torch.Tensor:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - torch.max(x)
    x = torch.exp(x)
    if normalise:
        x = x / torch.sum(x)
    return x


def column_to(type_: type, cols: str | list[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def wrapper(df):
        df = df.copy()
        if isinstance(cols, str):
            cols_ = [cols]
        else:
            cols_ = cols
        for col in cols_:
            df[col] = df[col].astype(type_)
        return df

    return wrapper


def keep_serializable(dct: dict):
    return {k: v for k, v in dct.items() if isinstance(v, (int, float, str))}
