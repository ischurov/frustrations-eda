from collections import namedtuple
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, overload

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import Tensor

# This code is derived from:
# https://github.com/twesterhout/nqs-playground/blob/conda/nqs_playground/sampling.py
# and
# https://github.com/twesterhout/nqs-playground/blob/conda/nqs_playground/sgd.py
# And modified by Ilya Schurov (Ilia Shchurov) in 2023.

# Original license follows:

# Copyright Tom Westerhout (c) 2020-2021
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

_SamplingOptionsBase = namedtuple(
    "_SamplingOptions",
    [
        "number_samples",
        "number_chains",
        "number_discarded",
        "sweep_size",
        "mode",
        "device",
        "other",
    ],
)


class SamplingOptions(_SamplingOptionsBase):
    r"""Options for sampling spin configurations."""

    def __new__(
        cls,
        number_samples: int,
        number_chains: int = 1,
        number_discarded: Optional[int] = None,
        sweep_size: Optional[int] = None,
        mode: Optional[str] = None,
        device: Union[str, torch.device, None] = None,
        other: Optional[Dict[str, Any]] = None,
    ):
        r"""Create SamplingOptions.

        Parameters
        ----------
        number_samples: int
            Number of samples per Markov chain. Must be a positive integer.
            'full' sampler will ignore this parameter.
        number_chains: int, optional
            Number of independent Markov chains. Must be a positive integer.
            This parameter only makes sense for MCMC samplers such as
            Metropolis-Hastings algorithm or Zanella process. Exact samplers
            ('exact' and 'autoregressive') will just multiply `number_samples`
            by `number_chains`.
        number_discarded: int, optional
            Number of samples to discard at the beginning of each Markov chain
            (i.e. how long the thermalization procedure should be). If
            specified, must be a positive integer. Otherwise, 10% of
            `number_samples` will be used. This parameter only makes sense for
            MCMC samplers (i.e. 'exact', 'autoregressive', and 'full' samplers
            will ignore this argument).
        sweep_size: int, optional
            Sweep size, i.e. how many Markov chain steps are made until the
            next sample is saved. `sweep_size = 1` means that every sample is
            saved. `sweep_size = 5` means that per every 5 steps of the MCMC
            process we only store one sample. If not specified, the default
            value of `1` will be used. This parameter only makes sense for MCMC
            samplers (i.e. 'exact', 'autoregressive', and 'full' samplers will
            ignore this argument).
        mode: str, optional
            Which algorithm to use for sampling. Valid choices are:

              * `metropolis` -- use Metropolis-Hastings algorithm with 1- or
                2-spin flips.
              * `zanella` -- use Zanella algorithm with 2-spin flips.
              * `exact` -- exactly sample from the discrete probability
                distribution using `torch.multinomial` or
                `numpy.random.choice`. This algorithm works for small systems
                only.
              * `full` -- skip sampling altogether and just return the full
                Hilbert space basis. This algorithm works for small systems
                only.
              * `autoregressive` -- assume that the probability distribution
                has a custom `sample` method and use it.
        device: str or torch.device
            On which device to run the sampling.
        other: Dict[str, Any]
            Extra arguments for a specific sampler.
        """
        number_samples = int(number_samples)
        if number_samples <= 0:
            raise ValueError("negative number_samples: {}".format(number_samples))
        number_chains = int(number_chains)
        if number_chains <= 0:
            raise ValueError("negative number_chains: {}".format(number_chains))

        if number_discarded is not None:
            number_discarded = int(number_discarded)
            if number_discarded < 0:
                raise ValueError(
                    "invalid number_discarded: {}; expected either a non-negative "
                    "integer or None".format(number_chains)
                )
        else:
            logger.info(
                "`number_discarded` not specified when constructing SamplingOptions, "
                "1/10 of `number_samples` will be used."
            )
            number_discarded = number_samples // 10
        if sweep_size is not None:
            sweep_size = int(sweep_size)
            if sweep_size <= 0:
                raise ValueError("negative sweep_size: {}".format(sweep_size))
        else:
            sweep_size = 1
            logger.warning(
                "`sweep_size` not specified when constructing SamplingOptions, "
                "`sweep_size` will be set to 1. Make sure this is what you want!"
            )
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)
        if other is None:
            other = dict()
        return super(SamplingOptions, cls).__new__(
            cls,
            number_samples,
            number_chains,
            number_discarded,
            sweep_size,
            mode,
            device,
            other,
        )

    def hparams(self) -> Dict[str, Any]:
        p = {
            "number_samples": self.number_samples,
            "number_chains": self.number_chains,
            "mode": self.mode,
        }
        if "mode" in ["zanella", "metropolis"]:
            p["sweep_size"] = self.sweep_size
        return p


def _determine_batch_size(options: SamplingOptions) -> int:
    batch_size = options.other.get("batch_size")
    if batch_size is None:
        batch_size = 8192
        logger.debug("'batch_size' not specified, will use the default value of 8192.")
    else:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(
                "invalid 'batch_size': {}; expected a positive integer".format(batch_size)
            )
    return batch_size


def split_into_batches(
    xs: Tensor | npt.NDArray | tuple[Tensor | npt.NDArray, ...] | list[Tensor | npt.NDArray],
    batch_size: int,
    device=None,
):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))

    expanded = False
    if isinstance(xs, (np.ndarray, Tensor)):
        xs = (xs,)
        expanded = True
    else:
        assert isinstance(xs, (tuple, list))
    n = xs[0].shape[0]
    if any(filter(lambda x: x.shape[0] != n, xs)):
        raise ValueError("tensors 'xs' must all have the same batch dimension")
    if n == 0:
        return None

    i = 0
    while i + batch_size <= n:
        chunks = tuple(x[i : i + batch_size] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks
        i += batch_size
    if i != n:  # Remaining part
        chunks = tuple(x[i:] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks


@overload
def forward_with_batches(
    f: Callable[[Tensor], Tensor],
    xs: Tensor,
    batch_size: int,
    device=None,
) -> Tensor:
    ...


@overload
def forward_with_batches(
    f: Callable[[npt.NDArray], npt.NDArray],
    xs: npt.NDArray,
    batch_size: int,
    device=None,
) -> npt.NDArray:
    ...


def forward_with_batches(
    f: Callable[[Tensor], Tensor] | Callable[[npt.NDArray], npt.NDArray],
    xs: Tensor | npt.NDArray,
    batch_size: int,
    device=None,
) -> Tensor | npt.NDArray:
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
    samples at a time. ``xs`` is split into batches along the first dimension
    (i.e. dim=0). ``f`` must return a torch.Tensor or npt.NDArray.
    """
    if xs.shape[0] == 0:
        raise ValueError("invalid xs: {}; input should not be empty".format(xs))
    out = []
    for chunk in split_into_batches(xs, batch_size, device):
        out.append(f(chunk))
    if isinstance(out[0], Tensor):
        return torch.cat(out, dim=0)
    elif isinstance(out[0], np.ndarray):
        return np.concatenate(out, axis=0)
    else:
        raise TypeError("f must return either torch.Tensor or numpy.ndarray")


def _check_log_prob_shape(log_prob: Tensor, device: Optional[torch.device]) -> None:
    if log_prob.dim() != 1:
        raise ValueError(
            "log_prob_fn should return the logarithm of the probability, "
            "but output tensor has dimension {}; did you by accident use "
            "sign instead of amplitude network?"
            "".format(log_prob.dim())
        )
    if device is not None and log_prob.device != device:
        raise ValueError(
            "log_prob_fn should return tensors residing on {}; received "
            "tensors residing on {} instead; make sure options.device matches "
            "the location of log_prob_fn".format(device, log_prob.device)
        )


@torch.jit.script
def safe_exp(x: Tensor, normalise: bool = True) -> Tensor:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - torch.max(x)
    torch.exp_(x)
    if normalise:
        x /= torch.sum(x)
    return x


@torch.no_grad()
def sample_full(
    log_prob_fn: Callable[[Tensor], Tensor],
    basis: ls.SpinBasis,
    options: SamplingOptions,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    r"""Instead of sampling, take all basis vectors in the Hilbert space."""
    batch_size = _determine_batch_size(options)
    device = options.device
    states = torch.from_numpy(basis.states.view(np.int64))
    if device is not None:
        states = states.to(device)
    logger.debug(
        "Applying 'log_prob_fn' to all basis vectors in the Hilbert space using batch_size={}..."
        "".format(batch_size)
    )
    # states = pad_states(states)
    log_prob = forward_with_batches(log_prob_fn, states, batch_size=batch_size, device=device)
    if log_prob.dim() > 1:
        log_prob.squeeze_(dim=1)
    _check_log_prob_shape(log_prob, device)
    logger.debug("Computing weights...")
    log_prob = log_prob.unsqueeze_(dim=1)
    weights = safe_exp(log_prob, normalise=True)
    states = states.unsqueeze_(dim=1)
    return states, log_prob, {"weights": weights}


@overload
def sample_exactly(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    basis: ls.SpinBasis,
    options: SamplingOptions,
    return_all_probs: Literal[False],
) -> tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def sample_exactly(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    basis: ls.SpinBasis,
    options: SamplingOptions,
    return_all_probs: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


@torch.no_grad()
def sample_exactly(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    basis: ls.SpinBasis,
    options: SamplingOptions,
    return_all_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Sample states by explicitly constructing the discrete probability distribution.

    Number of samples is `options.number_chains * options.number_samples`, and
    `options.number_discarded` and `options.sweep_size` are ignored, since
    samples are already i.i.d.
    """
    states, log_prob, _extra = sample_full(log_prob_fn, basis, options)
    states = states.squeeze_(dim=1)
    log_prob = log_prob.squeeze_(dim=1)
    prob = _extra["weights"].squeeze_(dim=1)
    device = options.device
    number_samples = options.number_chains * options.number_samples
    if len(prob) < (1 << 24):
        logger.debug("Using torch.multinomial to sample indices...")
        # PyTorch only supports discrete probability distributions
        # shorter than 2²⁴.
        # NOTE: replacement=True is IMPORTANT because it more closely
        # emulates the actual Monte Carlo behaviour
        indices = torch.multinomial(prob, num_samples=number_samples, replacement=True)
    else:
        logger.debug("Using numpy.random.choice to sample indices...")
        # If we have more than 2²⁴ different probabilities chances are,
        # NumPy will complain about probabilities not being normalised
        # since float32 precision is not enough. The simplest
        # workaround is to convert the probabilities to float64 and
        # then renormalise which is what we do.
        prob = prob.to(device="cpu", dtype=torch.float64)
        prob /= torch.sum(prob)
        indices = np.random.choice(len(prob), size=number_samples, replace=True, p=prob)
        indices = torch.from_numpy(indices).to(device)

    # Choose the samples
    log_prob = log_prob[indices]
    states = states[indices]
    shape = (options.number_samples, options.number_chains)
    if return_all_probs:
        prob /= torch.sum(prob)
        return states.view(*shape), log_prob.view(*shape), prob
    return states.view(*shape), log_prob.view(*shape)
