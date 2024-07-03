# Authors: Tom Westerhout, Nikita Astrakhantsev

import math
import os
import pickle
import sys

import numba
import numpy as np
import scipy.special
from numba import float32, float64, uint64


def index_to_spin(index, number_spins):
    return (
        (
            index.reshape(-1, 1).astype(np.int64)
            & (1 << np.arange(number_spins).astype(np.int64))
        )
    ) > 0


def spin_to_index(spin, number_spins):
    a = 2 ** np.arange(number_spins, dtype=np.int64)
    return spin.dot(a)


@numba.jit("uint64(uint64, uint64)", nogil=True, nopython=True)
def _binom(n, k):
    r"""Compute the number of ways to choose k elements out of a pile of n.

    :param n: the size of the pile of elements
    :param k: the number of elements to take from the pile
    :return: the number of ways to choose k elements out of a pile of n
    """
    assert 0 <= n and n < 40
    assert 0 <= k and k <= n

    if k == 0 or k == n:
        return 1
    total_ways = 1
    for i in range(min(k, n - k)):
        total_ways = total_ways * (n - i) // (i + 1)
    return total_ways


@numba.jit("uint64[:, :](uint64, uint64)", nogil=True, nopython=True)
def make_binom_cache(max_n, max_k):
    assert 0 < max_k and max_k <= max_n
    assert max_n < 40

    cache = np.zeros((max_n + 1, max_k + 2), dtype=np.uint64)
    for i in range(max_n + 1):
        for j in range(min(i, max_k) + 2):
            cache[i, j] = _binom(i, j) if j <= i else 0
    return cache


@numba.jit("uint64(uint64)", nogil=True, nopython=True)
def _hamming_weight(n: int):
    # See https://stackoverflow.com/a/9830282
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)
    return n


@numba.jit("uint64[:](uint64, uint64)", nogil=True, nopython=True)
def generate_binaries(length: int, hamming: int):
    assert length > 0
    assert hamming >= 0 and hamming <= length
    if hamming == 0:
        return np.zeros(1, dtype=np.uint64)

    size = _binom(length, hamming)
    set_of_vectors = np.empty(size, dtype=np.uint64)
    val = (1 << hamming) - 1
    for i in range(size):
        set_of_vectors[i] = val
        c = val & -val
        r = val + c
        val = (((r ^ val) >> 2) // c) | r
    return set_of_vectors


@numba.jit("uint64(uint64, uint64, uint64)", nogil=True, nopython=True)
def _merge(vec1, vec2, dim2):
    return (vec1 << dim2) | vec2


@numba.jit("uint64(uint64, uint64, uint64[:, :])", nogil=True, nopython=True)
def _get_index(x, dim, binom_cache):
    n = 0
    h = 0
    if x & 1 == 1:
        h += 1
    x >>= 1
    for i in range(1, dim):
        if x & 1 == 1:
            h += 1
            n += binom_cache[i, h]
        x >>= 1
    return n


@numba.jit(
    "complex128(uint64, uint64, uint64, uint64, uint64, complex128[::1], uint64[:, :])",
    nogil=True,
    nopython=True,
)
def _density_matrix_element(dim1, dim, hamming, vec1, vec2, amplitudes, binom_cache):
    hamming1 = _hamming_weight(vec1)
    smallest_number = 2 ** (hamming - hamming1) - 1
    k = dim - dim1
    index1 = _get_index(_merge(vec1, smallest_number, k), dim, binom_cache)
    index2 = _get_index(_merge(vec2, smallest_number, k), dim, binom_cache)
    size_of_traced = _binom(k, hamming - hamming1)
    matrix_element = np.dot(
        amplitudes[index1 : index1 + size_of_traced].conj(),
        amplitudes[index2 : index2 + size_of_traced],
    )

    return matrix_element


@numba.jit(
    "complex128[:, :](uint64, uint64, uint64, uint64, complex128[::1])",
    nogil=True,
    nopython=True,
    parallel=False,
)
def sector_density_matrix(sector_dim, dim, sector_hamming, hamming, amplitudes):
    assert sector_hamming <= hamming and sector_dim <= dim
    assert hamming - sector_hamming <= dim - sector_dim

    sector_basis = generate_binaries(sector_dim, sector_hamming)
    binom_cache = make_binom_cache(dim, hamming)
    n = len(sector_basis)
    matrix = np.empty((n, n), dtype=np.complex128)
    for i in range(len(sector_basis)):
        for j in range(i, len(sector_basis)):
            matrix[i, j] = _density_matrix_element(
                sector_dim,
                dim,
                hamming,
                sector_basis[i],
                sector_basis[j],
                amplitudes,
                binom_cache,
            )
            matrix[j, i] = np.conj(matrix[i, j])

    return matrix


def density_matrix(sub_dim, dim, hamming, amplitudes):
    return [
        sector_density_matrix(sub_dim, dim, sub_hamming, hamming, amplitudes)
        for sub_hamming in range(
            max(0, hamming - (dim - sub_dim)), min(hamming, sub_dim) + 1
        )
    ]


if __name__ == "__main__":
    dim = int(sys.argv[1])
    nph = int(sys.argv[2])
    hamming = nph

    entropies = []
    t = np.linspace(0.005, 1.0, 200)[int(sys.argv[3])]

    sizes = {
        6: (2, 3),
        8: (2, 4),
        9: (3, 3),
        10: (2, 5),
        12: (3, 4),
        15: (3, 5),
        16: (4, 4),
        18: (3, 6),
        20: (4, 5),
        24: (4, 6),
        25: (5, 5),
        28: (4, 7),
        30: (5, 6),
        35: (5, 7),
    }
    Ly, Lx = sizes[dim]

    try:
        rearr = np.load(
            "/home/nktastr_google_com/XEB/wfs_for_entropy/map_{:d}_{:d}.npy".format(
                dim, nph
            )
        )
    except:
        rearr = []

        batch_size = 2**24
        basis = np.load(
            "/home/nktastr_google_com/XEB/wfs_for_entropy/basis_{:d}_{:d}.npy".format(
                dim, nph
            )
        )

        sites = np.arange(dim)
        new_sites = []
        for x in range(Lx):
            for y in range(Ly):
                new_sites.append(x + y * Lx)

        permutation = np.array(new_sites)

        for i in range(len(basis) // batch_size + 1):
            print("building basis", i, "of", len(basis) // batch_size + 1, flush=True)
            states = basis[i * batch_size : i * batch_size + batch_size]
            spins = index_to_spin(states, number_spins=dim)

            spins = spins[:, permutation]

            idxs = spin_to_index(spins, number_spins=dim)

            rearr.append(np.searchsorted(basis, idxs))
        rearr = np.concatenate(rearr, axis=0)
        np.save(
            "/home/nktastr_google_com/XEB/wfs_for_entropy/map_{:d}_{:d}.npy".format(
                dim, nph
            ),
            rearr,
        )

    state = np.array(
        np.load(
            "/home/nktastr_google_com/XEB/wfs_for_entropy/wf_PINK_0204_XXb_0.000_XXe_0.000_XIXl_0.000_XIXp_-0.094_XIXm_0.000_ZZ_-0.346_XZX_-0.013_XXZ_-0.013_ZZZ_0.000_XZXZp_0.000_XZXZm_0.000_XZZX_0.000_Nq_{:d}_Nph_{:d}_disorder_0_time_{:.3f}.npy".format(
                dim, nph, t
            )
        ),
        order="C",
    )

    # state = state[rearr]
    state_new = state * 0
    state_new[rearr] = state

    rho = density_matrix(dim // 2, dim, hamming, state_new)
    entropy = 0

    for iloop in range(len(rho)):
        spectrum, _ = np.linalg.eigh(rho[iloop])
        np.save(
            "/home/nktastr_google_com/XEB/entropies_rearranged/spectrum_{:d}_{:d}_{:.3f}_{:d}.npy".format(
                dim, nph, t, iloop
            ),
            spectrum,
        )
        print(iloop, spectrum, flush=True)
        entropy += np.sum(-spectrum * np.log(spectrum + 1e-10))
        print(np.sum(-spectrum * np.log(spectrum + 1e-10)), flush=True)

    print(entropy)
    entropies.append(entropy)

    np.save(
        "/home/nktastr_google_com/XEB/entropies_rearranged/ent_{:d}_{:d}_{:.3f}.npy".format(
            dim, nph, t
        ),
        np.array(entropies),
    )
