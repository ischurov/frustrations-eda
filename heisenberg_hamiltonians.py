import pickle
from hashlib import md5
from pathlib import Path
from typing import Optional, Any

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import scipy.sparse.linalg
from loguru import logger
from typing_extensions import Literal

from misc_utils import (
    batched_state_info_df,
    make_packed_configurations,
    make_unpacked_configurations,
)
from spin_lattices import SpinLattice
from sympy.combinatorics import Permutation
from sympy import Rational
import json


class DictKeyedCache:
    def __init__(
        self,
        cache_dir: Path,
        default_params: dict[str, Any] = {},
        params: dict[str, Any] = {},
    ):
        """
        Implements cache whose keys are dictionaries. The cache is stored in a directory
        with a file for each set of parameters. The cache is stored as a pickle file.

        Parameters
        ----------
        cache_dir : Path
            Directory where the cache is stored

        default_params : dict
            Default parameters. If new parameter is added, its value that were
            actual before it was introduced should be added to this dictionary

        params : dict
            Parameters to be used to generate the cache key
        """
        if not default_params:
            default_params = {}
        if not params:
            params = {}

        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = cache_dir
        self.default_params = default_params
        self.params = params

    def get_minimal_params(self, additional_params: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in sorted((self.params | additional_params).items())
            if v != self.default_params.get(k)
        }

    def get_cache_id(self, additional_params: dict[str, Any]) -> str:
        return md5(
            json.dumps(self.get_minimal_params(additional_params)).encode("utf8")
        ).hexdigest()

    def get_cache_path(
        self, additional_params: dict[str, Any], suffix: str = ""
    ) -> Path:
        return self.cache_dir / (self.get_cache_id(additional_params) + suffix)

    def get_cache(self, additional_params: dict[str, Any], suffix: str = "") -> Any:
        cache_path = self.get_cache_path(additional_params, suffix + ".pickle")
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes())
        return None

    def set_cache(
        self, additional_params: dict[str, Any], value: Any, suffix: str = ""
    ) -> None:
        cache_path = self.get_cache_path(additional_params, suffix + ".pickle")
        cache_path.write_bytes(pickle.dumps(value))
        self.get_cache_path(additional_params, ".params.repr").write_text(
            repr(self.params | additional_params)
        )


class SpinSystem:
    def __init__(
        self,
        lattice: SpinLattice,
        basis: ls.SpinBasis,
        hamiltonian: ls.Operator,
        ground_state_cache_dir: Path | None = None,
    ):
        self.lattice = lattice
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.number_spins = len(lattice.sites)
        self.ground_state_cache = (
            DictKeyedCache(
                cache_dir=ground_state_cache_dir,
                default_params=self.get_default_cache_params(),
                params=self.get_cache_params(),
            )
            if ground_state_cache_dir is not None
            else None
        )

        self.eigenstates = None
        self.eigenvalues = None

    def get_cache_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_default_cache_params(self) -> dict[str, Any]:
        return {}

    @property
    def ground_state(self) -> npt.NDArray[np.float64]:
        if self.eigenstates is None:
            self.get_eigenstates(1)
        assert self.eigenstates is not None
        return self.eigenstates[:, 0]

    @property
    def ground_energy(self) -> float:
        if self.eigenvalues is None:
            self.get_eigenstates(1)
        assert self.eigenvalues is not None
        return self.eigenvalues[0]

    def unpack_configurations(self):
        """
        Unpacks all configurations in the basis into an np.array
        of one-dimensional np.arrays with 0's and 1's
        """
        return make_unpacked_configurations(self.basis.states, self.number_spins)

    def _find_cached_eigenstate(self, k) -> tuple[np.ndarray, np.ndarray] | None:
        if self.ground_state_cache is None:
            return None
        for eigenstate in range(k, 20):
            if cached_value := self.ground_state_cache.get_cache(
                additional_params={"up_to_eigenstate": eigenstate}
            ):
                logger.debug(
                    f"Using cached version of eigenvalues / eigenstates up to {eigenstate}, "
                    f"cache params = {self.ground_state_cache.params}"
                )
                eigenvalues, eigenstates = cached_value
                return eigenvalues, eigenstates

    def get_eigenstates(self, k=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Records ground_energy and ground_state into to self.ground_energy and
        self.ground_state (only one value, the smallest energy),returns k
        smallest eigenvalue / eigenvectors and records them into
        self.eigenvalues / self.eigenstates

        Returns
        -------
        eigenvalues : np.array
            k smallest eigenvalues

        eigenstates : np.array
            k eigenvectors (as vector-columns) corresponding to the k smalles
            eigenvalues

        """
        if (
            self.eigenstates is not None
            and self.eigenvalues is not None
            and self.eigenstates.shape[1] >= k
        ):
            return self.eigenvalues, self.eigenstates
        if (cached_eigenstate := self._find_cached_eigenstate(k)) is not None:
            self.eigenvalues, self.eigenstates = cached_eigenstate
            return self.eigenvalues, self.eigenstates

        # Diagonalize the Hamiltonian using ARPACK

        logger.debug("Calculating eigenvalues / eigenstates")
        eigenvalues, eigenstates = scipy.sparse.linalg.eigsh(
            self.hamiltonian, k=k, which="SA"
        )

        if self.ground_state_cache is not None:
            self.ground_state_cache.set_cache(
                additional_params={"up_to_eigenstate": k},
                value=(eigenvalues, eigenstates),
            )

        logger.debug("Ground state energy is {:.10f}".format(eigenvalues[0]))

        self.eigenvalues = eigenvalues
        self.eigenstates = eigenstates

        return eigenvalues, eigenstates

    def make_unpacked_configurations(
        self, states: npt.NDArray[np.uint64]
    ) -> npt.NDArray:
        return make_unpacked_configurations(states, self.number_spins)

    def make_packed_configurations(
        self, unpacked_configurations: npt.NDArray
    ) -> npt.NDArray[np.uint64]:
        return make_packed_configurations(unpacked_configurations, self.number_spins)


heisenberg_expr = "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + σᶻ₀ σᶻ₁"


def heisenberg_expr_rot(phi):
    C = f"({np.sin(phi)} σˣ₀ + {np.cos(phi)} σᶻ₀)({np.sin(phi)} σˣ₁ + {np.cos(phi)} σᶻ₁)"
    return f"2 {C}(σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) {C} + {C} σᶻ₀ σᶻ₁{C}"


class HeisenbergJ1J2(SpinSystem):
    def __init__(
        self,
        lattice: SpinLattice,
        basis: ls.SpinBasis,
        J1: float = 1.0,
        J2: float = 1.0,
        ground_state_cache_dir: Path | None = Path("groundstates_cache"),
        expr_str=heisenberg_expr,
    ):
        self.expr_str = expr_str

        J1 = float(J1)
        J2 = float(J2)
        self.J1 = J1
        self.J2 = J2

        # Constructing the Hamiltonian

        # fmt: off
        expr = (J1 * ls.Expr(expr_str, sites=lattice.kind_to_edges[1]) + 
                J2 * ls.Expr(expr_str, sites=lattice.kind_to_edges[2]))
        # fmt: on

        hamiltonian = ls.Operator(expr, basis)

        super().__init__(
            lattice=lattice,
            basis=basis,
            hamiltonian=hamiltonian,
            ground_state_cache_dir=ground_state_cache_dir,
        )

    def get_cache_params(self) -> dict[str, Any]:
        return {
            "basis.number_sites": self.basis.number_sites,
            "basis.hamming_weight": self.basis.hamming_weight,
            "basis.spin_inversion": self.basis.spin_inversion,
            "basis.symmetries": [
                (symmetry.list(), (character.numerator, character.denominator))
                for symmetry, character in self.basis.symmetries
            ],
            "hamiltonian.expression": str(self.hamiltonian.expression),
        }
