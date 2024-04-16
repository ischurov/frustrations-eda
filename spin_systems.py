import pickle
from hashlib import md5
from pathlib import Path
from typing import Any

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse.linalg
from loguru import logger
from typing_extensions import Literal
from functools import reduce
import operator
from typing import Callable, Iterable

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

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.default_params = default_params
        self.params = params

    def get_minimal_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in sorted(params.items())
            if k not in self.default_params or v != self.default_params[k]
        }

    def get_cache_id(self, additional_params: dict[str, Any]) -> str:
        return md5(
            json.dumps(self.get_minimal_params(self.params | additional_params)).encode(
                "utf8"
            )
        ).hexdigest()

    def get_cache_path(
        self, additional_params: dict[str, Any], suffix: str = ""
    ) -> Path:
        return self.cache_dir / (self.get_cache_id(additional_params) + suffix)

    def get_cache(self, additional_params: dict[str, Any], suffix: str = "") -> Any:
        cache_path = self.get_cache_path(additional_params, suffix + ".pickle")
        if cache_path.exists():
            cached_params = json.loads(
                self.get_cache_path(additional_params, ".params.json").read_text()
            )
            if self.get_minimal_params(cached_params) != json.loads(
                json.dumps(self.get_minimal_params(self.params | additional_params))
            ):
                logger.debug("Caching error")
                logger.debug(f"{cached_params=}")
                logger.debug(f"{self.params | additional_params=}")
                logger.debug(f"{self.get_minimal_params(cached_params)=}")
                logger.debug(
                    f"{self.get_minimal_params(self.params | additional_params)=}"
                )
                raise ValueError("Caching error")

            return pickle.loads(cache_path.read_bytes())
        return None

    def set_cache(
        self, additional_params: dict[str, Any], value: Any, suffix: str = ""
    ) -> None:
        cache_path = self.get_cache_path(additional_params, suffix + ".pickle")
        cache_path.write_bytes(pickle.dumps(value))
        self.get_cache_path(additional_params, ".params.json").write_text(
            json.dumps(self.params | additional_params)
        )


class SpinSystem:
    def __init__(
        self,
        lattice: SpinLattice,
        hamiltonian: ls.Operator,
        ground_state_cache_dir: Path | None = None,
    ):
        """
        User-friendly wrapper over ls.Operator that provides methods to
        calculate eigenstates and eigenvalues with proper caching.

        It is usually better to use `spin_system` function to construct SpinSystem
        """
        self.lattice = lattice
        self.hamiltonian = hamiltonian
        hamiltonian.basis.build()
        self.number_spins = lattice.number_spins
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

    @property
    def basis(self) -> ls.SpinBasis:
        return self.hamiltonian.basis  # type: ignore

    def get_cache_params(self) -> dict[str, Any]:
        return {
            "basis.number_sites": self.basis.number_sites,
            "basis.hamming_weight": self.basis.hamming_weight,
            "basis.spin_inversion": self.basis.spin_inversion,
            "basis.symmetries": sorted(
                [
                    (symmetry.list(), (character.numerator, character.denominator))
                    for symmetry, character in self.basis.symmetries
                ]
            ),
            "hamiltonian.expression": str(self.hamiltonian.expression),
        }

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
        return self.lattice.unpack_configurations(self.basis.states)

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


class LatticeExpr:
    def __init__(self, lattice: SpinLattice, expr_str: str, params: dict[int, float]):
        self.lattice = lattice
        self.expr_str = expr_str
        self.params = params
        self.expr: ls.Expr = reduce(
            operator.add,
            (
                j_value * ls.Expr(expr_str, sites=lattice.kind_to_edges[i])
                for i, j_value in params.items()
            ),
        )


def heisenberg(lattice: SpinLattice, J1: float = 1.0, J2: float = 0.0) -> LatticeExpr:
    """
    Two-parametric Heisenberg Hamiltonian
    """
    return LatticeExpr(
        lattice,
        expr_str="2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + σᶻ₀ σᶻ₁",
        params={1: J1, 2: J2},
    )


def spin_system(
    expr: LatticeExpr,
    basis: Callable[[LatticeExpr], ls.Basis],
    ground_state_cache_dir: Path | None = Path("groundstates_cache"),
) -> SpinSystem:
    lattice = expr.lattice
    hamiltonian = ls.Operator(expr.expr, basis(expr))
    return SpinSystem(lattice, hamiltonian, ground_state_cache_dir)


def basis_factory(
    symmetries_factory: Callable[[LatticeExpr], list[tuple[Permutation, Rational]]],
    hamming_weight: int | Literal["half"] = "half",
    spin_inversion: int | None = None,
) -> Callable[[LatticeExpr], ls.Basis]:
    """
    Wraps symmetries_factory into a function that returns basis
    with given symmetries, hamming_weight and spin_inversion

    See `zero_sector_basis` and `no_symmetries_basis` for examples
    """

    def wrapper(expr: LatticeExpr) -> ls.Basis:
        return ls.SpinBasis(
            number_spins=(expr.lattice.number_spins),
            hamming_weight=(
                (expr.lattice.number_spins // 2)
                if hamming_weight == "half"
                else hamming_weight
            ),
            spin_inversion=spin_inversion,
            symmetries=symmetries_factory(expr),
        )

    return wrapper


def zero_sector_basis(
    hamming_weight: int | Literal["half"] = "half",
    spin_inversion: int | None = None,
    get_permutations: Callable[
        [LatticeExpr], Iterable[Permutation]
    ] = lambda expr: expr.expr.permutation_group(),  # type: ignore
) -> Callable[[LatticeExpr], ls.Basis]:
    """
    Creates basis with all zero sectors
    """

    return basis_factory(
        lambda expr: [
            (permutation, Rational(0)) for permutation in get_permutations(expr)
        ],
        hamming_weight=hamming_weight,
        spin_inversion=spin_inversion,
    )


def no_symmetries_basis(
    hamming_weight: int | Literal["half"] = "half",
    spin_inversion: int | None = None,
) -> Callable[[LatticeExpr], ls.Basis]:
    """
    Creates a basis without permuation symmetries
    (but possibly with hamming_weight and spin_inversion)
    """
    return basis_factory(
        lambda expr: [], hamming_weight=hamming_weight, spin_inversion=spin_inversion
    )
