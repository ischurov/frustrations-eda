import json
import operator
import pickle
from functools import reduce
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Iterable

import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse.linalg
import sympy as sp
from loguru import logger
from sympy import Rational
from sympy.combinatorics import Permutation
from typing_extensions import Literal

from misc_utils import spin_inv
from spin_lattices import SpinLattice


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

    def get_ground_state_coeffs(
        self, states, apply_symmetries=True
    ) -> npt.NDArray[np.float64 | np.complex128]:
        if not apply_symmetries:
            return np.real_if_close(self.ground_state[self.basis.index(states)])

        corresp_reprs, characters, norms = self.state_info(states)
        corresp_repr_indices = self.basis.index(corresp_reprs)
        return np.real_if_close(
            self.ground_state[corresp_repr_indices] * characters * norms
        )

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

    def _find_cached_eigenstate(self, k) -> tuple[np.ndarray, np.ndarray] | None:
        if self.ground_state_cache is None:
            return None
        for eigenstate in range(k, 20):
            if cached_value := self.ground_state_cache.get_cache(
                additional_params={"up_to_eigenstate": eigenstate}
            ):
                logger.debug(
                    f"Using cached version of eigenvalues / eigenstates up to {eigenstate}."
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

    def state_info(
        self, states: npt.NDArray[np.uint64]
    ) -> tuple[
        npt.NDArray[np.uint64], npt.NDArray[np.complex128], npt.NDArray[np.float64]
    ]:
        states_unpacked = self.lattice.unpack_configurations(states)
        perm_group_size = len(self.basis.symmetries)
        orbit = np.empty((perm_group_size, len(states)), dtype=np.uint64)
        spin_inversion = self.basis.spin_inversion
        for i, (permutation, _) in enumerate(self.basis.symmetries):
            orbit[i] = self.lattice.pack_configurations(
                states_unpacked[:, (permutation).array_form]
            )
        if spin_inversion is not None:
            orbit = np.concatenate([orbit, spin_inv(orbit, self.lattice.number_spins)])

        fix_size = (orbit[0] == orbit).sum(axis=0)
        n_orb = orbit.shape[0] // fix_size

        reprs = orbit.min(axis=0)
        is_repr = orbit == reprs

        g_idxs = np.argmin(orbit, axis=0)
        all_characters = np.array(
            [
                sp.exp(2 * sp.pi * sp.I * self.basis.symmetries[idx][1]).evalf()
                for idx in range(perm_group_size)
            ],
            dtype=np.complex128,
        )

        if spin_inversion is not None:
            all_characters = np.concatenate(
                [all_characters, spin_inversion * all_characters]
            )

        all_characters = all_characters.reshape(-1, 1)

        characters_matrix = np.where(
            is_repr, np.broadcast_to(all_characters, is_repr.shape), np.nan
        )

        if spin_inversion is not None:
            spin_invs, g_idxs = np.divmod(g_idxs, len(self.basis.symmetries))

        characters = np.nanmean(characters_matrix, axis=0)
        if spin_inversion is not None:
            characters *= np.where(spin_invs, spin_inversion, 1)

        norms = 1 / np.sqrt(n_orb)

        return reprs, characters, norms

    def to_ground_state_sector(self) -> "SpinSystem":
        """
        Returns a new SpinSystem with the same lattice and expression,
        with basis taken from self.hamiltonian.expression.ground_state_sectors, whose
        ground state energy is the same as self.ground_state_energy

        If self.basis.hamming_weight or self.basis.spin_inversion is not None,
        only sectors with the same hamming_weight or spin_inversion are considered

        Returns
        -------
        SpinSystem
            SpinSystem with the same lattice and expression, but different basis

        Raises
        ------
        ValueError
            If ground state sector with the same energy is not found
        """
        smallest_energy = None
        ground_state_system = None
        for basis in self.hamiltonian.expression.ground_state_sectors():
            if (
                self.basis.hamming_weight is not None
                and basis.hamming_weight != self.basis.hamming_weight
            ):
                continue
            if (
                self.basis.spin_inversion is not None
                and basis.spin_inversion != self.basis.spin_inversion
            ):
                continue

            new_hamiltonian = ls.Operator(self.hamiltonian.expression, basis)
            new_system = SpinSystem(
                lattice=self.lattice,
                hamiltonian=new_hamiltonian,
                ground_state_cache_dir=(
                    self.ground_state_cache.cache_dir
                    if self.ground_state_cache
                    else None
                ),
            )
            if smallest_energy is None or new_system.ground_energy < smallest_energy:
                smallest_energy = new_system.ground_energy
                ground_state_system = new_system

        if ground_state_system is None:
            raise ValueError("No sectors to consider")
        return ground_state_system


class LatticeExpr:
    def __init__(
        self,
        lattice: SpinLattice,
        edge_str: str,
        edge_params: dict[int, float],
        node_str: str | None = None,
    ):
        self.lattice = lattice
        self.edge_str = edge_str
        self.edge_params = edge_params

        if node_str is not None:
            node_terms = [ls.Expr(node_str, sites=[[x] for x in lattice.sites])]
        else:
            node_terms = []

        edge_terms = [
            j_value * ls.Expr(edge_str, sites=lattice.kind_to_edges[i])
            for i, j_value in edge_params.items()
        ]
        self.expr = reduce(operator.add, edge_terms + node_terms)

        # self.expr: ls.Expr = reduce(
        #     operator.add,
        #     (
        #         j_value * ls.Expr(edge_str, sites=lattice.kind_to_edges[i])
        #         for i, j_value in edge_params.items()
        #     ),
        # )


heisenberg_str = "2 (σ⁺₀ σ⁻₁ + σ⁺₁ σ⁻₀) + σᶻ₀ σᶻ₁"


def heisenberg(
    lattice: SpinLattice, J1: float = 1.0, J2: float | None = 0.0
) -> LatticeExpr:
    """
    Two-parameter Heisenberg Hamiltonian
    """
    return LatticeExpr(
        lattice,
        edge_str=heisenberg_str,
        edge_params={1: J1, 2: J2} if J2 is not None else {1: J1},
    )


def heisenberg_transversal_field(
    lattice: SpinLattice, J1: float = 1.0, J2: float = 0.0, h: float = 1.0
) -> LatticeExpr:
    """
    Heisenberg Hamiltonian with transversal field
    """
    return LatticeExpr(
        lattice,
        edge_str=heisenberg_str,
        edge_params={1: J1, 2: J2} if J2 is not None else {1: J1},
        node_str=f"{h} σˣ₀" if h != 0 else None,
    )


def spin_system(
    expr: LatticeExpr,
    basis: Callable[[LatticeExpr], ls.Basis],
    ground_state_cache_dir: Path | None | Literal[False] = None,
) -> SpinSystem:
    """
    Create a spin system based on the given lattice expression and basis.

    Args:
        expr (LatticeExpr): The lattice expression representing the spin system.
        basis (Callable[[LatticeExpr], ls.Basis]): A function that returns the basis for the spin system.
        ground_state_cache_dir (Path | None | Literal[False], optional): The directory to cache ground states.
            If set to False, caching is disabled. If not provided, a default directory "groundstates_cache"
            will be used.

    Returns:
        SpinSystem: The created spin system.

    """
    if ground_state_cache_dir == False:
        ground_state_cache_dir = None
    elif ground_state_cache_dir is None:
        ground_state_cache_dir = Path("groundstates_cache")
    lattice = expr.lattice
    hamiltonian = ls.Operator(expr.expr, basis(expr))
    return SpinSystem(lattice, hamiltonian, ground_state_cache_dir)


def basis_factory(
    symmetries_factory: Callable[[LatticeExpr], list[tuple[Permutation, Rational]]],
    hamming_weight: int | None | Literal["half"] = "half",
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
    hamming_weight: int | None | Literal["half"] = "half",
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
    hamming_weight: int | Literal["half"] | None = "half",
    spin_inversion: int | None = None,
) -> Callable[[LatticeExpr], ls.Basis]:
    """
    Creates a basis without permuation symmetries
    (but possibly with hamming_weight and spin_inversion)
    """
    return basis_factory(
        lambda expr: [], hamming_weight=hamming_weight, spin_inversion=spin_inversion
    )


def ground_state_basis(
    hamming_weight: int | Literal["half"] | None = "half",
    spin_inversion: int | None = None,
    ground_state_cache_dir: Path | None | Literal[False] = None,
) -> Callable[[LatticeExpr], ls.Basis]:
    """
    Creates a basis with symmetries that contains the ground state

    Parameters
    ----------
    hamming_weight : int | Literal["half"] | None
        Hamming weight

    spin_inversion : int | None
        Spin inversion

    ground_state_cache_dir : Path | None
        Directory where the ground state of the temporary system is stored
        If None, default directory is used (groundstates_cache)
        If False, no cache is used
    """

    def wrapper(expr: LatticeExpr) -> ls.Basis:
        system = spin_system(
            expr,
            no_symmetries_basis(
                hamming_weight=hamming_weight, spin_inversion=spin_inversion
            ),
            ground_state_cache_dir=ground_state_cache_dir,
        )
        return system.to_ground_state_sector().basis

    return wrapper
