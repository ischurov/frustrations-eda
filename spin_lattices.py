from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Union

import igraph as ig
import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from loguru import logger
from sympy.combinatorics import Permutation, PermutationGroup

from misc_utils import batched_state_info_df, make_unpacked_configurations
from parity import parity, popcount

# BASED ON: https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/


def scatter_plot(data: pd.DataFrame, x, y, color, alpha=None, ax=None, scatter_kws=None, dim=None):
    if scatter_kws is None:
        scatter_kws = {}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color_labels = sorted(data[color].unique())

    # List of colors in the color palettes
    rgb_values = sns.color_palette("Set2", len(color_labels))

    # Map continents to the colors
    color_map = dict(zip(color_labels, rgb_values))

    for key, group in data.groupby(color):
        if alpha:
            alphas = group[alpha]
        else:
            alphas = 1.0

        group.plot(
            ax=ax,
            kind="scatter",
            x=x,
            y=y,
            label=key,
            color=color_map[key],
            alpha=alphas,
            **scatter_kws,
        )

    return ax


# END BASED


@dataclass
class BasisData:
    reprs: npt.NDArray[np.uint64]
    bits_to_repr: npt.NDArray[np.uint64]
    bits_to_repr_index: npt.NDArray[np.uint64]
    bits_to_char: npt.NDArray[np.float64]


class SpinLattice:
    def __init__(self):
        self.lattice_basis: npt.NDArray[np.int_]
        self.edges: list[tuple[tuple[npt.NDArray, npt.NDArray], int]]  # (start, end), kind
        self.site_to_num: dict[tuple[float, float], int]  # (x, y) -> num
        self.fourier_repr: BasisData
        self.fourier_basis: ls.SpinBasis
        self.bases: dict[tuple[bool, int | None, int | None], ls.SpinBasis]
        self.state_info_dfs: dict[tuple[bool, int | None, int | None], pd.DataFrame]
        self.bases: dict[tuple[bool, int | None, int | None], ls.SpinBasis] = {}
        self.state_info_dfs: dict[tuple[bool, int | None, int | None], pd.DataFrame] = {}

    def __repr__(self):
        return f"<SpinLattice:{self.get_cache_id()}>"

    @abstractmethod
    def get_cache_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def sites_df(self):
        ...

    @property
    def edges_to_kind(self) -> dict[tuple[int, int], int]:
        edges_to_kind = {}
        for (start, end), kind in self.edges:
            edges_to_kind[(self.site_to_num[tuple(start)], self.site_to_num[tuple(end)])] = kind
        return edges_to_kind

    @property
    def sites(self) -> list[int]:
        """
        Returns a list of sites: [0, 1, 2, ..., n]
        """
        return sorted(set(self.site_to_num.values()))

    @property
    def kind_to_edges(self) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Returns a dictionary from kind (usually integer numbers 1, 2) to the list of the edges
        (two-tuples of points, each point is two-tuple of ints)
        """

        k_to_e = defaultdict(list)
        for edge, kind in self.edges_to_kind.items():
            k_to_e[kind].append(edge)
        return k_to_e

    def as_igraph(self) -> ig.Graph:
        edges, kinds = zip(*self.edges_to_kind.items())
        return ig.Graph(edges=edges, edge_attrs={"kind": kinds})

    def get_automorphisms(self) -> list[list[int]]:
        g = self.as_igraph()
        return g.get_automorphisms_vf2(edge_color=g.es["kind"])

    def make_fourier_basis(self):
        if hasattr(self, "fourier_basis"):
            logger.debug("Using cached fourier_basis")
            return self.fourier_basis

        logger.debug("Cached fourier_basis not found, building...")

        symmetries = ls.Symmetries(
            [ls.Symmetry(automorphism, sector=0) for automorphism in self.get_automorphisms()]
        )

        number_spins = self.number_spins

        self.fourier_basis = ls.SpinBasis(
            symmetries=symmetries,
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        self.fourier_basis.build()

        return self.fourier_basis

    def get_fourier_basis_data(self) -> BasisData:
        """
        Fourier basis consists of subsets factored by symmetries and spin inversion.
        Subsets are encoded as unsigned integers (uint64).

        If the orbit consists of subsets of different hamming weights,
        the representative is chosen to be the one with the smallest hamming weight, and
        the of the smallest value among those with the smallest hamming weight.

        Here we compute a dictionary with the following keys:
        - "reprs": all unique representatives of the subsets.
        - "bits_to_repr": an array whose i-th element is the representative
            of the i-th subset.
        - "bits_to_repr_index": an array whose i-th element is the index of the i-th representative
            in the array of all representatives.
        - "bits_to_char": character of a group element that brings the representative into an element

        """
        if hasattr(self, "fourier_repr"):
            logger.debug("Using cached fourier_repr")
            return self.fourier_repr

        basis = self.make_fourier_basis()
        all_subsets = np.arange(2**self.number_spins, dtype=np.uint64)
        subset_to_repr_original, _, _ = basis.state_info(all_subsets)
        assert isinstance(subset_to_repr_original, np.ndarray)

        hamming_weights = popcount(subset_to_repr_original)

        inverted = np.bitwise_xor(all_subsets, 2**self.number_spins - 1)
        subset_to_repr_inverted, _, _ = basis.state_info(inverted)

        subset_to_repr = np.where(
            hamming_weights < self.number_spins / 2,
            subset_to_repr_original,
            np.where(
                hamming_weights > self.number_spins / 2,
                subset_to_repr_inverted,
                np.where(
                    subset_to_repr_original < subset_to_repr_inverted,
                    subset_to_repr_original,
                    subset_to_repr_inverted,
                ),
            ),
        )

        if (self.number_spins // 2) % 2 == 0:
            characters = np.ones_like(subset_to_repr)
        else:
            characters = np.where(subset_to_repr == subset_to_repr_original, 1, -1)

        reprs = np.sort(np.unique(subset_to_repr))

        subset_to_repr_index = np.asarray(np.searchsorted(reprs, subset_to_repr), dtype=np.uint64)
        # TODO: replace with basis.index (?)

        self.fourier_repr = BasisData(
            reprs=reprs,
            bits_to_repr=subset_to_repr,
            bits_to_repr_index=subset_to_repr_index,
            bits_to_char=characters,
        )

        return self.fourier_repr

    def make_fourier_basis_state_info_sym_df(
        self, show_progress: bool = False
    ) -> tuple[npt.NDArray[np.uint64], pd.DataFrame]:
        """
        This function returns a tuple (subsets, df) where subsets is a list of subsets of sites
        and df is a dataframe similar to what is returned by `batched_state_info_df` function.

        Is used by boolean fourier analysis.
        """

        def find_fourier_character(x: pd.DataFrame, n_spins: int) -> np.ndarray:
            if (n_spins // 2) % 2 == 0:
                return np.ones_like(x["representative"])
            return np.where(
                x["representative"] != x["representative_x"],
                -1,
                np.where(x["representative"] != x["representative_y"], 1, 0),
            )

        if hasattr(self, "fourier_basis_state_info"):
            return self.fourier_basis_state_info

        symmetries = ls.Symmetries(
            [ls.Symmetry(automorphism, sector=0) for automorphism in self.get_automorphisms()]
        )
        number_spins = self.number_spins

        fourier_basis = ls.SpinBasis(
            symmetries=symmetries,
            number_spins=number_spins,
            hamming_weight=None,
            spin_inversion=None,
        )
        if show_progress:
            print("MFBSIS: Building Fourier basis...")
        fourier_basis.build()

        all_subsets = np.arange(2**number_spins, dtype="uint64")
        mask = 2**number_spins - 1

        if show_progress:
            print("MFBSIS: Computing state info...")

        fourier_basis_state_info = batched_state_info_df(fourier_basis, all_subsets).drop(
            "norm", axis=1
        )

        if show_progress:
            print("MFBSIS: Computing sign flip basis correspondence...")

        sign_flip_basis_correspondence = (
            batched_state_info_df(fourier_basis, fourier_basis.states ^ mask)
            .assign(initial_representative=fourier_basis.states)
            .drop(["character", "norm"], axis=1)
        )

        if show_progress:
            print("MFBSIS: Computing fourier_basis_state_info_df...")

        fourier_basis_state_info_df = (
            fourier_basis_state_info.reset_index()
            .rename(columns={"index": "state"})
            .merge(
                sign_flip_basis_correspondence,
                left_on="representative",
                right_on="initial_representative",
                how="left",
            )
            .assign(
                hamming_weight_x=lambda x: popcount(x["representative_x"]),
                hamming_weight_y=lambda x: popcount(x["representative_y"]),
            )
            .assign(
                representative=lambda x: np.where(  # if
                    x["hamming_weight_x"] < x["hamming_weight_y"],  # then
                    x["representative_x"],  # else
                    np.where(  # if
                        x["hamming_weight_x"] > x["hamming_weight_y"],  # then
                        x["representative_y"],  # else
                        np.minimum(x["representative_x"], x["representative_y"]),
                    ),
                )
            )
            .assign(character=lambda x: find_fourier_character(x, number_spins))
            .loc[:, ["state", "representative", "character"]]
            .set_index("state")
            .reindex(columns=["representative", "character"])
        )

        subsets = fourier_basis_state_info_df["representative"].unique()
        subsets = subsets[parity(subsets) == 0]
        # for any system invariant under spin inversion, all subsets with odd parity
        # have coefficient 0, so we can ignore them

        self.fourier_basis_state_info = (subsets, fourier_basis_state_info_df)

        return subsets, fourier_basis_state_info_df

    def get_heisenberg_symmetries(self) -> ls.Symmetries:
        """
        Returns ls.Symmetries object for the lattice.
        Each symmetry has sector=0, which is appropriate for some (not any!)
        Heisenberg hamiltonians.
        """
        return ls.Symmetries(
            [ls.Symmetry(automorphism, sector=0) for automorphism in self.get_automorphisms()]
        )

    @property
    def number_spins(self) -> int:
        return len(self.sites)

    def get_basis(
        self,
        use_symmetries: bool = True,
        hamming_weight: int | None = None,
        spin_inversion: int | None = None,
    ) -> ls.SpinBasis:
        """
        Returns a spin basis for the lattice.
        If use_symmetries is True, the basis will contain only the representatives of the
        symmetry-equivalent states.

        It currently supports only sector=0 symmetries (i.e. suitable for Heisenberg hamiltonians)

        If hamming_weight is not None, the basis will contain only states with the given
        hamming weight.

        If spin_inversion is not None, the spin inversion symmetry will be used to reduce the
        number of states in the basis. In this case, spin_inversion is a character of the
        spin inversion representation. I.e. spin_inversion=1 means that the ground state
        is invariant under the spin inversion.

        For Heisenberg models, it is usually hamming_weight=number_spins // 2 and
        spin_inversion=1.
        """

        basis = self.bases.get((use_symmetries, hamming_weight, spin_inversion))
        if basis is not None:
            return basis
        number_spins = self.number_spins
        if use_symmetries:
            symmetries = self.get_heisenberg_symmetries()
        else:
            symmetries = ls.Symmetries([])

        basis = ls.SpinBasis(
            symmetries=symmetries,
            number_spins=number_spins,
            hamming_weight=hamming_weight,
            spin_inversion=spin_inversion,
        )
        basis.build()
        self.bases[(use_symmetries, hamming_weight, spin_inversion)] = basis
        return basis

    def get_state_info_df(
        self,
        use_symmetries: bool = True,
        hamming_weight: int | None = None,
        spin_inversion: int | None = None,
    ) -> pd.DataFrame:
        """
        Returns state_info_df for the given basis with respect to the canonical basis.
        The canonical basis is the basis with the following parameters:
        use_symmetries=False, hamming_weight=number_spins // 2, spin_inversion=None.
        """
        state_info_df = self.state_info_dfs.get((use_symmetries, hamming_weight, spin_inversion))
        if state_info_df is not None:
            return state_info_df

        basis = self.get_basis(use_symmetries, hamming_weight, spin_inversion)
        canonical_basis = self.get_basis(
            use_symmetries=False,
            hamming_weight=self.number_spins // 2,
            spin_inversion=None,
        )
        state_info_df = batched_state_info_df(basis, canonical_basis.states)
        self.state_info_dfs[(use_symmetries, hamming_weight, spin_inversion)] = state_info_df
        return state_info_df

    # def get_canonical_heisenberg_basis(self):
    #     """
    #     This function builds a canonical basis for the Heisenberg model on the lattice.
    #     No symmetries, hamming_weight = number_spins // 2, spin_inversion = None.
    #     """

    #     if hasattr(self, "canonical_heisenberg_basis"):
    #         return self.canonical_heisenberg_basis
    #     number_spins = len(self.sites)
    #     self.canonical_heisenberg_basis = ls.SpinBasis(
    #         symmetries=ls.Symmetries([]),
    #         number_spins=number_spins,
    #         hamming_weight=number_spins // 2,
    #         spin_inversion=None,
    #     )
    #     self.canonical_heisenberg_basis.build()
    #     return self.canonical_heisenberg_basis

    # def get_heisenberg_basis_sym(self):
    #     """
    #     This function builds a basis with symmetries for the Heisenberg model
    #     on the lattice.

    #     The symmetries are the automorphisms of the lattice.
    #     hamming_weight = number_spins // 2, spin_inversion = 1.
    #     """
    #     if hasattr(self, "heisenberg_basis"):
    #         return self.heisenberg_basis

    #     number_spins = len(self.sites)
    #     symmetries_lst = [
    #         ls.Symmetry(automorphism, sector=0) for automorphism in self.get_automorphisms()
    #     ]
    #     symmetries = ls.Symmetries(symmetries_lst)
    #     self.heisenberg_basis = ls.SpinBasis(
    #         symmetries=symmetries,
    #         number_spins=number_spins,
    #         hamming_weight=number_spins // 2,
    #         spin_inversion=1,
    #     )
    #     self.heisenberg_basis.build()
    #     return self.heisenberg_basis

    def plot(
        self,
        spins: None | int | np.uint64 | npt.NDArray | list[int] = None,
        show_edges=True,
        show_numbers=True,
        permutation: None | list[int] | Permutation = None,
        ax=None,
        swap_axes=False,
    ):
        if spins is None:
            spins = 0

        if isinstance(spins, (int, np.uint64)):
            spins = np.array(
                make_unpacked_configurations(np.array(spins, dtype="uint64"), self.number_spins)
            )
        elif isinstance(spins, list):
            spins = np.eye(self.number_spins, dtype=np.int64)[spins].sum(axis=0)

        spins_df = pd.DataFrame(dict(spin=spins))
        sites_df = self.sites_df.merge(spins_df, left_on="num", right_index=True).assign(
            alpha=lambda df: df["is_canonical"].apply(lambda x: 1 if x else 0.3)
        )

        if ax is None:
            _, ax = plt.subplots()

        x, y = "emb_x", "emb_y"
        if swap_axes:
            x, y = y, x

        scatter_plot(
            sites_df,
            x=x,
            y=y,
            color="spin" if spins is not None else None,
            alpha="alpha",
            ax=ax,
            scatter_kws={"s": 100, "zorder": 10},
        )

        if show_edges:
            for (start, end), kind in self.edges:
                x_start, y_start = self.lattice_basis @ start
                x_end, y_end = self.lattice_basis @ end
                if swap_axes:
                    x_start, y_start = y_start, x_start
                    x_end, y_end = y_end, x_end

                ax.plot(
                    [x_start, x_end],
                    [y_start, y_end],
                    color="gray",
                    linestyle=["solid", "dashed", "dotted", "dashdot"][kind - 1],
                )

        if show_numbers:
            for site, num in self.site_to_num.items():
                x_annotate, y_annotate = self.lattice_basis @ site
                if swap_axes:
                    x_annotate, y_annotate = y_annotate, x_annotate
                ax.annotate("  " + str(num), (x_annotate, y_annotate))

        if permutation is not None:
            if isinstance(permutation, Permutation):
                permutation = permutation.array_form
            for _, row in self.sites_df.iterrows():
                x_text = row[x] + 0.2
                y_text = row[y] + 0.2
                ax.text(x_text, y_text, "(" + str(permutation[row["num"]]) + ")")

        ax.axis("equal")
        return ax

    def plot_subsets(
        self,
        subsets: npt.NDArray[np.uint64],
        titles: list[str],
        legend=True,
        size_scale=3,
        **kwargs,
    ):
        fig, axes = plt.subplots(
            1, len(subsets), figsize=(len(subsets) * size_scale, size_scale), squeeze=False
        )
        for ax, subset, title in zip(axes[0], subsets, titles):
            self.plot(spins=subset, ax=ax, **kwargs)
            ax.axis("off")
            ax.set_title(title)
            if not legend:
                ax.get_legend().remove()
        return fig


class ParallelogramSpinLattice(SpinLattice):
    def __init__(
        self,
        u,
        v,
        named_sites: dict[str, npt.NDArray],
        named_edges,
        fundamental_domain_size: int | npt.NDArray[np.int_] = 1,
        width=1,
        height=1,
        boundary_conditions: Literal["periodic", "open"] = "periodic",
        enumerate_along: Literal["x", "y"] | None = None,
        automorphisms: str = "all",
    ):
        """
        Generic class to generate lattices with different kinds of edges (i.e. for J1-J2 systems)

        Parameters
        ----------

        u, v : np.array([x, y])
            lattice generators

        named_sites : dict[str, np.array]
            dictionary name_of_site -> site_coordinates, like {"A": np.array([0, 0]), ...}

        named_edges : list[tuple[str, int]]
            list of two-tuples (name, kind), e.g. [("AB", 1), ("BC", 2), ...]

        fundamental_domain_size: int | np.array
            the size of fundamental domain, can be integer number or np.array([w, h])

        width, height: int
            weight and height of the lattice (in factors of width and height of the fundamental domain)
        """

        super().__init__()

        self.lattice_basis = np.c_[u, v]
        self.height = height
        self.width = width
        self.fundamental_domain_size = fundamental_domain_size * np.array([1, 1])
        self.boundary_conditions = boundary_conditions
        self.fourier_basis_state_info: tuple[np.ndarray, pd.DataFrame]
        self.enumerate_along = enumerate_along
        self.automorphisms = automorphisms

        self.num_tensor_order: npt.NDArray

        frame = fundamental_domain_size * (
            np.array([width, height]) + (boundary_conditions == "open")
        )
        self.frame = frame

        edges: list[tuple[tuple[npt.NDArray, npt.NDArray], Any]] = [
            ((named_sites[start], named_sites[end]), kind) for (start, end), kind in named_edges
        ]

        sites = []
        self.edges: list[tuple[tuple[npt.NDArray, npt.NDArray], int]] = []

        for i, j in product(range(width), range(height)):
            shift = fundamental_domain_size * np.array([i, j])
            for site in named_sites.values():
                sites.append(site + shift)
            for (start, end), kind in edges:
                self.edges.append(
                    (
                        (start + shift, end + shift),
                        kind,
                    )
                )

        self.site_to_num: dict[tuple[float, float], int] = {}
        new_num = 0

        if enumerate_along == "y":
            sites = sorted(sites, key=lambda x: (x[0], x[1]))
        elif enumerate_along == "x":
            sites = sorted(sites, key=lambda x: (x[1], x[0]))

        for site in sites:
            canonical_coords = tuple(site % frame)
            if canonical_coords in self.site_to_num:
                self.site_to_num[tuple(site)] = self.site_to_num[canonical_coords]
            else:
                self.site_to_num[canonical_coords] = new_num
                self.site_to_num[tuple(site)] = new_num
                new_num += 1

        self.x_translation = self.get_translation("x")
        self.y_translation = self.get_translation("y")

    def get_automorphisms(self) -> list[list[int]]:
        if self.automorphisms == "all":
            return super().get_automorphisms()
        elif self.automorphisms == "translations":
            return [
                g.array_form
                for g in PermutationGroup(
                    [
                        Permutation(self.get_translation("x")),
                        Permutation(self.get_translation("y")),
                    ]
                ).elements
            ]

    def spin_config_to_tensor(self, cfgs: npt.NDArray[np.uint64]) -> np.ndarray:
        raise NotImplementedError

    def get_translation(self, direction: str) -> list[int]:
        if direction not in ["x", "y"]:
            raise ValueError("direction must be 'x' or 'y'")

        n_direction = {"x": 0, "y": 1}[direction]
        sites_df_shifted = self.sites_df.query("is_canonical")[["num", "ix", "iy"]].assign(
            **{
                f"i{direction}_shifted": lambda df: (
                    df[f"i{direction}"] + self.fundamental_domain_size[n_direction]
                )
                % self.frame[n_direction]
            }
        )

        shift = (
            self.sites_df[["num", "ix", "iy"]]
            .merge(
                sites_df_shifted,
                left_on=["ix", "iy"],
                right_on=["ix_shifted", "iy"] if direction == "x" else ["ix", "iy_shifted"],
                suffixes=("", "__shifted"),
            )[["num", "num__shifted"]]
            .set_index("num__shifted")["num"]
            .to_dict()
        )

        assert len(shift) == self.number_spins

        return [shift[i] for i in range(self.number_spins)]

    def get_cache_id(self) -> str:
        boundary = "" if self.boundary_conditions == "periodic" else self.boundary_conditions
        ordered = f"-enumerate-along-{self.enumerate_along}" if self.enumerate_along else ""
        automorphisms = (
            f"-automorphisms-{self.automorphisms}" if self.automorphisms != "all" else ""
        )
        return f"{self.__class__.__name__}{self.width}x{self.height}{boundary}{ordered}{automorphisms}"

    @property
    def sites_df(self) -> pd.DataFrame:
        sites_df = pd.DataFrame(
            [
                [
                    num,
                    *coords,
                    (np.array(coords) == np.array(coords) % self.frame).all(),
                ]
                for coords, num in self.site_to_num.items()
            ],
            columns=["num", "ix", "iy", "is_canonical"],
        )

        sites_df[["emb_x", "emb_y"]] = (self.lattice_basis @ sites_df[["ix", "iy"]].T.values).T
        return sites_df


class ChainLattice(ParallelogramSpinLattice):
    def __init__(self, width=1, height=1, **kwargs):
        """
        Generates chain lattice.

        ```
        A ----- B
        ```
        Size of the fundamentail domain is 1×0
        """
        u = np.array([1, 0])
        v = np.array([0, 1])

        named_sites = {
            "A": np.array([0, 0]),
            "B": np.array([1, 0]),
        }

        named_edges = [
            ("AB", 1),
        ]

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=np.array([1, 1]),
            width=width,
            height=height,
            **kwargs,
        )


class SquareLattice(ParallelogramSpinLattice):
    def __init__(self, width=1, height=1, **kwargs):
        r"""
        Generates square J1-J2 lattice.

        The fundamental domain:

        ```
        C ----- D
        | \\ // |
        |  \V/  |
        |  /Ʌ\  |
        | // \\ |
        A ----- B
        ```

        Size of the fundamentail domain is 1×1
        """
        u = np.array([1, 0])
        v = np.array([0, 1])

        named_sites = {
            "A": np.array([0, 0]),
            "B": np.array([1, 0]),
            "C": np.array([0, 1]),
            "D": np.array([1, 1]),
        }

        named_edges = [("AB", 1), ("AC", 1), ("CD", 1), ("BD", 1), ("CB", 2), ("AD", 2)]

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=1,
            width=width,
            height=height,
            **kwargs,
        )

        t_frame = self.sites_df.query("is_canonical").set_index("num").reset_index()

        # TODO: refactor as a test
        assert t_frame["ix"].nunique() == self.width
        assert t_frame["iy"].nunique() == self.height
        assert t_frame.duplicated(["ix", "iy"]).sum() == 0

        self.num_tensor_order = np.asarray(
            t_frame.sort_values(["ix", "iy"], ignore_index=True)["num"].values
        )

    def spin_config_to_tensor(self, cfgs: npt.NDArray[np.uint64]) -> np.ndarray:
        return make_unpacked_configurations(cfgs, number_spins=self.number_spins)[
            ..., self.num_tensor_order
        ].reshape(-1, self.width, self.height, 1)


class SquareLattice1Diag(ParallelogramSpinLattice):
    def __init__(self, width=1, height=1, **kwargs):
        r"""
        Generates square lattice with one J2 diagonal. Equivalent to the
        Triangular lattice.

        The fundamental domain:

        ```
        C ----- D
        | \\    |
        |  \\   |
        |   \\  |
        |    \\ |
        A ----- B
        ```

        Size of the fundamentail domain is 1×1
        """
        u = np.array([1, 0])
        v = np.array([0, 1])

        named_sites = {
            "A": np.array([0, 0]),
            "B": np.array([1, 0]),
            "C": np.array([0, 1]),
            "D": np.array([1, 1]),
        }

        named_edges = [("AB", 1), ("AC", 1), ("CD", 1), ("BD", 1), ("CB", 2)]

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=1,
            width=width,
            height=height,
            **kwargs,
        )


class TriangularLattice(ParallelogramSpinLattice):
    def __init__(self, width=1, height=1, **kwargs):
        r"""
        Generates triangular lattice.

        The fundamental domain:

        ```
             C
            /\\
           /  \\
          /    \\
         A ----- B                    
        ```
        Size of the fundamentail domain is 1×1
        """
        theta = np.pi / 3
        u = np.array([1, 0])
        v = np.array([np.cos(theta), np.sin(theta)])

        named_sites = {
            "A": np.array([0, 0]),
            "B": np.array([1, 0]),
            "C": np.array([0, 1]),
        }

        named_edges = [("AB", 1), ("AC", 1), ("BC", 2)]

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=1,
            width=width,
            height=height,
            **kwargs,
        )

        t_frame = self.sites_df.query("is_canonical").set_index("num").reset_index()

        # TODO: refactor as a test
        assert t_frame["ix"].nunique() == self.width
        assert t_frame["iy"].nunique() == self.height
        assert t_frame.duplicated(["ix", "iy"]).sum() == 0

        self.num_tensor_order = np.asarray(
            t_frame.sort_values(["ix", "iy"], ignore_index=True)["num"].values
        )

    def spin_config_to_tensor(self, cfgs: npt.NDArray[np.uint64]) -> np.ndarray:
        return make_unpacked_configurations(cfgs, number_spins=self.number_spins)[
            ..., self.num_tensor_order
        ].reshape(-1, self.width, self.height, 1)


class KagomeLattice(ParallelogramSpinLattice):
    def __init__(self, width=1, height=1, isotropic=False, **kwargs):
        """
        Generates Kagome lattice.

        The fundamental domain:
        ```
             F -- G -- H
            /      \\ /
           D         E
         /  \\      /
        A -- B -- C
        ```
        Size of the fundamental domain is 2×2. Bonds DB and GE have kind 2 if isotropic=False.
        """

        theta = np.pi / 3
        u = np.array([1, 0])
        v = np.array([np.cos(theta), np.sin(theta)])

        named_sites = {
            "A": np.array([0, 0]),
            "B": np.array([1, 0]),
            "C": np.array([2, 0]),
            "D": np.array([0, 1]),
            "E": np.array([2, 1]),
            "F": np.array([0, 2]),
            "G": np.array([1, 2]),
            "H": np.array([2, 2]),
        }

        named_edges = [
            ("AB", 1),
            ("BC", 1),
            ("AD", 1),
            ("BD", 1 + (not isotropic)),
            ("DF", 1),
            ("FG", 1),
            ("GE", 1 + (not isotropic)),
            ("GH", 1),
            ("EH", 1),
            ("CE", 1),
        ]

        self.isotropic = isotropic

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=2,
            width=width,
            height=height,
            **kwargs,
        )

        t_frame = (
            self.sites_df.query("is_canonical")
            .assign(
                tx=lambda df: df["ix"] // 2,
                ty=lambda df: df["iy"] // 2,
                tz=lambda df: df["ix"] % 2 + 2 * (df["iy"] % 2),
            )
            .set_index("num")
            .reset_index()
        )

        # TODO: refactor as a test
        assert t_frame["tz"].nunique() == 3
        assert t_frame["tx"].nunique() == self.width
        assert t_frame["ty"].nunique() == self.height
        assert t_frame.duplicated(["tx", "ty", "tz"]).sum() == 0

        self.num_tensor_order = np.asarray(
            t_frame.sort_values(["tx", "ty", "tz"], ignore_index=True)["num"].values
        )

    def get_cache_id(self) -> str:
        return super().get_cache_id() + ("_isotropic" if self.isotropic else "")

    def spin_config_to_tensor(self, cfgs: npt.NDArray[np.uint64]) -> np.ndarray:
        return make_unpacked_configurations(cfgs, number_spins=self.number_spins)[
            ..., self.num_tensor_order
        ].reshape(-1, self.width, self.height, 3)


def do_images_intersect(
    domain: list[int], group_elements: list[Permutation], verbose=False
) -> bool:
    # For each element in the fundamental_domain
    for a in domain:
        # Apply each group element
        for g in group_elements:
            ga = g(a)
            # If the result is in the fundamental_domain and different from a, return False
            if ga in domain and ga != a:
                if verbose:
                    print(f"Found intersection between {a} and {ga}")
                    print(f"Group element: {g.array_form}")
                return True
    return False


def get_factor_graph(
    graph: ig.Graph, group_elements: list[Permutation], fundamental_domain: list[int]
) -> ig.Graph:
    assert not do_images_intersect(fundamental_domain, group_elements)
    # Create an empty graph with vertices corresponding to the fundamental_domain
    factor_graph = ig.Graph()
    factor_graph.add_vertices(map(str, fundamental_domain))

    # For each pair of vertices a and b in fundamental_domain
    for a in fundamental_domain:
        for b in fundamental_domain:
            if a == b:
                continue
            #           print(f"Considering vertices {a} and {b}")
            # If there exists a permutation g in group such that a and g(b) are
            # connected by an edge in the original graph
            for g in [Permutation([[max(fundamental_domain)]])] + group_elements:
                gb = g(b)
                #              print(f"Considering vertex {gb=}")
                if graph.are_connected(a, gb):
                    #                    print("There is a connection!")
                    # Check if the edge already exists in the factor_graph
                    if factor_graph.get_eid(str(a), str(b), error=False) == -1:
                        #                     print("Adding edge to factor graph")
                        # If not, then add an edge between a and b in factor_graph,
                        # preserving the edge attribute 'kind' from the original graph
                        edge_id = graph.get_eid(a, gb)
                        kind = graph.es[edge_id]["kind"]
                        factor_graph.add_edge(str(a), str(b), kind=kind)
    #                else:
    #                    print("Edge already exists, skipping")

    return factor_graph


class FactorLattice(SpinLattice):
    def __init__(
        self,
        initial_lattice: SpinLattice,
        group_elements: list[Permutation],
        fundamental_domain: list[int],
    ):
        assert set(group_elements).issubset(
            {Permutation(g) for g in initial_lattice.get_automorphisms()}
        )

        self.initial_lattice = initial_lattice
        self.lattice_basis = initial_lattice.lattice_basis
        self.fundamental_domain = fundamental_domain

        factor_graph = get_factor_graph(
            initial_lattice.as_igraph(), group_elements, fundamental_domain
        )

        vertex_coords = (
            initial_lattice.sites_df.query("is_canonical")
            .set_index("num")
            .loc[[int(v["name"]) for v in factor_graph.vs]][["ix", "iy"]]
            .to_numpy()
        )

        self.edges = []

        for i, edge in enumerate(factor_graph.es):
            self.edges.append(
                (
                    (
                        vertex_coords[edge.source],
                        vertex_coords[edge.target],
                    ),
                    edge["kind"],
                )
            )

        self.site_to_num = {}

        for i, vertex_coord in enumerate(vertex_coords):
            self.site_to_num[tuple(vertex_coord)] = i

        super().__init__()

    @property
    def sites_df(self) -> pd.DataFrame:
        sites_df = pd.DataFrame(
            [
                [
                    num,
                    *coords,
                    True,
                ]
                for coords, num in self.site_to_num.items()
            ],
            columns=["num", "ix", "iy", "is_canonical"],
        )

        sites_df[["emb_x", "emb_y"]] = (self.lattice_basis @ sites_df[["ix", "iy"]].T.values).T
        return sites_df

    def get_cache_id(self):
        return f"{self.initial_lattice.get_cache_id()}_factor_{'-'.join(map(str, self.fundamental_domain))}"


class AllToAllLattice(SpinLattice):
    def __init__(self, original_lattice: SpinLattice):
        super().__init__()
        self.original_lattice = original_lattice
        self.lattice_basis = original_lattice.lattice_basis
        self.site_to_num = original_lattice.site_to_num
        sites = original_lattice.sites_df.query("is_canonical")[["ix", "iy"]].to_numpy()
        self.edges: list[tuple[tuple[npt.NDArray, npt.NDArray], int]] = [  # (start, end), kind
            ((p, q), 1) for p in sites for q in sites
        ]

    @property
    def sites_df(self) -> pd.DataFrame:
        return self.original_lattice.sites_df

    def get_cache_id(self) -> str:
        return f"AllToAll_{self.original_lattice.get_cache_id()}"
