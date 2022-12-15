import matplotlib.pyplot as plt
from itertools import product
import igraph as ig
import numpy as np
from collections import defaultdict
from bitarray.util import int2ba
import pandas as pd
import seaborn as sns

# BASED ON: https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/


def scatter_plot(data, x, y, color, ax=None, scatter_kws=None):
    if scatter_kws is None:
        scatter_kws = {}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Get Unique continents
    color_labels = sorted(data[color].unique())

    # List of colors in the color palettes
    rgb_values = sns.color_palette("Set2", len(color_labels))

    # Map continents to the colors
    color_map = dict(zip(color_labels, rgb_values))

    for key, group in data.groupby(color):
        group.plot(
            ax=ax,
            kind="scatter",
            x=x,
            y=y,
            label=key,
            color=color_map[key],
            **scatter_kws,
        )

    return ax


# END BASED


class SpinLattice:
    def __init__(
        self,
        u,
        v,
        named_sites,
        named_edges,
        fundamental_domain_size=1,
        width=1,
        height=1,
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

        self.lattice_basis = np.c_[u, v]
        self.height = height
        self.width = width

        edges = [
            ((named_sites[start], named_sites[end]), kind)
            for (start, end), kind in named_edges
        ]

        sites = []
        self.edges = []

        for i, j in product(range(width), range(height)):
            shift = fundamental_domain_size * np.array([i, j])
            for site in named_sites.values():
                sites.append(site + shift)
            for (start, end), kind in edges:
                self.edges.append(((start + shift, end + shift), kind,))

        self.site_to_num = {}
        new_num = 0

        frame = fundamental_domain_size * np.array([width, height])
        for site in sites:
            canonical_coords = tuple(site % frame)
            if canonical_coords in self.site_to_num:
                self.site_to_num[tuple(site)] = self.site_to_num[canonical_coords]
            else:
                self.site_to_num[canonical_coords] = new_num
                self.site_to_num[tuple(site)] = new_num
                new_num += 1

        self.edges_to_kind = {}
        for (start, end), kind in self.edges:
            self.edges_to_kind[
                (self.site_to_num[tuple(start)], self.site_to_num[tuple(end)])
            ] = kind

        sites_df = pd.DataFrame(
            [
                [num, *coords, (np.array(coords) == np.array(coords) % frame).all()]
                for coords, num in self.site_to_num.items()
            ],
            columns=["num", "ix", "iy", "is_canonical"],
        )

        sites_df[["emb_x", "emb_y"]] = (
            self.lattice_basis @ sites_df[["ix", "iy"]].T.values
        ).T
        self.sites_df = sites_df

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

    @property
    def file_stem(self):
        return f"{self.__class__.__name__}{self.width}x{self.height}"

    def as_igraph(self) -> ig.Graph:
        edges, kinds = zip(*self.edges_to_kind.items())
        return ig.Graph(edges=edges, edge_attrs={"kind": kinds})

    def get_automorphisms(self) -> list[list[int]]:
        g = self.as_igraph()
        return g.get_automorphisms_vf2(edge_color=g.es["kind"])

    def plot(self, spins=None, show_edges=True, ax=None):
        """Plots the lattice and optionally visualizes some spin configuration"""
        if spins is not None:
            spins_df = pd.DataFrame(dict(spin=spins))
            sites_df = self.sites_df.merge(spins_df, left_on="num", right_index=True)
        else:
            sites_df = self.sites_df

        if ax is None:
            _, ax = plt.subplots()

        scatter_plot(
            sites_df,
            x="emb_x",
            y="emb_y",
            color="spin" if spins is not None else None,
            ax=ax,
            scatter_kws={"s": 100, "zorder": 10},
        )

        if show_edges:
            for (start, end), kind in self.edges:
                ax.plot(
                    *zip(self.lattice_basis @ start, self.lattice_basis @ end),
                    color="gray",
                    linestyle=["solid", "dashed", "dotted", "dashdot"][kind - 1],
                )

        for site, num in self.site_to_num.items():
            ax.annotate("  " + str(num), self.lattice_basis @ site)

        ax.axis("equal")
        return ax


class ChainLattice(SpinLattice):
    def __init__(self, width=1, height=1):
        """

        A ----- B

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
        )


class SquareLattice(SpinLattice):
    def __init__(self, width=1, height=1):
        """
        Generates square J1-J2 lattice.

        The fundamental domain:

        C ----- D
        | \\ // |
        |  \V/  |
        |  /Ʌ\  |
        | // \\ |
        A ----- B

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
        )


class KagomeLattice(SpinLattice):
    def __init__(self, width=1, height=1):
        """
        Generates Kagome lattice.

        The fundamental domain:

             F -- G -- H
            /      \\ /
           D         E
         /  \\      /
        A -- B -- C

        Size of the fundamental domain is 2×2
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
            ("BD", 2),
            ("DF", 1),
            ("FG", 1),
            ("GE", 2),
            ("GH", 1),
            ("EH", 1),
            ("CE", 1),
        ]

        super().__init__(
            u=u,
            v=v,
            named_sites=named_sites,
            named_edges=named_edges,
            fundamental_domain_size=2,
            width=width,
            height=height,
        )
