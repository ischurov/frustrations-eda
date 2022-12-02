import matplotlib.pyplot as plt
from itertools import product
import igraph as ig
import numpy as np
from collections import defaultdict
from bitarray.util import int2ba
import pandas as pd
import seaborn as sns

class SpinLattice:
    def __init__(self, u, v, named_sites, named_edges, fundamental_domain_size=1, width=1, height=1):
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
                self.edges.append(
                    (
                        (start + shift, end + shift),
                        kind,
                    )
                )

        self.site_to_num = {}
        new_num = 0
        
        frame = (fundamental_domain_size * np.array([width, height]))
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
            [[num, *coords, (np.array(coords) == np.array(coords) % frame).all()]
             for coords, num in self.site_to_num.items()],
            columns=['num', 'ix', 'iy', 'is_canonical']
        )
        
        sites_df[['emb_x', 'emb_y']] = (self.lattice_basis @ sites_df[['ix', 'iy']].T.values).T
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
            
    def as_igraph(self) -> ig.Graph:
        edges, kinds = zip(*self.edges_to_kind.items())
        return ig.Graph(edges=edges, edge_attrs={'kind': kinds})
    
    def get_automorphisms(self) -> list[list[int]]:
        g = self.as_igraph()
        return g.get_automorphisms_vf2(edge_color=g.es['kind'])
                        
    def plot(self, spins=None, show_edges=True):
        """Plots the lattice and optionally visualizes some spin configuration"""
        if spins is not None:
            spins_df = pd.DataFrame(dict(spin=spins))
            sites_df = self.sites_df.merge(spins_df, left_on='num', right_index=True)
        else:
            sites_df = self.sites_df
        
        fg = sns.lmplot(sites_df,  hue='spin' if spins is not None else None,
                    x='emb_x', y='emb_y', fit_reg=False, scatter_kws={"s": 100, "zorder":10})
        
        if show_edges:
            for (start, end), kind in self.edges:
                fg.ax.plot(
                    *zip(self.lattice_basis @ start, self.lattice_basis @ end),
                    color=f"C{kind}",
                )
        
        for site, num in self.site_to_num.items():
            fg.ax.annotate("  " + str(num), self.lattice_basis @ site)
            
                