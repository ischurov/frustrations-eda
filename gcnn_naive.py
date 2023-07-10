import functools
import itertools
import operator

import torch
from sympy.combinatorics import Permutation, PermutationGroup
from torch import nn


class GConvLattice(nn.Module):
    def __init__(
        self,
        group_elements: list[Permutation],
        filter_idxs: list[int] | torch.Tensor,
        out_channels: int,
    ):
        """
        Implements graph equivariant layer that maps signal on a lattice
        to a signal on a (sub)group of lattice's automorphisms.

        Args:
            group_elements: list of permutations that form a group
            filter_idxs: indices of spins that are used in the filter
            out_channels: number of output channels
        """
        super().__init__()
        self.filter_size = len(filter_idxs)
        self.out_channels = out_channels
        self.filter_idxs = filter_idxs
        self.group_elements = group_elements
        self.inv_group_elements_tensor = torch.tensor(
            [(~g).array_form for g in self.group_elements], dtype=torch.long
        )
        self.number_spins = self.inv_group_elements_tensor.size()[1]
        self.filter = nn.Parameter(torch.empty(out_channels, self.filter_size))
        torch.nn.init.xavier_uniform_(self.filter)
        self.filter.requires_grad = True

    def _forward_reference(self, batch: torch.Tensor):
        """
        This is reference implementation, kept for testing purposes.
        """
        output = torch.zeros((batch.size()[0], self.out_channels, len(self.group_elements)))
        filter_extended = torch.zeros((self.out_channels, self.number_spins))
        filter_extended[:, self.filter_idxs] = self.filter

        for i, g in enumerate(self.group_elements):
            permuted_filter = filter_extended[:, (~g).array_form]
            output[:, :, i] = torch.einsum("bn,an->ba", batch, permuted_filter)
        return output

    def forward(self, batch: torch.Tensor):
        filter_extended = torch.zeros((self.out_channels, self.number_spins))

        filter_extended[:, self.filter_idxs] = self.filter

        permuted_filters = filter_extended[:, self.inv_group_elements_tensor.view(-1)].view(
            self.out_channels, len(self.group_elements), self.number_spins
        )

        output = torch.einsum("bs,ogs->bog", batch, permuted_filters)
        return output


def self_action(g: Permutation, group_elements: list[Permutation]) -> Permutation:
    # permutation acts from the right, so it's h * g instead of g * h
    return Permutation([group_elements.index(h * g) for h in group_elements])


class GConvG(nn.Module):
    def __init__(
        self,
        group_elements: list[Permutation],
        filter_idxs: list[int] | torch.Tensor,
        in_channels: int,
        out_channels: int,
    ):
        """
        Implements graph equivariant layer that maps signal on a group
        to signal on a group.

        Args:
            group_elements: list of permutations that form a group
            filter_idxs: indices of group elements that are used in the filter
            in_channels: number of input channels
            out_channels: number of output channels
        """

        super().__init__()
        self.filter_size = len(filter_idxs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_idxs = filter_idxs
        self.group_elements = group_elements
        self.inv_group_elements_tensor = torch.tensor(
            [(~self_action(g, group_elements)).array_form for g in self.group_elements],
            dtype=torch.long,
        )
        self.filter = nn.Parameter(torch.empty(out_channels, in_channels, self.filter_size))
        torch.nn.init.xavier_uniform_(self.filter)
        self.filter.requires_grad = True

    def _forward_reference(self, batch: torch.Tensor):
        """
        This is reference implementation, kept for testing purposes.
        """

        output = torch.zeros((batch.size()[0], self.out_channels, len(self.group_elements)))
        filter_extended = torch.zeros(
            (self.out_channels, self.in_channels, len(self.group_elements))
        )
        filter_extended[:, :, self.filter_idxs] = self.filter

        for action_index, action in enumerate(self.group_elements):
            permuted_filter = filter_extended[
                :, :, self_action(~action, self.group_elements).array_form
            ]
            output[:, :, action_index] = torch.einsum("big,oig->bo", batch, permuted_filter)
            # b: batch, i: in_channels, g: group_elements, o: out_channels

        return output

    def forward(self, batch: torch.Tensor):
        output = torch.zeros((batch.size()[0], self.out_channels, len(self.group_elements)))
        filter_extended = torch.zeros(
            (self.out_channels, self.in_channels, len(self.group_elements))
        )
        filter_extended[:, :, self.filter_idxs] = self.filter

        permuted_filters = filter_extended[:, :, self.inv_group_elements_tensor.view(-1)].view(
            self.out_channels,
            self.in_channels,
            len(self.group_elements),
            len(self.group_elements),
        )

        return torch.einsum("big,oiag->boa", batch, permuted_filters)
        # b: batch, i: input channels, g: group element (as domain coordinate)
        # o: output channel, a: group element (as action)


def mult_comb(elements, degrees):
    """
    Returns e_1 ** d_1 * ... * e_n ** d_n for elements = [e_1, ..., e_n] and degrees = [d_1, ..., d_n]
    """
    return functools.reduce(operator.mul, (e**d for e, d in zip(elements, degrees)))


class SplitGroupConvNet2(nn.Module):
    def __init__(
        self,
        tx: Permutation,
        ty: Permutation,
        filter1_sites: list[int] | torch.Tensor,
        additional_generators: list[Permutation] | None = None,
        filter1_width=2,
        filter1_height=2,
        filter2_width=2,
        filter2_height=2,
        channels1=32,
        channels2=64,
    ):
        """
        We expect that the group is generated by tx, ty and additional_generators.
        """
        if additional_generators is None:
            additional_generators = []

        super().__init__()

        generators = [tx, ty] + additional_generators
        group_elements: list[Permutation] = sorted(
            list(PermutationGroup(*generators).elements), key=lambda g: tuple(g.array_form)
        )

        filter_idxs_1 = sorted(
            set(
                int((tx**a * ty**b)(site))
                for site in filter1_sites
                for a in range(filter1_width)
                for b in range(filter1_height)
            )
        )

        self.group_elements = group_elements

        self.layer1 = GConvLattice(
            group_elements=group_elements, filter_idxs=filter_idxs_1, out_channels=channels1
        )

        filter_idxs_2 = [
            group_elements.index(
                mult_comb([tx, ty] + additional_generators, [x_deg, y_deg] + additional_deg)
            )
            for x_deg, y_deg, *additional_deg in itertools.product(
                range(filter2_width),
                range(filter2_height),
                *[range(int(g.order())) for g in additional_generators]
            )
        ]

        self.layer2 = GConvG(
            group_elements=group_elements,
            filter_idxs=filter_idxs_2,
            in_channels=channels1,
            out_channels=channels2,
        )

        self.fc = nn.Linear(channels2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


class SplitGroupConvNet(nn.Module):
    def __init__(
        self,
        tx: Permutation,
        ty: Permutation,
        filter1_sites: list[int] | torch.Tensor,
        additional_generators: list[Permutation] | None = None,
        filter_sizes=[(2, 2), (2, 2)],
        channels=[32, 64],
    ):
        """
        We expect that the group is generated by tx, ty and additional_generators.
        """
        if additional_generators is None:
            additional_generators = []

        super().__init__()

        generators = [tx, ty] + additional_generators
        group_elements: list[Permutation] = sorted(
            list(PermutationGroup(*generators).elements), key=lambda g: tuple(g.array_form)
        )

        filter_idxs_1 = sorted(
            set(
                int((tx**a * ty**b)(site))
                for site in filter1_sites
                for a in range(filter_sizes[0][0])
                for b in range(filter_sizes[0][1])
            )
        )

        self.group_elements = group_elements

        self.layer1 = GConvLattice(
            group_elements=group_elements, filter_idxs=filter_idxs_1, out_channels=channels[0]
        )

        filter_idxs = [
            [
                group_elements.index(
                    mult_comb([tx, ty] + additional_generators, [x_deg, y_deg] + additional_deg)
                )
                for x_deg, y_deg, *additional_deg in itertools.product(
                    range(filter_width),
                    range(filter_height),
                    *[range(int(g.order())) for g in additional_generators]
                )
            ]
            for filter_width, filter_height in filter_sizes[1:]
        ]

        self.other_layers = nn.ModuleList(
            [
                GConvG(
                    group_elements=group_elements,
                    filter_idxs=filter_idxs[i],
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
                for i in range(len(filter_idxs))
            ]
        )

        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        for layer in self.other_layers:
            x = layer(x)
            x = torch.relu(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


class SplitGroupResConvNet(nn.Module):
    def __init__(
        self,
        tx: Permutation,
        ty: Permutation,
        filter1_sites: list[int] | torch.Tensor,
        additional_generators: list[Permutation] | None = None,
        filter_size_head=(1, 1),
        filter_size=(2, 2),
        channels=4,
        blocks=2,
    ):
        """
        We expect that the group is generated by tx, ty and additional_generators.
        """
        if additional_generators is None:
            additional_generators = []

        super().__init__()

        generators = [tx, ty] + additional_generators
        group_elements: list[Permutation] = sorted(
            list(PermutationGroup(*generators).elements), key=lambda g: tuple(g.array_form)
        )

        filter_idxs_1 = sorted(
            set(
                int((tx**a * ty**b)(site))
                for site in filter1_sites
                for a in range(filter_size_head[0])
                for b in range(filter_size_head[1])
            )
        )

        self.group_elements = group_elements

        self.layer1 = GConvLattice(
            group_elements=group_elements, filter_idxs=filter_idxs_1, out_channels=channels
        )

        filter_idxs = [
            [
                group_elements.index(
                    mult_comb([tx, ty] + additional_generators, [x_deg, y_deg] + additional_deg)
                )
                for x_deg, y_deg, *additional_deg in itertools.product(
                    range(filter_size[0]),
                    range(filter_size[1]),
                    *[range(int(g.order())) for g in additional_generators]
                )
            ]
            for _ in range(blocks)
        ]

        self.other_layers = nn.ModuleList(
            [
                GConvG(
                    group_elements=group_elements,
                    filter_idxs=filter_idxs[i],
                    in_channels=channels,
                    out_channels=channels,
                )
                for i in range(len(filter_idxs))
            ]
        )

        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        for layer in self.other_layers:
            x = x + layer(x)
            x = torch.relu(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x
