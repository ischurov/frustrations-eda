from sympy.combinatorics import Permutation
import torch
from torch import nn
from gcnn_naive import SplitGroupResConvNet, GConvLattice, GConvG


class GXorLattice(GConvLattice):
    def __init__(
        self,
        group_elements: list[Permutation],
        xor_masks: torch.Tensor,
    ):
        """
        Implements a xor convolution on a lattice.

        Args:
            group_elements: list of permutations that form a group
            xor_masks: tensor of shape (n_masks, n_nodes) that contains the xor masks.
                       The masks should be sequences of 0 and 1
        """
        out_channels, number_spins = xor_masks.size()
        super().__init__(group_elements, torch.arange(number_spins), out_channels)
        del self.filter
        self.filter = xor_masks.to(torch.float32)
        self.filter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies xor convolution to a signal on a lattice.

        Args:
            x: tensor of shape (batch_size, n_nodes) that contains the binary signal
               with 0-1 encoding.

        Returns:
            tensor of shape (batch_size, n_masks, len(self.group_elements)) that contains the
            transformed signal
        """
        conv_result = super().forward(x)
        return (conv_result.to(torch.int8) % 2).to(torch.float32)


class SplitGroupXorResConvNet(SplitGroupResConvNet):
    def __init__(
        self,
        tx: Permutation,
        ty: Permutation,
        filter1_sites: list[int] | torch.Tensor,
        xor_masks: torch.Tensor,
        additional_generators: list[Permutation] | None = None,
        extend_filter1: tuple[int, int] = (1, 1),
        filter_size: tuple[int, int] = (2, 2),
        channels: int = 4,
        blocks: int = 2,
    ):
        super().__init__(
            tx=tx,
            ty=ty,
            filter1_sites=filter1_sites,
            additional_generators=additional_generators,
            extend_filter1=extend_filter1,
            filter_size=filter_size,
            channels=channels,
            blocks=blocks,
        )
        internal_channels = channels + xor_masks.shape[0]
        self.other_layers = nn.ModuleList(
            [
                GConvG(
                    group_elements=self.group_elements,
                    filter_idxs=self.filter_idxs[i],
                    in_channels=internal_channels,
                    out_channels=internal_channels,
                )
                for i in range(len(self.filter_idxs))
            ]
        )
        self.fc = nn.Linear(internal_channels, 1)

        self.xor_module = GXorLattice(self.group_elements, xor_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_layer_output = torch.relu(self.layer1(x))
        xor_embedding = self.xor_module(x)
        x = torch.concat([first_layer_output, xor_embedding], dim=1)
        for layer in self.other_layers:
            x = layer(x)
            x = torch.relu(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x
