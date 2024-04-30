from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from spin_lattices import ParallelogramSpinLattice
from my_stopwatch import stopwatch
from loguru import logger
from itertools import chain


def circular_pad2d(input: torch.Tensor, pad: int | Sequence[int]):
    if isinstance(pad, int):
        pad = [pad] * 2
    pad_x, pad_y = pad
    width, height = input.shape[-1], input.shape[-2]

    replication_x, pad_x = divmod(pad_x, width)
    replication_y, pad_y = divmod(pad_y, height)

    input = input.repeat(
        *([1] * (input.dim() - 2) + [replication_y + 1, replication_x + 1])
    )

    front = input[..., input.shape[-2] - pad_y :, :]
    padded_input = torch.cat([front, input], dim=-2)

    left = padded_input[..., :, padded_input.shape[-1] - pad_x :]
    padded_x = torch.cat([left, padded_input], dim=-1)

    return padded_x


class CircularPad2d(nn.Module):
    def __init__(self, pad: int | Sequence[int]):
        super().__init__()
        self.pad = pad

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return circular_pad2d(input, self.pad)


class EquivariantConv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.pad = CircularPad2d(kernel_size + (kernel_size - 1) * (dilation - 1) - 1)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class InvariantSpinModel(nn.Module):
    def __init__(
        self, lattice: ParallelogramSpinLattice, layers: nn.Module, fc: nn.Module
    ):
        super().__init__()
        self.lattice = lattice
        self.layers = layers
        self.fc = fc

    def forward(self, x: torch.Tensor):
        #        assert x.device.type == "cuda"

        x_tensor = self.lattice.spin_config_to_tensor(x).permute(0, 3, 1, 2)

        if isinstance(x_tensor, np.ndarray):
            x_tensor = x_tensor.astype(np.float32)
        else:
            x_tensor = x_tensor.float()

        # for i, layer in enumerate(self.layers):
        #     x_tensor = layer(x_tensor)

        x_tensor = self.layers(x_tensor)

        x_tensor = x_tensor.mean(dim=(2, 3))

        x_tensor = self.fc(x_tensor)

        return x_tensor


class InvariantSpinCNNRegression(InvariantSpinModel):
    def __init__(
        self,
        lattice: ParallelogramSpinLattice,
        hidden_channels: list[int],
        out_dim=1,
        dilations: list[int] | None = None,
        kernel_size=3,
        last_layer_bias=True,
    ):
        if dilations is None:
            dilations = [1] * len(hidden_channels)

        assert len(hidden_channels) == len(dilations)
        in_channels = lattice.spin_config_to_tensor(
            np.array([1], dtype=np.uint64)
        ).shape[-1]

        layers_list = []

        for hidden_ch, dilation in zip(hidden_channels, dilations):
            conv2d = EquivariantConv2d(in_channels, hidden_ch, kernel_size, dilation)
            layers_list.extend([conv2d, nn.ReLU()])
            in_channels = hidden_ch

        layers = nn.Sequential(*layers_list)

        fc = nn.Linear(hidden_channels[-1], out_dim, bias=last_layer_bias)
        super().__init__(lattice=lattice, layers=layers, fc=fc)


### FROM: https://stackoverflow.com/a/57233045/3025981
class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


### END FROM


class InvariantSpinCNNResRegression(InvariantSpinModel):
    def __init__(
        self,
        lattice: ParallelogramSpinLattice,
        hidden_channels: int,
        resnet_blocks: int,
        resnet_block_depth: int,
        out_dim=1,
        kernel_size=3,
    ):
        in_channels = lattice.spin_config_to_tensor(
            np.array([1], dtype=np.uint64)
        ).shape[-1]

        layers_list = [
            EquivariantConv2d(in_channels, hidden_channels, kernel_size, 1),
            nn.ReLU(),
        ]
        for _ in range(resnet_blocks):
            resnet_block_list = []
            for _ in range(resnet_block_depth):
                resnet_block_list.extend(
                    [
                        EquivariantConv2d(
                            hidden_channels, hidden_channels, kernel_size, 1
                        ),
                        nn.ReLU(),
                    ]
                )
            resnet_block = ResNet(nn.Sequential(*resnet_block_list))
            layers_list.append(resnet_block)

        layers = nn.Sequential(*layers_list)

        fc = nn.Linear(hidden_channels, out_dim, bias=True)
        super().__init__(lattice=lattice, layers=layers, fc=fc)
