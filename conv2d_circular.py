from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from spin_lattices import ParallelogramSpinLattice


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


class InvariantSpinCNNRegression(nn.Module):
    def __init__(
        self,
        lattice: ParallelogramSpinLattice,
        hidden_channels: list[int],
        out_dim=1,
        dilations: list[int] | None = None,
        kernel_size=3,
        last_layer_bias=True,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1] * len(hidden_channels)
        assert len(hidden_channels) == len(dilations)

        self.lattice = lattice

        layers = []
        in_channels = lattice.spin_config_to_tensor(
            np.array([1], dtype=np.uint64)
        ).shape[-1]

        for i, (hidden_ch, dilation) in enumerate(zip(hidden_channels, dilations)):
            reception_field = kernel_size + (kernel_size - 1) * (dilation - 1)
            layers.extend(
                [
                    CircularPad2d(reception_field - 1),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_ch,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = hidden_ch

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels[-1], out_dim, bias=last_layer_bias)

    def forward(self, x: torch.Tensor | npt.NDArray):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        x = torch.from_numpy(
            self.lattice.spin_config_to_tensor(x).astype(np.float32)
        ).permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x
