import torch
import torch.nn as nn
import torch.nn.functional as F
from spin_lattices import KagomeLattice
import numpy as np


def conv2d_circular(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Apply circular padding
    if padding > 0:
        input = F.pad(input, (padding, 0, padding, 0), mode="circular")

    return F.conv2d(input, weight, bias, stride, 0, dilation, groups)


class CircularConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CircularConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.padding = padding

    def forward(self, x):
        return F.relu(
            conv2d_circular(
                x,
                self.conv.weight,
                self.conv.bias,
                self.conv.stride,
                self.padding,
                self.conv.dilation,
                self.conv.groups,
            )
        )


class KagomeCNNRegression(nn.Module):
    def __init__(
        self, lattice: KagomeLattice, hidden_channels1=32, hidden_channels2=64, kernel_size=2
    ):
        super().__init__()
        self.lattice = lattice
        self.conv1 = CircularConv2d(
            3, hidden_channels1, kernel_size=(kernel_size, kernel_size), padding=kernel_size
        )
        self.conv2 = CircularConv2d(
            hidden_channels1,
            hidden_channels2,
            kernel_size=(kernel_size, kernel_size),
            padding=kernel_size,
        )
        self.fc = nn.Linear(hidden_channels2, 1)  # Output a single value for regression

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        x = torch.from_numpy(self.lattice.spin_config_to_tensor(x).astype(np.float32)).permute(
            0, 3, 1, 2
        )
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(dim=(2, 3))  # Average over spatial dimensions
        x = self.fc(x)
        return x
