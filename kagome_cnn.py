import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv2d_circular import CircularConv2d
from spin_lattices import KagomeLattice


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
