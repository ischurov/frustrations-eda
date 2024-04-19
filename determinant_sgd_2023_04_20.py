# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from spin_systems import HeisenbergJ1J2
from slater_determinant import SlaterDeterminant
from spin_lattices import KagomeLattice, SquareLattice1Diag, TriangularLattice


def sign_overlap(ground_state, predict_signs):
    probs = ground_state**2
    return torch.dot(ground_state, predict_signs * torch.abs(ground_state)) / probs.sum()


# %%
# Writer will output to ./runs/ directory by default
writer = SummaryWriter(
    log_dir=f"experiments/2021_04_18/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
)


# %%

lattice = KagomeLattice(2, 4)
system = HeisenbergJ1J2(lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
system.get_eigenstates(1)

det = SlaterDeterminant(system.canonical_basis, sign_cache_dir=Path("signs_cache"))

ground_state_np = np.real_if_close(system.get_ground_state_in_canonical_basis())
ground_state = torch.from_numpy(ground_state_np)

# %%
# with torch.no_grad():
#     det.f.copy_(
#         nn.Parameter(
#             torch.randn(system.number_spins, system.number_spins, dtype=torch.float64)
#             / np.sqrt(system.number_spins)
#         )
#     )

eps_train = 0.001

test_size = 10000

epochs = 100000


train_set_numpy = np.random.choice(
    len(system.canonical_basis.states),
    int(eps_train * len(system.canonical_basis.states)),
    replace=False,
    p=ground_state**2,
)

train_set = torch.from_numpy(train_set_numpy)
logger.debug(f"{len(train_set)=}")

target = (ground_state[train_set] > 0).double()


# %%
rest_set_np = np.setdiff1d(np.arange(len(system.canonical_basis.states)), train_set_numpy)
rest_probs = ground_state_np[rest_set_np] ** 2
rest_probs /= rest_probs.sum()
test_set = torch.from_numpy(np.random.choice(rest_set_np, test_size, replace=False))

# %%
batch_size = 64

n_batches = len(train_set) // batch_size


criterion = nn.BCELoss()

optimizer = torch.optim.Adam(det.parameters(), lr=1e-2)

epoch = 0
logger.debug(f"{n_batches=}")
log = []
for epoch in range(epochs):  # loop over the dataset multiple times
    i = None
    loss = None

    for i in range(n_batches):
        x = train_set[i * batch_size : (i + 1) * batch_size]
        y = target[i * batch_size : (i + 1) * batch_size]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = torch.sigmoid(det(x))
        #        print(det(x))
        loss = criterion(outputs, y)
        loss.backward()
        #        print(det.f.grad.norm().item())

        optimizer.step()

    assert loss is not None
    overlap_train = sign_overlap(ground_state[train_set], torch.sign(det(train_set)))
    overlap_test = sign_overlap(ground_state[test_set], torch.sign(det(test_set)))
    # log.append(
    #     {
    #         "epoch": epoch,
    #         "loss": loss.item(),
    #         "overlap_train": overlap_train.item(),
    #         "overlap_test": overlap_test.item(),
    #     }
    # )
    # logger.debug(
    #     f"Epoch {epoch} loss: {loss.item():.4f} overlap_train: {overlap_train.item():.4f} "
    #     f"overlap_test: {overlap_test.item():.4f}"
    # )
    writer.add_scalar("Loss/train", loss.item(), epoch)
    writer.add_scalar("Overlap/train", overlap_train.item(), epoch)
    writer.add_scalar("Overlap/test", overlap_test.item(), epoch)

# %%
loss.item()

# %%
overlap_train.item()

# %%
overlap_test.item()

# %%
