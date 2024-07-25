from functools import reduce
import operator
from fourier_supervised_cleanroom_2023_09_27 import get_lattice
import numpy as np
from spin_systems import heisenberg, spin_system, no_symmetries_basis
from misc_utils import (
    kronecker_power_pytorch,
    rotation_matrix,
    eigenstate_in_full_basis,
)
import torch
import torch.nn as nn
from misc_utils import differentiable_safe_exp
from rotations_fixedpoints_configs import default_config, configs
from loguru import logger
from pathlib import Path
import jsonlines
import fire

self_name = Path(__file__).stem
output_dir = Path("experiments") / self_name


class WavefunctionFromAmplitudes(nn.Module):
    def __init__(self, log_amplitudes: torch.Tensor, signs: torch.Tensor):
        super().__init__()
        self.log_amplitudes = nn.Parameter(log_amplitudes)
        self.signs = signs

    def forward(self):
        amplitudes = differentiable_safe_exp(self.log_amplitudes)
        amplitudes = amplitudes / torch.norm(amplitudes)
        return self.signs * amplitudes


class RotationLoss(nn.Module):
    def __init__(self, angle: float):
        super().__init__()
        self.angle = angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float64)
        rotated_x = kronecker_power_pytorch(
            x, torch.tensor(rotation_matrix(self.angle)).to(torch.float64).to(x.device)
        )
        return ((x - rotated_x) ** 2).sum()


def get_system(config: dict):
    lattice = get_lattice(config["system.lattice"])
    return spin_system(
        heisenberg(lattice, J2=config["system.J2"]), no_symmetries_basis()
    )


def main(task_id: int):
    config = default_config | configs[task_id]
    system = get_system(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = WavefunctionFromAmplitudes(
        torch.zeros(system.ground_state.shape).to(device),
        torch.from_numpy(np.sign(system.ground_state)).to(device),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config["optimization.lr"])
    criterions = [RotationLoss(angle) for angle in config["angles"]]

    test_criterions = [RotationLoss(angle) for angle in config["test_angles"]]

    for step in range(config["optimization.max_steps"]):
        optimizer.zero_grad()
        wavefunction = network.forward()
        overlap = wavefunction.detach().cpu().numpy() @ system.ground_state
        wavefunction_fullbasis = eigenstate_in_full_basis(wavefunction, system.basis)
        loss = reduce(
            operator.add,
            (criterion(wavefunction_fullbasis) for criterion in criterions),
        )
        loss.backward()
        optimizer.step()

        (output_dir / f"{task_id}").mkdir(parents=True, exist_ok=True)

        test_losses = {
            f"test_loss.{angle}": criterion(wavefunction_fullbasis).item()
            for angle, criterion in zip(config["test_angles"], test_criterions)
        }

        with jsonlines.open(output_dir / f"{task_id}" / "results.jsonl", "a") as writer:
            writer.write(
                {
                    "step": step,
                    "loss": loss.item(),
                    "overlap": overlap,
                    # "wavefunction": wavefunction.detach().cpu().numpy().tolist(),
                }
                | test_losses
                | config
            )
        logger.info(f"step {step}, loss {loss.item()}, overlap {overlap}")


if __name__ == "__main__":
    fire.Fire(main)
