# %%
from vmc_ising import (
    output_dir,
    get_config,
    get_network,
    get_system,
)
import numpy as np
import torch
from vmc_ising import get_energy
from spin_lattices import AllToAllLattice
from heisenberg_hamiltonians import HeisenbergJ1J2
import fire


def main(task_id: int, step: int):
    config = get_config(task_id)
    system = get_system(config)
    all_to_all_lattice = AllToAllLattice(system.lattice)
    full_spin_system = HeisenbergJ1J2(
        lattice=all_to_all_lattice,
        use_symmetries=system.use_symmetries,
        spin_inversion=system.spin_inversion,
    )
    system.get_eigenstates(1)
    log_prob_fn = get_network(config, get_system(config))

    # %%

    true_signs = np.sign(system.get_ground_state())
    log_prob_fn.load_state_dict(
        torch.load(
            output_dir / f"{task_id}/log_prob_fn_{step}.pt",
            map_location=torch.device("cpu"),
        )
    )

    # %%
    signs_greedy = torch.load(output_dir / f"{task_id}/reconstructed_signs_{step}.pt")

    # %%
    print(f"Energy true signs = {get_energy(true_signs, system, log_prob_fn)}")
    print(f"Energy greedy signs = {get_energy(signs_greedy, system, log_prob_fn)}")
    print(
        f"Full spin true signs = {get_energy(true_signs, full_spin_system, log_prob_fn)}"
    )
    print(
        f"Full spin greedy signs = {get_energy(signs_greedy, full_spin_system, log_prob_fn)}"
    )


if __name__ == "__main__":
    fire.Fire(main)
