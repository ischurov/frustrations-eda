# %%
from vmc_ising import (
    output_dir,
    get_config,
    get_lattice,
    default_config,
    get_network,
    get_system,
    reconstruct_signs,
    find_sign_overlap,
)
from misc_utils import read_jsonl_to_df, keep_serializable, concat_columns, column_to
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nqs_playground_helpers import safe_exp
import torch
from pathlib import Path
from collections.abc import Iterable
import jsonlines
import pandas as pd
from vmc_ising import get_energy
import fire


def main(task_id: int, step: int):
    config = get_config(task_id)
    system = get_system(config)
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


if __name__ == "__main__":
    fire.Fire(main)
