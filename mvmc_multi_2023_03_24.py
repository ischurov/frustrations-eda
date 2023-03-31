from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fast_boolean_analysis import fourier_expand
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromEigenstateSeries
from mvmc_configs import MVMCConfig
from spin_lattices import (
    ChainLattice,
    KagomeLattice,
    SpinLattice,
    SquareLattice,
    TriangleLattice,
)
from utils import get_abslargest_terms

script_name = Path(__file__).stem

outdir = Path("experiments") / script_name.removesuffix(".py")
outdir.mkdir(exist_ok=True, parents=True)

lattices: list[SpinLattice] = [SquareLattice(6, 4), TriangleLattice(6, 4), KagomeLattice(2, 4)]
J2s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
mvmc_config = dict(NSROptItrStep=300)
iterations = 10


def get_sign_complexity(system, wavefunction):
    signal = LBFFromEigenstateSeries(system.lattice, np.sign(wavefunction))
    series = fourier_expand(signal)
    iipr = 1 / series.ipr(ignore_free_term=True)
    success, terms, _ = series.how_many_terms_to_achieve_score(0.8, "accuracy")
    total_hamming_weight = series.total_hamming_weight(terms)
    subsets, coeffs = get_abslargest_terms(series.coeffs, terms)
    return {
        "iipr": iipr,
        "terms": terms,
        "success": success,
        "total_hamming_weight": total_hamming_weight,
        "subsets": subsets,
        "coeffs": coeffs,
    }


def main():
    for lattice, J2 in product(lattices, J2s):
        rows = []
        system = HeisenbergJ1J2(lattice, J1=1, J2=J2, ground_state_cache_dir=Path("groundstates"))
        ed_energy, _ = system.get_eigenstates(1)
        ed_series = system.get_df_ground_state(canonical_basis=True)[
            "eigenstate_coeff"
        ].sort_index()
        ed_wavefunction = np.asarray(ed_series.values)
        ed_sign_complexity = get_sign_complexity(system, ed_series)
        rows.append(
            {
                "J2": J2,
                "energy": ed_energy[0],
                "type": "ed",
            }
            | ed_sign_complexity
        )
        for iteration in range(iterations):
            mvmc = MVMCConfig(system=system, total_spin=0, **mvmc_config)
            mvmc_series, mvmc_energy = mvmc.get_ground_state(report_energy=True)
            mvmc_energy = np.asarray(mvmc_energy)
            mvmc_wavefunction = np.asarray(mvmc_series.values)
            assert (mvmc_series.index == ed_series.index).all()
            overlap = (
                (ed_wavefunction * mvmc_wavefunction).sum()
                / np.linalg.norm(mvmc_wavefunction)
                / np.linalg.norm(ed_wavefunction)
            )
            mvmc_sign_complexity = get_sign_complexity(system, mvmc_series)
            rows.append(
                {
                    "J2": J2,
                    "overlap": np.real_if_close(overlap),
                    "iteration": iteration,
                    "energy": mvmc_energy[-1] * 4,
                    "energy_full": mvmc_energy * 4,
                    "type": "mvmc",
                    "iteration": iteration,
                }
                | mvmc_sign_complexity
            )
        df = pd.DataFrame(rows).assign(overlap=lambda df: df["overlap"].astype(np.float64))
        outfile = outdir / f"{lattice.get_cache_id()}_J2={J2:.2f}.feather"
        if outfile.exists():
            print(f"======= !!!! ERROR. File {outfile} already exists.")
        df.to_feather(outfile)


if __name__ == "__main__":
    main()
