import json
import re
import sys
from dataclasses import astuple, dataclass
from itertools import product
from pathlib import Path
from typing import NamedTuple

import lattice_symmetries as ls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import parse
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from fast_boolean_analysis import (
    FourierSeries,
    ScorerType,
    fourier_expand,
    get_scorer,
    keep_everything,
    keep_largest_n,
)
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LBFFromSpinSystem,
    SignalKind,
    SignSignalKind,
)
from misc_utils import get_abslargest_terms, read_jsonl_to_df
from parity import popcount
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangularLattice

ground_state_cache_dir = Path("groundstates")

outdir = Path("experiments/boolean-analysis-complexity")


def get_score_column(df: pd.DataFrame, scorer: str) -> list:
    return [
        row["series"].truncate(keep_largest_n(row["terms"])).prediction_score(scorer=scorer)
        for _, row in (df.iterrows())
    ]


logger.add(Path("add_weight.log", level="DEBUG", colorize=True))


class LatticeOptions(NamedTuple):
    lattice: SpinLattice
    use_symmetries: bool
    max_terms: int | None


lattice_opts = [
    LatticeOptions(lattice=SquareLattice(width=4, height=4), use_symmetries=True, max_terms=None),
    LatticeOptions(
        lattice=SquareLattice(width=4, height=5), use_symmetries=False, max_terms=2**11
    ),
    LatticeOptions(
        lattice=SquareLattice(width=4, height=6), use_symmetries=True, max_terms=2**15
    ),
    LatticeOptions(TriangularLattice(width=4, height=4), use_symmetries=True, max_terms=None),
    LatticeOptions(TriangularLattice(width=4, height=5), use_symmetries=False, max_terms=2**11),
    LatticeOptions(TriangularLattice(width=4, height=6), use_symmetries=True, max_terms=2**15),
    LatticeOptions(KagomeLattice(width=2, height=2), use_symmetries=False, max_terms=None),
    LatticeOptions(KagomeLattice(width=2, height=3), use_symmetries=False, max_terms=None),
    LatticeOptions(KagomeLattice(width=2, height=4), use_symmetries=True, max_terms=2**16),
    LatticeOptions(SquareLattice(width=7, height=4), False, max_terms=2 ** (16 + 5 + 2)),
    LatticeOptions(TriangularLattice(width=7, height=4), False, max_terms=2 ** (16 + 5 + 2)),
    LatticeOptions(TriangularLattice(width=5, height=6), False, max_terms=2 ** (16 + 5 + 5)),
    LatticeOptions(KagomeLattice(width=2, height=5), False, max_terms=2 ** (16 + 5 + 5)),
    LatticeOptions(SquareLattice(width=5, height=6), False, max_terms=2 ** (16 + 5 + 5)),
    LatticeOptions(KagomeLattice(width=3, height=3), False, max_terms=2 ** (16 + 5)),
]

name_to_lat_and_sym = {
    lat_opt.lattice.get_cache_id(): (lat_opt.lattice, lat_opt.use_symmetries)
    for lat_opt in lattice_opts
}

name_to_signal_kind = {
    signal.name: signal for signal in [AmplitudeMedianBinSignalKind(), SignSignalKind()]
}

if __name__ == "__main__":
    for file in outdir.glob("*.json"):
        row = json.loads(file.read_text())
        logger.debug(f"Processing {file}, row: {row}")

        if (terms := row.get("terms")) is None:
            logger.debug(f"Skipping {file} because it has no terms")
            continue

        if "total_fourier_weight" in row and "terms_fourier_weight" in row:
            logger.debug(f"Skipping {file} because it already has a weight")
            continue

        J2 = row["J2"]
        lattice, use_symmetries = name_to_lat_and_sym[row["lattice_name"]]

        if lattice.number_spins >= 31:
            logger.debug(f"Skipping {file} because it has too many spins")
            continue

        signal_kind = name_to_signal_kind[row["signal_kind"]]

        system = HeisenbergJ1J2(
            lattice=lattice,
            J1=1,
            J2=J2,
            use_symmetries=use_symmetries,
            spin_inversion=1 if use_symmetries else None,
            ground_state_cache_dir=ground_state_cache_dir,
        )
        logger.debug("Finding ground state")
        system.get_eigenstates(1)
        logger.debug(f"Finding fourier expansion")
        series = fourier_expand(LBFFromSpinSystem(system=system, eigenstate=0, kind=signal_kind))
        _, largest_coeffs = get_abslargest_terms(series.coeffs, terms)
        row["total_fourier_weight"] = np.sum(series.coeffs**2)
        row["terms_fourier_weight"] = np.sum(largest_coeffs**2)
        file.write_text(json.dumps(row))
    logger.debug("End of script")
