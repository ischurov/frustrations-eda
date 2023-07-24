import json
import shutil
from itertools import product
from pathlib import Path
from typing import Type

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from fast_boolean_analysis import FourierSeries, fourier_expand
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LBFFromNN,
    LBFFromSpinSystem,
    SignalKind,
    SignSignalKind,
)
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangleLattice
from misc_utils import ensure_newfile, get_abslargest_terms, make_unpacked_configurations

self_name = Path(__file__).name

target_rel_weight = 0.16


def mkdir(path: Path):
    if __name__ == "__main__" and path.exists():
        print(f"{path} already exists. Remove it? (y/n)")
        if input() == "y":
            # remove directory and all its contents
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"{path} already exists")

    path.mkdir(parents=True, exist_ok=__name__ != "__main__")
    return path


ground_state_cache_dir = Path("groundstates")

experiment_dir = mkdir(Path("experiments") / self_name.removesuffix(".py"))
logger.add(experiment_dir / "log.log", level="DEBUG", colorize=False)

lattices: list[SpinLattice] = [
    SquareLattice(width=6, height=4),
    TriangleLattice(width=6, height=4),
    KagomeLattice(width=2, height=4),
    SquareLattice(width=5, height=5),
    TriangleLattice(width=5, height=5),
    KagomeLattice(width=3, height=3),
]

signal_kinds = [SignSignalKind(), AmplitudeSignalKind(), AmplitudeMedianBinSignalKind()]

J2s: dict[Type[SpinLattice], list[float]] = {
    TriangleLattice: [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
    ],
    SquareLattice: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    KagomeLattice: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
if __name__ == "__main__":
    for lattice, signal_kind in product(lattices, signal_kinds):
        J2s_for_lattice = J2s[type(lattice)]
        for J2 in tqdm(J2s_for_lattice):
            system = HeisenbergJ1J2(
                lattice=lattice, J1=1.0, J2=J2, ground_state_cache_dir=ground_state_cache_dir
            )
            system.get_eigenstates(1)
            signal = LBFFromSpinSystem(system, kind=signal_kind)
            series = fourier_expand(signal)
            terms = series.how_many_terms_to_achieve_relative_weight(target_rel_weight)
            subsets, coeffs = get_abslargest_terms(series.coeffs, terms)
            row = {
                "J2": J2,
                "lattice": lattice.get_cache_id(),
                "signal_kind": signal_kind.name,
                "rel_weight_terms": terms,
                "rel_weight_target": target_rel_weight,
                "rel_weight_total_hamming_weight": series.total_hamming_weight(terms),
                "rel_weight_subsets": [int(x) for x in subsets.tolist()],
                "rel_weight_coeffs": coeffs.tolist(),
                "ipr": series.ipr(),
                "ipr_hamming": series.ipr(hamming_weighted=True),
            }
            ensure_newfile(experiment_dir / f"{J2=}_{lattice=}_{signal_kind=}.json").write_text(
                json.dumps(row, indent=4)
            )
