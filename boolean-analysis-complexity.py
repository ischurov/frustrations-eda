import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

import fire
import jsonlines
import lattice_symmetries as ls
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

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
from parity import popcount
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangleLattice
from utils import get_abslargest_terms, read_jsonl_to_df

ground_state_cache_dir = Path("groundstates")

outdir = Path("experiments/boolean-analysis-complexity")
outdir.mkdir(exist_ok=True, parents=True)

outfile = outdir / "complexity.jsonl"

scorers = ["sign_overlap", "accuracy", "f1"]
target_score = 0.8
target_scorer = "f1"


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
    LatticeOptions(TriangleLattice(width=4, height=4), use_symmetries=True, max_terms=None),
    LatticeOptions(TriangleLattice(width=4, height=5), use_symmetries=False, max_terms=2**11),
    LatticeOptions(TriangleLattice(width=4, height=6), use_symmetries=True, max_terms=2**15),
    LatticeOptions(KagomeLattice(width=2, height=2), use_symmetries=False, max_terms=None),
    LatticeOptions(KagomeLattice(width=2, height=3), use_symmetries=False, max_terms=None),
    LatticeOptions(KagomeLattice(width=2, height=4), use_symmetries=True, max_terms=2**16),
    LatticeOptions(SquareLattice(width=7, height=4), False, max_terms=2 ** (16 + 5 + 2)),
    LatticeOptions(TriangleLattice(width=7, height=4), False, max_terms=2 ** (16 + 5 + 2)),
    LatticeOptions(TriangleLattice(width=5, height=6), False, max_terms=2 ** (16 + 5 + 5)),
    LatticeOptions(KagomeLattice(width=2, height=5), False, max_terms=2 ** (16 + 5 + 5)),
    LatticeOptions(SquareLattice(width=5, height=6), False, max_terms=2 ** (16 + 5 + 5)),
]


# lattice_opts = [
#     LatticeOptions(
#         lattice=KagomeLattice(width=3, height=3), use_symmetries=False, max_terms=2 ** (16 + 5)
#     ),
# ]
name_to_lattice_opt = {lattice.lattice.get_cache_id(): lattice for lattice in lattice_opts}


signal_kinds: list[SignalKind] = [SignSignalKind(), AmplitudeMedianBinSignalKind()]

max_keep_terms = 10


def main(J2s: list[float]):
    logger.remove()
    logger.add(
        Path(
            f"logs/boolean-analysis-complexity-{','.join(map(str, J2s))}.log",
            level="DEBUG",
            colorize=True,
        )
    )

    for (lattice_opt, J2, signal_kind) in product(lattice_opts, J2s, signal_kinds):
        how_many_terms = None
        existing_row = {}
        success = "Error"

        if outfile.exists():
            with jsonlines.open(outfile) as reader:
                if (
                    existing_row := (
                        next(
                            (
                                row
                                for row in reader
                                if (
                                    row.get("lattice_name") == lattice_opt.lattice.get_cache_id()
                                    and row.get("J2") == J2
                                    and row.get("signal_kind") == signal_kind.name
                                    and row.get("target_score") == target_score
                                    and row.get("target_scorer") == target_scorer
                                    and row.get("success")
                                )
                            ),
                            {},
                        )
                    )
                ) != {}:
                    how_many_terms = existing_row["terms"]
                    success = True
                    logger.debug(
                        f"Found existing row with {how_many_terms} terms for lattice: {lattice_opt.lattice.get_cache_id()}, J2: {J2}, signal_kind: {signal_kind.name}"
                    )

        lattice, use_symmetries, max_terms = lattice_opt
        logger.debug(
            f"lattice: {lattice.get_cache_id()}, J2: {J2}, signal_kind: {signal_kind.name}"
        )
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

        if how_many_terms is None:
            success, how_many_terms, prediction = series.how_many_terms_to_achieve_score(
                target_score, scorer=target_scorer, max_terms=max_terms, orbitwise=False
            )
            scores = {
                scorer: series.prediction_score(scorer=scorer, prediction=prediction)[0]
                for scorer in scorers
            }
        else:
            scores = {}

        reprs = series.signal.lattice.get_fourier_repr().reprs
        coeffs = series.coeffs
        idxs, coeffs = get_abslargest_terms(coeffs[reprs], min(max_keep_terms, how_many_terms))
        subsets = reprs[idxs]

        with jsonlines.open(outfile, "a") as writer:
            new_row = (
                existing_row
                | {
                    "lattice_name": lattice.get_cache_id(),
                    "J2": J2,
                    "signal_kind": signal_kind.name,
                    "success": success,
                    "terms": how_many_terms,
                    "target_score": target_score,
                    "target_scorer": target_scorer,
                    "total_hamming_weight": series.total_hamming_weight(how_many_terms),
                    "number_spins": lattice.number_spins,
                    "largest_terms": [int(x) for x in subsets.tolist()],
                    "largest_coeffs": coeffs.tolist(),
                }
                | scores
            )
            logger.debug(f"Writing row: {new_row}")
            writer.write(new_row)
        logger.debug("Done")


if __name__ == "__main__":
    fire.Fire(main)
    logger.debug("End of script")
