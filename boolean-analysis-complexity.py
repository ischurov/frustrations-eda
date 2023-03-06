import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

import fire
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


def mk_filename(row):
    return (
        "-".join(
            [
                f"{col}={row[col]}"
                for col in ["lattice_name", "J2", "signal_kind", "target_scorer", "target_score"]
            ]
        )
        + ".json"
    )


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
        success = "Error"

        lattice, use_symmetries, max_terms = lattice_opt
        logger.debug(
            f"lattice: {lattice.get_cache_id()}, J2: {J2}, signal_kind: {signal_kind.name}"
        )

        row = {
            "lattice_name": lattice.get_cache_id(),
            "J2": J2,
            "signal_kind": signal_kind.name,
            "target_score": target_score,
            "target_scorer": target_scorer,
        }

        outfile = outdir / mk_filename(row)

        if outfile.exists():
            row.update(json.loads(outfile.read_text()))
            logger.debug(f"File already exists, current row is {row}")

        if (
            row.get("success")
            and "terms" in row
            and "total_hamming_weight" in row
            and "largest_terms" in row
            and "largest_coeffs" in row
        ):
            logger.debug("Everything already done, skipping")
            continue

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

        if "terms" not in row or not row.get("sucess"):
            logger.debug("Finding number of terms to achieve target score")
            success, how_many_terms, prediction = series.how_many_terms_to_achieve_score(
                target_score, scorer=target_scorer, max_terms=max_terms, orbitwise=False
            )
            scores = {
                scorer: series.prediction_score(scorer=scorer, prediction=prediction)[0]
                for scorer in scorers
            }
            row["success"] = success
            row["terms"] = how_many_terms
            row["total_hamming_weight"] = series.total_hamming_weight(how_many_terms)

            row.update(scores)

            update_largest = True

        else:
            how_many_terms = row["terms"]
            success = row["success"]

            update_largest = False

        if update_largest or "largest_terms" not in row or "largest_coeffs" not in row:
            reprs = series.signal.lattice.get_fourier_repr().reprs
            coeffs = series.coeffs
            idxs, coeffs = get_abslargest_terms(coeffs[reprs], min(max_keep_terms, how_many_terms))
            subsets = reprs[idxs]

            row["largest_terms"] = [int(x) for x in subsets.tolist()]
            row["largest_coeffs"] = coeffs.tolist()

        logger.debug(f"Writing row: {row}")
        outfile.write_text(json.dumps(row, indent=4))

        logger.debug("Done")


if __name__ == "__main__":
    fire.Fire(main)
    logger.debug("End of script")
