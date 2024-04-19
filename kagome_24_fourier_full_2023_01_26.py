import lzma
import pickle
from itertools import product
from pathlib import Path

from scipy.optimize import bisect
from tqdm import tqdm

from boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    BooleanFourierAnalyzer,
    SignalKind,
    SignalOption,
    SignSignalKind,
    ValueSignalKind,
)
from spin_systems import HeisenbergJ1J2
from spin_lattices import KagomeLattice

batch_size = 1000

fourier_learners_cache_dir = Path("fourier_learners_cache")
ground_state_cache_dir = Path("groundstates")
experiment_dir = Path("experiments/kagome-24-fourier-full-2023-01-26")
experiment_dir.mkdir(parents=True, exist_ok=True)
accuracy_file = experiment_dir / f"acc-0.7-terms.csv"
overlap_file = experiment_dir / f"overlap-0.95-terms.csv"


def get_terms(height: int, J2: float, signal_kind: SignalKind, lattices: dict[int, KagomeLattice]):
    print(f"height={height}, J2={J2!r}, signal_kind={signal_kind.name}")
    analyzer = BooleanFourierAnalyzer(
        system=HeisenbergJ1J2(
            lattice=lattices[height],
            J1=1,
            J2=J2,
            use_symmetries=True,
            spin_inversion=1,
            ground_state_cache_dir=ground_state_cache_dir,
            show_progress=False,
        ),
        use_subset_symmetries=True,
        show_progress=True,
        cache_dir=fourier_learners_cache_dir,
    )
    train_set = analyzer.system.canonical_basis.states
    analyzer.fit(train_set, signal_opt=SignalOption(kind=signal_kind), batch_size=batch_size)
    # terms, scores = analyzer.how_many_terms_to_achieve_score(
    #     scorer="accuracy",
    #     target_score=0.7,
    #     max_terms=10002,
    #     step={12: 1, 18: 1, 24: 10}[analyzer.system.number_spins],
    #     additional_scorers=["sign_overlap"],
    # )
    # with open(accuracy_file, "a") as f:
    #     print(
    #         f"{height},{J2!r},{signal_kind.name},{terms},{scores['accuracy']},"
    #         f"{scores['sign_overlap']}",
    #         file=f,
    #     )

    terms, scores = analyzer.how_many_terms_to_achieve_score(
        scorer="sign_overlap",
        target_score=0.95,
        max_terms=10002,
        step={12: 1, 18: 1, 24: 10}[analyzer.system.number_spins],
        additional_scorers=["accuracy"],
    )
    with open(overlap_file, "a") as f:
        print(
            f"{height},{J2!r},{signal_kind.name},{terms},{scores['accuracy']},"
            f"{scores['sign_overlap']}",
            file=f,
        )

    if terms is None:
        return 11
    return terms


if __name__ == "__main__":
    lattices = {height: KagomeLattice(width=2, height=height) for height in [2, 3, 4]}

    accuracy_file.write_text("height,J2,signal_kind,terms,accuracy,sign_overlap\n")
    overlap_file.write_text("height,J2,signal_kind,terms,accuracy,sign_overlap\n")
    J2_left = 0.5206
    J2_right = 0.5207
    bisect(lambda J2: get_terms(4, J2, SignSignalKind(), lattices) - 11, 0.5206, 0.5207)
    