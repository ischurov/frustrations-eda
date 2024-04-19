import lzma
import pickle
from itertools import product
from pathlib import Path

from tqdm import tqdm

from boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    BooleanFourierAnalyzer,
    SignalOption,
    SignSignalKind,
    ValueSignalKind,
)
from spin_systems import HeisenbergJ1J2
from spin_lattices import KagomeLattice

batch_size = 1000

fourier_learners_cache_dir = Path("fourier_learners_cache")
ground_state_cache_dir = Path("groundstates")
experiment_dir = Path("experiments/kagome-24-fourier-full-2023-01-24")
experiment_dir.mkdir(parents=True, exist_ok=True)
accuracy_file = experiment_dir / f"acc-0.7-terms.csv"

if __name__ == "__main__":

    accuracy_file.write_text("height,J2,signal_kind,terms\n")

    for height, J2, signal_kind in product(
        [2, 3, 4],
        [
            0.0,
            0.8,
            0.3,
            0.9,
            0.5,
            0.7,
            1.0,
            0.6,
            0.4,
            0.2,
            0.1,
            0.55,
            0.65,
            0.75,
            0.85,
            0.95,
            0.52,
            0.522,
            0.524,
            0.526,
            0.528,
            0.53,
            0.532,
            0.534,
            0.536,
            0.538,
            0.54,
        ],
        [SignSignalKind(), AmplitudeMedianBinSignalKind()],
    ):
        analyzer = BooleanFourierAnalyzer(
            system=HeisenbergJ1J2(
                KagomeLattice(width=2, height=height),
                J1=1,
                J2=J2,
                use_symmetries=True,
                spin_inversion=1,
                ground_state_cache_dir=ground_state_cache_dir,
                show_progress=True,
            ),
            use_subset_symmetries=True,
            show_progress=True,
            cache_dir=fourier_learners_cache_dir,
        )
        train_set = analyzer.system.canonical_basis.states
        analyzer.fit(train_set, signal_opt=SignalOption(kind=signal_kind), batch_size=batch_size)
        terms = analyzer.how_many_terms_to_achieve_score(
            scorer="accuracy",
            target_score=0.7,
            max_terms=10002,
            step={12: 1, 18: 1, 24: 10}[analyzer.system.number_spins],
        )
        with open(accuracy_file, "a") as f:
            print(
                f"{height},{J2!r},{signal_kind.name},{terms}",
                file=f,
            )
