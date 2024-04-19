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

for height, J2, signal_kind in product(
    [2, 3, 4],
    tqdm([0.0, 0.8, 0.3, 0.9, 0.5, 0.7, 1.0, 0.6, 0.4, 0.2, 0.1, 0.55, 0.65, 0.75, 0.85, 0.95]),
    [AmplitudeMedianBinSignalKind(), SignSignalKind()],
):
    analyzer = BooleanFourierAnalyzer(
        system=HeisenbergJ1J2(
            KagomeLattice(width=2, height=height),
            J1=1,
            J2=J2,
            use_symmetries=True,
            spin_inversion=1,
            ground_state_cache_dir=ground_state_cache_dir,
        ),
        use_subset_symmetries=True,
        show_progress=True,
        cache_dir=fourier_learners_cache_dir,
    )
    train_set = analyzer.system.canonical_basis.states
    analyzer.fit(train_set, signal_opt=SignalOption(kind=signal_kind), batch_size=batch_size)
