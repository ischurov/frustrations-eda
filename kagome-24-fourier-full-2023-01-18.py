import lzma
import pickle
from itertools import product
from pathlib import Path

from tqdm import tqdm

from boolean_analysis import BooleanFourierAnalyser
from heisenberg_hamiltonians import HeisenbergJ1J2
from spin_lattices import KagomeLattice

batch_size = 1000
experiment_dir = Path("experiments/kagome24-2023-01-18")
experiment_dir.mkdir(exist_ok=True, parents=True)

for J2 in tqdm([0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.55, 0.65, 0.75, 0.85, 0.95]):
    pickle_to = f"learner-J2={J2!r}.pickle.lz"

    analyzer = BooleanFourierAnalyser(
        system=HeisenbergJ1J2(
            KagomeLattice(width=2, height=4),
            J1=1,
            J2=J2,
            use_symmetries=True,
            spin_inversion=1,
            ground_state_cache_dir=Path("groundstates"),
        ),
        use_subset_symmetries=True,
        show_progress=True,
    )
    train_set = analyzer.system.canonical_basis.states
    analyzer.fit(train_set, batch_size=batch_size)
    with lzma.open(experiment_dir / pickle_to, "wb") as f:
        pickle.dump(analyzer.learner, f)
