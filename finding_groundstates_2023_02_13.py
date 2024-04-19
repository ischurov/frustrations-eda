from pathlib import Path

import fire
from loguru import logger

from spin_systems import HeisenbergJ1J2, SpinSystem
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangularLattice

ground_state_cache_dir = Path("groundstates")

lattices = [
    (KagomeLattice(width=3, height=3), False),
    (KagomeLattice(width=2, height=5), False),
    (SquareLattice(width=5, height=6), False),
    (SquareLattice(width=7, height=4), False),
    (TriangularLattice(width=5, height=6), False),
    (TriangularLattice(width=7, height=4), False),
]  # type: list[tuple[SpinLattice, bool]]


def main(J2s: list[float]):
    logger.add(
        Path(
            "logs/"
            + __file__.split("/")[-1].removesuffix(".py")
            + "-"
            + "-".join(map(str, J2s))
            + ".log"
        ),
        level="DEBUG",
        colorize=False,
    )

    for lattice, use_symmetries in lattices:
        logger.debug(f"lattice: {lattice.get_cache_id()}, use_symmetries: {use_symmetries}")
        for J2 in J2s:
            system = HeisenbergJ1J2(
                lattice,
                J1=1.0,
                J2=J2,
                use_symmetries=use_symmetries,
                spin_inversion=1 if use_symmetries else None,
                ground_state_cache_dir=ground_state_cache_dir,
            )
            system.get_eigenstates(1)


if __name__ == "__main__":
    fire.Fire(main)
