import contextlib
import itertools
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from heisenberg_hamiltonians import HeisenbergJ1J2

# FROM: https://github.com/python/cpython/blob/b3722ca058f6a6d6505cf2ea9ffabaf7fb6b6e19/Lib/contextlib.py#L767-L779


class chdir(contextlib.AbstractContextManager):
    """Non thread-safe context manager to change the current working directory."""

    def __init__(self, path):
        self.path = path
        self._old_cwd = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        os.chdir(self._old_cwd.pop())


# END FROM


# BASED ON code by Nikita Astrakhantsev
# Modified by Ilya Schurov


def write_delim(f):
    f.write("--------------------\n")
    f.flush()
    return


VMC_OUT = [
    "/home/ischurov/nix-portable",
    "nix",
    "shell",
    "/vol/tcm10/ischurov/mVMC?submodules=1#default",
    "--command",
    "vmc.out",
]


class MVMCConfig:
    def __init__(
        self,
        system: HeisenbergJ1J2,
        total_spin: int,
        directory: Path | str | None = None,
        NSROptItrStep: int = 500,
        DSROptStepDt: float = 0.0200000000,
        NVMCSample: int = 1000,
        seed: int | None = None,
        in_files: bool = False,
    ):
        """
        Generates a configuration for the mVMC package. Can be used only with
        spin system in the sector of zero magnetization (hamming_weight = 0).

        See parameters description in the mVMC manual:
        https://issp-center-dev.github.io/mVMC/doc/master/en/expert.html#modpara-file-modpara-def
        """

        self.n_sites = system.number_spins
        self.n_symmetries = 1
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        if directory is None:
            self.temp_directory = tempfile.TemporaryDirectory()
            self.directory = Path(self.temp_directory.name)
        else:
            self.temp_directory = None
            self.directory = Path(directory)

        self.directory.mkdir(parents=True, exist_ok=True)
        self.edges = system.lattice.edges_to_kind
        self.j = {1: system.J1, 2: system.J2}
        self.rng = np.random.default_rng(seed)
        self.total_spin = total_spin
        self.NSROptItrStep = NSROptItrStep
        self.DSROptStepDt = DSROptStepDt
        self.NVMCSample = NVMCSample
        self.in_files = in_files

    def _write_modpara(self, NVMCCalMode: int) -> None:
        with open(self.directory / "modpara.def", "w") as f:
            write_delim(f)
            f.write("Model_Parameters   0\n")
            write_delim(f)
            f.write("VMC_Cal_Parameters\n")
            write_delim(f)
            f.write("CDataFileHead  zvo\n")
            f.write("CParaFileHead  zqp\n")
            write_delim(f)
            f.write(f"NVMCCalMode    {NVMCCalMode:d}\n")
            write_delim(f)
            f.write("NDataIdxStart  1\n")
            f.write("NDataQtySmp    1\n")
            write_delim(f)
            f.write("Nsite          {:d}\n".format(self.n_sites))
            f.write("Ncond          0\n")
            f.write("2Sz            0\n")
            f.write("NSPGaussLeg    8\n")
            f.write("NSPStot        {:d}\n".format(self.total_spin))
            f.write("NMPTrans       {:d}\n".format(self.n_symmetries))
            f.write(f"NSROptItrStep  {self.NSROptItrStep:d}\n")
            f.write("DSROptRedCut   0.0010000000\n")
            f.write("DSROptStaDel   0.0200000000\n")
            f.write(f"DSROptStepDt   {self.DSROptStepDt}\n")
            f.write(f"NVMCWarmUp      {self.NVMCSample // 10}\n")
            f.write("NVMCInterval   1\n")
            f.write(f"NVMCSample     {self.NVMCSample:d}\n")
            f.write("NExUpdatePath  2\n")
            f.write("RndSeed        {:d}\n".format(self.seed))
            f.write("NSplitSize     1\n")
            f.write("NExUpdatePath  2\n")
            f.write("NStore         1\n")
            f.write("NSRCG          0\n")

    def _write_locspn(self):
        with open(self.directory / "locspn.def", "w") as f:
            f.write("================================\n")
            f.write("NlocalSpin    {:d}\n".format(self.n_sites))
            f.write("================================\n")
            f.write("========i_0LocSpn_1IteElc ======\n")
            f.write("================================\n")
            for i in range(self.n_sites):
                f.write("    {:d}      1\n".format(i))

    def _write_trans(self):
        with open(self.directory / "trans.def", "w") as f:
            f.write("=======================\n")
            f.write("NTransfer       0\n")
            f.write("=======================\n")
            f.write("========i_j_s_tijs======\n")
            f.write("=======================\n")

    def _write_coulombinter(self):
        with open(self.directory / "coulombinter.def", "w") as f:
            f.write("=============================================\n")
            f.write(f"NCoulombInter         {len(self.edges):d}\n")
            f.write("=============================================\n")
            f.write("================== CoulombInter ================\n")
            f.write("=============================================\n")

            for (origin, destination), kind in self.edges.items():
                f.write(
                    "    {:d}     {:d}        {:.15f}\n".format(
                        origin, destination, -self.j[kind] / 4.0
                    )
                )

    def _write_hund(self):
        with open(self.directory / "hund.def", "w") as f:
            f.write("=============================================\n")
            f.write("NHund         {:d}\n".format(len(self.edges)))
            f.write("=============================================\n")
            f.write("================== Hund coupling ============\n")
            f.write("=============================================\n")

            for (origin, destination), kind in self.edges.items():
                f.write(
                    "    {:d}     {:d}        {:.15f}\n".format(
                        origin, destination, -self.j[kind] / 2.0
                    )
                )

    def _write_exchange(self):
        with open(self.directory / "exchange.def", "w") as f:
            f.write("=============================================\n")
            f.write("NExchange         {:d}\n".format(len(self.edges)))
            f.write("=============================================\n")
            f.write("================== ExchangeCoupling coupling ============\n")
            f.write("=============================================\n")

            for edge, kind in self.edges.items():
                f.write(
                    "    {:d}     {:d}        {:.15f}\n".format(
                        edge[0], edge[1], -self.j[kind] / 2.0
                    )
                )

    def _write_gutzwilleridx(self):
        with open(self.directory / "gutzwilleridx.def", "w") as f:
            f.write("=============================================\n")
            f.write("NGutzwillerIdx          1\n")
            f.write("ComplexType          0\n")
            f.write("=============================================\n")
            f.write("=============================================\n")

            for site in range(self.n_sites):
                f.write("    {:d}      {:d}\n".format(site, 0))

            for g in range(1):
                f.write("    {:d}      1\n".format(g))

    def _write_in_gutzwiller(self):
        with open(self.directory / "InGutzwiller.def", "w") as f:
            f.write("======================\n")
            f.write("NGutzwillerIdx  {:d}\n".format(1))
            f.write("======================\n")
            f.write("== i_j_GutzwillerIdx  ===\n")
            f.write("======================\n")
            for i in range(1):
                f.write("{:d} {:.20f}  {:.20}\n".format(i, self.rng.random(), 0.0))

    def _write_in_jastrow(self):
        with open(self.directory / "InJastrow.def", "w") as f:
            f.write("======================\n")
            f.write("NJastrowIdx  {:d}\n".format(1))
            f.write("======================\n")
            f.write("== i_j_JastrowIdx  ===\n")
            f.write("======================\n")
            for i in range(1):
                f.write("{:d} {:.20f}  {:.20}\n".format(i, self.rng.random(), 0.0))

    def _write_jastrowidx(self):
        with open(self.directory / "jastrowidx.def", "w") as f:
            f.write("=============================================\n")
            f.write("NJastrowIdx          {:d}\n".format(1))
            f.write("ComplexType          0\n")
            f.write("=============================================\n")
            f.write("=============================================\n")

            for site_i in range(self.n_sites):
                for site_j in range(self.n_sites):
                    if site_i == site_j:
                        continue
                    f.write("    {:d}      {:d}      {:d}\n".format(site_i, site_j, 0))
            for z in range(1):
                f.write("    {:d}      1\n".format(z))

    def _write_orbitalidx(self):
        n_orb = self.n_sites**2
        with open(self.directory / "orbitalidx.def", "w") as f:
            f.write("=============================================\n")
            f.write("NOrbitalIdx          {:d}\n".format(n_orb))
            f.write("ComplexType          0\n")
            f.write("=============================================\n")
            f.write("=============================================\n")
            for idx, (first, second) in enumerate(
                itertools.product(np.arange(self.n_sites), repeat=2)
            ):
                f.write("    {:d}      {:d}      {:d}\n".format(first, second, idx))

            for z in range(n_orb):
                f.write("    {:d}      1\n".format(z))

    def _write_in_orbital(self):
        n_orb = self.n_sites**2
        with open(self.directory / "InOrbital.def", "w") as f:
            f.write("======================\n")
            f.write("NOrbitalIdx  {:d}\n".format(n_orb))
            f.write("======================\n")
            f.write("== i_j_OrbitIdx  ===\n")
            f.write("======================\n")

            for i in range(n_orb):
                f.write("{:d} {:.20f}  {:.20}\n".format(i, self.rng.random(), 0.0))

    def _write_qptransidx(self):
        with open(self.directory / "qptransidx.def", "w") as f:
            f.write("=============================================\n")
            f.write("NQPTrans          {:d}\n".format(self.n_symmetries))
            f.write("=============================================\n")
            f.write("======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n")
            f.write("=============================================\n")
            f.write("{:d}    {:.6f} {:.6f}\n".format(0, 1.0, 0.0))
            for i in range(self.n_sites):
                f.write("    {:d}      {:d}      {:d}\n".format(0, i, i))

            # for i in range(len(all_symmetries)):
            #     f.write(
            #         "{:d}    {:.6f} {:.6f}\n".format(i, np.real(eigenvalues[i]), np.imag(eigenvalues[i]))
            #     )
            # for idx_symm, symm in enumerate(all_symmetries):
            #     for idx_from, idx_to in enumerate(symm):
            #         f.write("    {:d}      {:d}      {:d}\n".format(idx_symm, idx_from, idx_to))

    def _write_greenone(self):
        with open(self.directory / "greenone.def", "w") as f:
            f.write("===============================\n")
            f.write("NCisAjs          0\n")
            f.write("===============================\n")
            f.write("======== Green functions ======\n")
            f.write("===============================\n")

    def _write_greentwo(self):
        with open(self.directory / "greentwo.def", "w") as f:
            f.write("=============================================\n")
            f.write("NCisAjsCktAltDC        {:d}\n".format(4 * (self.n_sites) ** 2))
            f.write("=============================================\n")
            f.write("======== Green functions for Sq AND Nq ======\n")
            f.write("=============================================\n")

            for i, j, s1, s2 in itertools.product(
                range(self.n_sites), range(self.n_sites), range(2), range(2)
            ):
                f.write(
                    "    {:d}     {:d}     {:d}     {:d}     {:d}     {:d}     {:d}     {:d}\n".format(
                        i, s1, i, s1, j, s2, j, s2
                    )
                )

    def _write_namelist(self):
        with open(self.directory / "namelist.def", "w") as f:
            f.write("         ModPara  modpara.def\n")
            f.write("         LocSpin  locspn.def\n")
            f.write("           Trans  trans.def\n")
            f.write("    CoulombInter  coulombinter.def\n")
            f.write("            Hund  hund.def\n")
            f.write("        Exchange  exchange.def\n")
            f.write("        OneBodyG  greenone.def\n")
            f.write("        TwoBodyG  greentwo.def\n")
            f.write("      Gutzwiller  gutzwilleridx.def\n")
            f.write("         Jastrow  jastrowidx.def\n")
            f.write("         Orbital  orbitalidx.def\n")
            f.write("        TransSym  qptransidx.def\n")
            if self.in_files:
                f.write("       InJastrow  InJastrow.def\n")
                f.write("    InGutzwiller  InGutzwiller.def\n")
                f.write("       InOrbital  InOrbital.def\n")

    def write_configuration(self, NVMCCalMode: int):
        self._write_coulombinter()
        self._write_exchange()
        self._write_greenone()
        self._write_greentwo()
        self._write_gutzwilleridx()
        self._write_hund()
        self._write_jastrowidx()
        self._write_locspn()
        self._write_modpara(NVMCCalMode=NVMCCalMode)
        self._write_namelist()
        self._write_orbitalidx()
        self._write_qptransidx()
        self._write_trans()
        if self.in_files:
            self._write_in_gutzwiller()
            self._write_in_jastrow()
            self._write_in_orbital()

    def cleanup(self):
        if self.temp_directory is not None:
            self.temp_directory.cleanup()

    def do_monte_carlo_optimization(self):
        with chdir(self.directory):
            self.write_configuration(NVMCCalMode=0)
            subprocess.run(VMC_OUT + ["-e", "namelist.def"], check=True, encoding="utf-8")

    def extract_wavefunction(self, walk: bool = True):
        with chdir(self.directory):
            wavefunction_file = Path("./wavefunction.dat")
            if wavefunction_file.exists():
                wavefunction_file.unlink()
            self.write_configuration(NVMCCalMode=1)
            my_env = os.environ.copy()
            my_env["EXTRACT_WAVEFUNCTION"] = str(wavefunction_file)
            if walk:
                my_env["WALK"] = "1"

            subprocess.run(
                VMC_OUT + ["-e", "namelist.def", "output/zqp_opt.dat"],
                check=True,
                encoding="utf-8",
                env=my_env,
            )
            series = (
                pd.read_csv(
                    wavefunction_file,
                    sep="\t",
                    header=None,
                    names=["idx", "eigenstate_coeff"],
                    dtype={"idx": "uint64", "eigenstate_coeff": "float64"},
                )
                .set_index("idx")
                .assign(
                    eigenstate_coeff=lambda df: df["eigenstate_coeff"]
                    / np.sqrt((df["eigenstate_coeff"] ** 2).sum())
                )["eigenstate_coeff"]
                .sort_index()
            )

        return series

    def get_ground_state(self, walk : bool = True, keep_files: bool = False):
        try:
            self.do_monte_carlo_optimization()
            series = self.extract_wavefunction(walk=walk)
        finally:
            if not keep_files:
                self.cleanup()
        return series


# END BASED
