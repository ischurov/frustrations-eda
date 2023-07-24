import json
import shutil
from itertools import product
from pathlib import Path
from typing import Type

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from loguru import logger
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from fast_boolean_analysis import FourierSeries, fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    LBFFromNN,
    LBFFromSpinSystem,
    SignalKind,
    SignSignalKind,
)
from pytorchtools import EarlyStopping
from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangleLattice
from spin_nn import FC1SpinNN, SpinNN
from misc_utils import ensure_newfile, get_abslargest_terms, make_unpacked_configurations

self_name = Path(__file__).name


def mkdir(path: Path):
    #    if __name__ == "__main__" and path.exists():
    #        print(f"{path} already exists. Remove it? (y/n)")
    #        if input() == "y":
    #            # remove directory and all its contents
    #            shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


ground_state_cache_dir = Path("groundstates")

experiment_dir = mkdir(Path("experiments") / self_name.removesuffix(".py"))
logger.add(experiment_dir / "log.log", level="DEBUG", colorize=False)

nn_checkpoints_dir = mkdir(experiment_dir / "nn_checkpoints")

model_evaluation_dir = mkdir(experiment_dir / "model_evaluation")
nn_terms_dir = mkdir(experiment_dir / "nn-terms")
system_terms_dir = mkdir(experiment_dir / "system-terms")

lattices: list[SpinLattice] = [
    KagomeLattice(width=2, height=4),
    TriangleLattice(width=6, height=4),
]

signal_kinds = [SignSignalKind()]


eps_trains = [3e-1, 5e-3, 1e-2]
J2s: dict[Type[SpinLattice], list[float]] = {
    TriangleLattice: [
        1.0,
        1.1,
        1.2,
    ],
    KagomeLattice: [
        0.6,
        0.8,
        1.0,
    ],
}


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


networks = {
    "1hidden-64-relu": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
    "2hidden-64-relu": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(64, 64, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
    "1hidden-64-sigmoid": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
    "2hidden-64-sigmoid": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(64, 64, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
    "1hidden-64-sin": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        Sin(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
    "2hidden-64-sin": lambda system: nn.Sequential(
        nn.Linear(system.number_spins, 64, dtype=torch.float64),
        Sin(),
        nn.Linear(64, 64, dtype=torch.float64),
        Sin(),
        nn.Linear(64, 2, dtype=torch.float64),
    ),
}


val_eps = 5e-2
test_eps = 5e-2
epochs = 5000
dump_each_epoch = 10

batch_size = 1024

# # test run:
# eps_trains = [1e-3]
# J2s = {TriangleLattice: [0.0]}
# lattices = [TriangleLattice(width=6, height=4)]
# epochs = 50
# # end test run

f1scorer = BinaryF1Score()


def get_inputs_and_labels(
    df: pd.DataFrame, number_spins: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        make_unpacked_configurations(
            np.asarray(df.index, dtype="uint64"), number_spins=number_spins
        ).astype("float64"),
        dtype=torch.float64,
    )
    y = torch.tensor(df["y"].values.astype("int8"), dtype=torch.long)
    probs = torch.tensor(df["prob"].values.astype("float"), dtype=torch.float64)
    return X, y, probs


def train_net(
    net: nn.Module,
    batch_size: int,
    epochs: int,
    df_train: pd.DataFrame,
    criterion,
    optimizer,
    n_batches: int,
    number_spins: int,
    dump_each_epoch: int,
):
    epoch = 0
    logger.debug(f"{n_batches=}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        i = None
        loss = None

        net.train()
        for i in range(n_batches):
            data = df_train.iloc[i * batch_size : (i + 1) * batch_size]
            inputs, labels, probs = get_inputs_and_labels(data, number_spins=number_spins)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        assert loss is not None

        net.eval()
        if epoch % dump_each_epoch == 0:
            yield epoch, net, loss.item()


@torch.no_grad()
def evaluate(net, inputs, labels, probs):
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)

    sign_overlap = (((1 - 2 * predicted) * (1 - 2 * labels) * probs).sum() / probs.sum()).item()

    f1 = f1scorer(predicted, labels).item()
    return accuracy, sign_overlap, f1


def write_terms_to_file(
    file: Path,
    series: FourierSeries,
    params: dict | None = None,
    keep_terms: int = 100,
):
    if params is None:
        params = {}

    row = {}

    idxs, coeffs = get_abslargest_terms(series.coeffs, keep_terms)
    row["iipr"] = 1.0 / series.ipr()
    row["iipr_hamming"] = 1.0 / series.ipr(hamming_weighted=True)
    row["iipr_no_free_term"] = 1.0 / series.ipr(ignore_free_term=True)
    row["iipr_hamming_no_free_term"] = 1.0 / series.ipr(
        hamming_weighted=True, ignore_free_term=True
    )
    row["largest_terms"] = [int(x) for x in idxs.tolist()]
    row["largest_coeffs"] = coeffs.tolist()
    row["lattice"] = series.signal.lattice.get_cache_id()

    ensure_newfile(file).write_text(json.dumps(row | params))


def get_train_val_test(
    system: SpinSystem, signal_kind: SignalKind, eps_train: float, val_eps: float, test_eps: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = (
        system.get_df_ground_state(canonical_basis=True, add_amplitude=False).assign(
            y=(
                lambda df: (signal_kind.transform_data(df["eigenstate_coeff"].values) < 0).astype(
                    int
                )
            ),
            prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
        )
    )[["y", "prob"]]

    logger.debug(f"{eps_train=}, {val_eps=}, {test_eps=}")

    logger.debug("Making train, val, test splits")

    df_train = df.sample(frac=eps_train, weights="prob", replace=True)
    df_for_val = df.drop(df_train.index)
    df_val = df_for_val.sample(frac=val_eps, weights="prob", replace=True)
    df_for_test = df_for_val.drop(df_val.index)
    df_test = df_for_test.sample(frac=test_eps, weights="prob", replace=True)

    logger.debug(f"{df_train.shape=}, {df_val.shape=}, {df_test.shape=}")
    return df_train, df_val, df_test


def main():
    for lattice, signal_kind in product(lattices, signal_kinds):
        for J2 in J2s[lattice.__class__]:
            system = HeisenbergJ1J2(
                lattice=lattice,
                J1=1.0,
                J2=J2,
                ground_state_cache_dir=ground_state_cache_dir,
            )

            system.get_eigenstates(1)
            logger.debug(f"lattice={lattice.get_cache_id()}, {J2=} {signal_kind.name=}")
            logger.debug("Generating dataset")

            system_signal = LBFFromSpinSystem(system=system, eigenstate=0, kind=signal_kind)
            system_series = fourier_expand(system_signal)

            write_terms_to_file(
                file=system_terms_dir
                / f"{J2=}_lattice={lattice.get_cache_id()}_signal_kind={signal_kind.name}.json",
                series=system_series,
                params={"J2": J2},
            )

            for eps_train in eps_trains:
                df_train, df_val, df_test = get_train_val_test(
                    system, signal_kind, eps_train, val_eps, test_eps
                )

                logger.debug(f"{eps_train=}, {len(df_train)=}, {len(df_val)=}, {len(df_test)=}")
                n_batches = (len(df_train) + batch_size - 1) // batch_size
                for net_id, net_factory in networks.items():
                    net = net_factory(system)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

                    inputs_train, labels_train, probs_train = get_inputs_and_labels(
                        df_train, number_spins=system.number_spins
                    )

                    inputs_val, labels_val, probs_val = get_inputs_and_labels(
                        df_val, number_spins=system.number_spins
                    )

                    for epoch, net, loss in train_net(
                        net=net,
                        epochs=epochs,
                        df_train=df_train,
                        batch_size=batch_size,
                        dump_each_epoch=dump_each_epoch,
                        criterion=criterion,
                        optimizer=optimizer,
                        n_batches=n_batches,
                        number_spins=system.number_spins,
                    ):
                        model_path = ensure_newfile(
                            nn_checkpoints_dir
                            / f"{net_id}-sign-{system.get_cache_id()}_eps_train={eps_train}_signal_kind={signal_kind.name}_epoch={epoch}.pt"
                        )

                        logger.debug(f"Saving model to {model_path}")
                        torch.save(net.state_dict(), model_path)

                        accuracy_train, sign_overlap_train, f1_train = evaluate(
                            net, inputs_train, labels_train, probs_train
                        )

                        accuracy_val, sign_overlap_val, f1_val = evaluate(
                            net, inputs_val, labels_val, probs_val
                        )

                        logger.debug(
                            f"{epoch=}, {accuracy_train=}, {sign_overlap_train=}, {f1_train=}, {accuracy_val=}, {sign_overlap_val=}, {f1_val=}"
                        )

                        ensure_newfile(
                            model_evaluation_dir
                            / f"{net_id}-{J2=}_lattice={lattice.get_cache_id()}_{eps_train=}_signal_kind={signal_kind.name}_epoch={epoch}.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "J2": J2,
                                    "eps_train": eps_train,
                                    "accuracy_train": accuracy_train,
                                    "sign_overlap_train": sign_overlap_train,
                                    "f1_train": f1_train,
                                    "accuracy_val": accuracy_val,
                                    "sign_overlap_val": sign_overlap_val,
                                    "f1_val": f1_val,
                                    "train_loss": loss,
                                    "epoch": epoch,
                                    "lattice": lattice.get_cache_id(),
                                    "train_size": len(df_train),
                                    "path": str(model_path),
                                    "signal_kind": signal_kind.name,
                                    "net_id": net_id,
                                }
                            )
                        )

                        nn_signal = LBFFromNN(
                            lattice=lattice,
                            nn=net,
                            probs=system.get_df_ground_state(canonical_basis=True).assign(
                                prob=lambda df: np.abs(df["eigenstate_coeff"]) ** 2
                            )["prob"],
                            binarize=False,
                        )

                        # nn_series = fourier_expand(nn_signal)

                        # write_terms_to_file(
                        #     file=nn_terms_dir
                        #     / (
                        #         f"{net_id}-{J2=}_lattice={lattice.get_cache_id()}_{eps_train=}"
                        #         f"_signal_kind={signal_kind.name}_epoch={epoch}.json"
                        #     ),
                        #     series=nn_series,
                        #     params={
                        #         "J2": J2,
                        #         "eps_train": eps_train,
                        #         "epoch": epoch,
                        #         "binarized": False,
                        #         "net_id": net_id,
                        #     },
                        # )

                        median_score = float(np.median(nn_signal(system.canonical_basis.states)))

                        nn_signal_binarized = LBFFromNN(
                            lattice=lattice,
                            nn=net,
                            probs=system.get_df_ground_state(canonical_basis=True).assign(
                                prob=lambda df: np.abs(df["eigenstate_coeff"]) ** 2
                            )["prob"],
                            binarize=True,
                            binarization_threshold=median_score,
                        )
                        nn_series_binarized = fourier_expand(nn_signal_binarized)

                        write_terms_to_file(
                            file=nn_terms_dir
                            / (
                                f"{net_id}-{J2=}_lattice={lattice.get_cache_id()}_{eps_train=}"
                                f"_signal_kind={signal_kind.name}_epoch={epoch}_binarized_adj.json"
                            ),
                            series=nn_series_binarized,
                            params={
                                "J2": J2,
                                "eps_train": eps_train,
                                "epoch": epoch,
                                "binarized": True,
                                "net_id": net_id,
                            },
                        )


if __name__ == "__main__":
    main()
