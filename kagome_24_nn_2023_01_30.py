import lzma
import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from scipy.optimize import bisect
from tqdm import tqdm

from spin_systems import HeisenbergJ1J2
from lattice_boolean_analysis import (
    AmplitudeMedianBinSignalKind,
    AmplitudeSignalKind,
    LatticeBooleanAnalyzer,
    LBFFromNN,
    LBFFromSpinSystem,
    SignalKind,
    SignSignalKind,
    ValueSignalKind,
)
from pytorchtools import EarlyStopping
from spin_lattices import KagomeLattice, SpinLattice
from spin_nn import FC1SpinNN, SpinNN
from misc_utils import make_unpacked_configurations

self_name = os.path.basename(__file__)

batch_size = 1000

fourier_learners_cache_dir = Path("fourier_learners_cache")
ground_state_cache_dir = Path("groundstates")
experiment_dir = Path("experiments") / self_name.removesuffix(".py")
experiment_dir.mkdir(parents=True, exist_ok=True)

nn_checkpoints_dir = experiment_dir / "nn_checkpoints"
nn_checkpoints_dir.mkdir(parents=True, exist_ok=True)


nn_accuracy_file = experiment_dir / f"nn_acc-0.7-terms.csv"
nn_overlap_file = experiment_dir / f"nn_overlap-0.95-terms.csv"

system_accuracy_file = experiment_dir / f"system_acc-0.7-terms.csv"
system_overlap_file = experiment_dir / f"system_overlap-0.95-terms.csv"

model_evaluation_file = experiment_dir / f"model_evaluation.csv"

J2s = [0.0, 0.4, 0.6, 0.8, 1.0]
lattice = KagomeLattice(width=2, height=4)
signal_kind = SignSignalKind()

eps_trains = [1e-3, 1e-2]
val_eps = 5e-2
test_eps = 5e-2
epochs = 20000
patience = 1000
delta = 0.01


def get_inputs_and_labels(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        make_unpacked_configurations(
            np.asarray(df.index, dtype="uint64"), number_spins=system.lattice.number_spins
        ).astype("float32")
    )
    y = torch.tensor(df["y"].values.astype("int8"), dtype=torch.long)
    probs = torch.tensor(df["prob"].values.astype("float"), dtype=torch.float32)
    return X, y, probs


def train_net(net: nn.Module, epochs: int, early_stopping: EarlyStopping):
    for epoch in range(epochs):  # loop over the dataset multiple times
        i = None
        loss = None

        net.train()
        for i in range(n_batches):
            data = df_train.iloc[i * batch_size : (i + 1) * batch_size]
            inputs, labels, probs = get_inputs_and_labels(data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
        net.eval()

        accuracy, sign_overlap = evaluate(net, inputs, labels, probs)  # type: ignore

        accuracy_val, sign_overlap_val = evaluate(net, inputs_val, labels_val, probs_val)

        early_stopping(-sign_overlap_val, net)

        if epoch % 100 == 0 or early_stopping.early_stop:
            print(f"[{epoch + 1}, {i}] loss: {loss}")
            print(f"Test set: accuracy: {100 * accuracy} %, sign overlap: {sign_overlap}")
            print(
                f"Validation set: accuracy: {100 * accuracy_val} %, sign overlap: {sign_overlap_val}"
            )

        if early_stopping.early_stop:
            print("Early stopping")
            break

    net.load_state_dict(torch.load(early_stopping.path))
    return net


def evaluate(net, inputs, labels, probs):
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        sign_overlap = (
            ((1 - 2 * predicted) * (1 - 2 * labels) * probs).sum() / probs.sum()
        ).item()
        return accuracy, sign_overlap


def write_terms_to_file(file, analyzer, J2, scorer, additional_scorers, target_score):
    terms, scores = analyzer.how_many_terms_to_achieve_score(
        scorer=scorer,
        target_score=target_score,
        max_terms=10002,
        step=10,
        additional_scorers=additional_scorers,
    )
    with open(file, "a") as f:
        print(
            f"{J2!r},{terms},{scores['accuracy']}," f"{scores['sign_overlap']}",
            file=f,
        )


if __name__ == "__main__":
    system_accuracy_file.write_text("J2,terms,accuracy,sign_overlap\n")
    system_overlap_file.write_text("J2,terms,accuracy,sign_overlap\n")
    nn_accuracy_file.write_text("J2,terms,accuracy,sign_overlap\n")
    nn_overlap_file.write_text("J2,terms,accuracy,sign_overlap\n")

    model_evaluation_file.write_text("J2,eps_train,test_accuracy,test_sign_overlap\n")

    for J2 in J2s:
        system = HeisenbergJ1J2(
            lattice=lattice,
            J1=1,
            J2=J2,
            use_symmetries=True,
            spin_inversion=1,
            ground_state_cache_dir=ground_state_cache_dir,
            show_progress=True,
        )
        system.get_eigenstates(1)

        df = (
            system.get_df_ground_state(
                canonical_basis=True,
            )
            .assign(
                sign=(lambda df: np.sign(df["eigenstate_coeff"])),
                prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
            )
            .assign(y=lambda df: (df["sign"] == 1).astype(int))
        )
        df_rep = (
            df.join(
                system.lattice.get_state_info_df(hamming_weight=system.lattice.number_spins // 2)
            )
            .groupby("representative")
            .agg({"sign": "mean", "prob": "sum", "y": "mean"})
        )

        for eps_train in eps_trains:
            batch_size = 64

            df_train = df_rep.sample(frac=eps_train, weights="prob")
            df_rep_for_val = df_rep.drop(df_train.index)
            df_val = df_rep_for_val.sample(frac=val_eps, weights="prob")
            df_rep_for_test = df_rep_for_val.drop(df_val.index)
            df_test = df_rep_for_test.sample(frac=test_eps, weights="prob")

            n_batches = int(np.ceil(len(df_train) / batch_size))

            net = FC1SpinNN(lattice=system.lattice, hidden_size=64)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

            inputs_val, labels_val, probs_val = get_inputs_and_labels(df_val)

            model_path = (
                nn_checkpoints_dir
                / f"FC1SpinNN-1hidden-64-sign-{system.get_cache_id()}_eps_train={eps_train}.pt"
            )
            early_stopping = EarlyStopping(
                patience=patience, delta=delta, verbose=False, path=str(model_path)
            )
            net = train_net(net, epochs, early_stopping)

            inputs_test, labels_test, probs_test = get_inputs_and_labels(df_test)
            accuracy_test, sign_overlap_test = evaluate(net, inputs_test, labels_test, probs_test)

            print(
                f"Test set: accuracy: {100 * accuracy_test} %, sign overlap: {sign_overlap_test}"
            )
            with open(model_evaluation_file, "a") as f:
                print(
                    f"{J2!r},{eps_train},{accuracy_test},{sign_overlap_test}",
                    file=f,
                )

            nn_signal = LBFFromNN(
                lattice=lattice,
                nn=net,
                probs=system.get_df_ground_state(canonical_basis=True).assign(
                    prob=lambda df: np.abs(df["eigenstate_coeff"]) ** 2
                )["prob"],
                network_id=str(model_path).replace("/", "_"),
            )

            nn_analyzer = LatticeBooleanAnalyzer(
                signal=nn_signal, show_progress=True, cache_dir=fourier_learners_cache_dir
            )
            nn_analyzer.fit()

            write_terms_to_file(
                file=nn_accuracy_file,
                analyzer=nn_analyzer,
                J2=J2,
                scorer="accuracy",
                additional_scorers=["sign_overlap"],
                target_score=0.7,
            )
            write_terms_to_file(
                file=nn_overlap_file,
                analyzer=nn_analyzer,
                J2=J2,
                scorer="sign_overlap",
                additional_scorers=["accuracy"],
                target_score=0.95,
            )

        system_signal = LBFFromSpinSystem(system=system, eigenstate=0, kind=signal_kind)
        system_analyzer = LatticeBooleanAnalyzer(
            signal=system_signal, show_progress=True, cache_dir=fourier_learners_cache_dir
        )

        system_analyzer.fit()

        write_terms_to_file(
            file=system_accuracy_file,
            analyzer=system_analyzer,
            J2=J2,
            scorer="accuracy",
            additional_scorers=["sign_overlap"],
            target_score=0.7,
        )

        write_terms_to_file(
            file=system_overlap_file,
            analyzer=system_analyzer,
            J2=J2,
            scorer="sign_overlap",
            additional_scorers=["accuracy"],
            target_score=0.95,
        )
