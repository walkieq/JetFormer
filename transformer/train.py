import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from h5py import File as hdf5_file
from contextlib import redirect_stdout
from torch.nn.functional import one_hot
from pathlib import Path
from typing import Tuple, Optional, List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm
from time import time
import random

from model.net import ConstituentNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/vol/bitbucket/rz1224/ml_project/Transformer4Physics/transformer/data"
DATASET_DIR = os.path.join(DATA_ROOT, "processed")


def seed_everything(seed=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_h5_dataset(
    dir_name: str, num_particles: int, num_feats: int
) -> Tuple[np.array, np.array]:
    X = np.empty(shape=(0, num_particles, num_feats), dtype=np.float32)
    y = np.empty(shape=(0), dtype=np.float32)

    # suffix = f"{num_feats}f.h5"
    # file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name)
    #              if f.endswith(suffix) and os.path.isfile(os.path.join(dir_name, f))]
    # if len(file_list) == 0:
    #     raise FileNotFoundError(f"No file endswith '{suffix}' in {dir_name}")
    # if len(file_list) > 1:
    #     raise RuntimeError(f"More than one file endswith '{suffix}' in {dir_name}: {file_list}")

    # file_path = file_list[0]
    # with hdf5_file(file_path) as f:
    #     jetConstituentList = np.array(f['jetConstituentList'])
    #     jets = np.array(f['jets'])
    #     target = np.argmax(jets[:, -6:-1], axis=1)

    #     X = np.concatenate((X, jetConstituentList), axis=0, dtype=np.float32)
    #     y = np.concatenate((y, target), axis=0, dtype=np.float32)

    # print(f"X shape: {X.shape}, y shape: {y.shape}")

    pathlist = Path(dir_name).rglob("*.h5")
    files_num = len([name for name in os.listdir(dir_name) if os.path.isfile(name)])

    with tqdm(pathlist, unit="samples chunk", total=files_num) as t_pathlist:

        mode = dir_name.split("/")[-2]
        t_pathlist.set_description(f"{mode}")
        for file_path in t_pathlist:
            with hdf5_file(file_path) as f:
                jetConstituentList = np.array(f["jetConstituentList"])
                jets = np.array(f["jets"])
                target = np.argmax(jets, axis=1)

                X = np.concatenate((X, jetConstituentList), axis=0, dtype=np.float32)
                y = np.concatenate((y, target), axis=0, dtype=np.float32)

                unique, counts = np.unique(target, return_counts=True)
                print()
                for cls, cnt in zip(unique, counts):
                    print(f"Class {cls}: {cnt} samples")

    return X, y


def fetch_hls4ml_dataset(test_ratio=0.2) -> None:
    X, y = fetch_openml("hls4ml_lhc_jets_hlf", cache=True, return_X_y=True)
    assert len(X) == len(y)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = torch.from_numpy(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(os.path.join(DATASET_DIR, "1/train", "X_train.npy"), X_train)
    np.save(os.path.join(DATASET_DIR, "1/test", "X_test.npy"), X_test)
    np.save(os.path.join(DATASET_DIR, "1/train", "y_train.npy"), y_train)
    np.save(os.path.join(DATASET_DIR, "1/test", "y_test.npy"), y_test)
    np.save(os.path.join(DATASET_DIR, "1/classes.npy"), le.classes_)

    print("Fetched hls4ml_lhc_jets_hlf dataset")


def fetch_N_dataset(num_particles: int, num_feats: int) -> None:

    train_dir = os.path.join(DATASET_DIR, str(num_particles), "train")
    test_dir = os.path.join(DATASET_DIR, str(num_particles), "test")

    X_train, y_train = fetch_h5_dataset(
        dir_name=f"{DATASET_DIR}/{num_particles}/train/",
        num_particles=num_particles,
        num_feats=num_feats,
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), f"{X_train.shape[0]=}, {y_train.shape[0]=}"
    np.save(os.path.join(train_dir, f"X_train_{num_feats}f.npy"), X_train)
    np.save(os.path.join(train_dir, f"y_train_{num_feats}f.npy"), y_train)

    X_test, y_test = fetch_h5_dataset(
        dir_name=f"{DATASET_DIR}/{num_particles}/test/",
        num_particles=num_particles,
        num_feats=num_feats,
    )

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    assert X_test.shape[0] == y_test.shape[0]
    np.save(os.path.join(test_dir, f"X_test_{num_feats}f.npy"), X_test)
    np.save(os.path.join(test_dir, f"y_test_{num_feats}f.npy"), y_test)

    print(f"Fetched {num_particles}-particles {num_feats} features dataset")


def load_dataset(
    num_particles: int,
    num_feats: int,
    batch_size: int = 128,
    tiny_size: int = 1,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    train_dir = os.path.join(DATASET_DIR, str(num_particles), "train")
    test_dir = os.path.join(DATASET_DIR, str(num_particles), "test")

    X_train_val = np.load(os.path.join(train_dir, f"X_train_{num_feats}f.npy"))
    X_test = np.ascontiguousarray(
        np.load(os.path.join(test_dir, f"X_test_{num_feats}f.npy"))
    )
    y_train_val = np.load(os.path.join(train_dir, f"y_train_{num_feats}f.npy"))
    y_test = np.load(os.path.join(test_dir, f"y_test_{num_feats}f.npy"))

    if num_particles == 1:
        classes = np.load(
            os.path.join(DATASET_DIR, "1", "classes.npy"), allow_pickle=True
        )
    else:
        classes = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]

    # Split train and validation sets
    if val_ratio > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val
        )

        tensor_X_train = torch.Tensor(X_train)
        tensor_X_val = torch.Tensor(X_val)
        tensor_X_test = torch.Tensor(X_test)

    if num_particles == 1:
        tensor_X_train = tensor_X_train.unsqueeze(dim=1)
        tensor_X_val = tensor_X_val.unsqueeze(dim=1)
        tensor_X_test = tensor_X_test.unsqueeze(dim=1)

    tensor_y_train = torch.LongTensor(y_train)
    tensor_y_val = torch.LongTensor(y_val)
    tensor_y_test = torch.LongTensor(y_test)

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(tensor_X_train, tensor_y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(tensor_X_val, tensor_y_val), batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(tensor_X_test, tensor_y_test), batch_size=batch_size
    )

    # Tiny loaders
    if num_particles == 1:
        tiny_tensor_X_test = tensor_X_test[: tiny_size + 1, :, :]

        tiny_loader = DataLoader(
            TensorDataset(tiny_tensor_X_test, tensor_y_test[: tiny_size + 1]),
            batch_size=1,
        )

    else:
        assert tiny_size <= num_particles
        tiny_tensor_X_test = tensor_X_test[:1, : tiny_size + 1, :]

        tiny_loader = DataLoader(
            TensorDataset(tiny_tensor_X_test, tensor_y_test[:1]), batch_size=1
        )

    return train_loader, val_loader, test_loader, tiny_loader, classes


def train_validate_loop(
    train_loader: DataLoader,
    validate_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_particles: int = 1,
    # print_predictions: bool = False,
    # model_path: Optional[str] = None,
    # script_path: Optional[str] = None,
    # state_path: Optional[str] = None,
) -> None:
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Training epoch {epoch+1}/{num_epochs}")
            for idx, (data, labels) in enumerate(tepoch):
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                all_train_preds.append(preds)
                all_train_labels.append(labels.detach().cpu().numpy())
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = accuracy_score(
            np.concatenate(all_train_labels), np.concatenate(all_train_preds)
        )
        train_accs.append(train_acc)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for data, labels in validate_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                outputs = model(data)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                all_val_preds.append(preds)
                all_val_labels.append(labels.detach().cpu().numpy())
        avg_val_loss = epoch_val_loss / len(validate_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(
            np.concatenate(all_val_labels), np.concatenate(all_val_preds)
        )
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train loss={avg_train_loss:.4f}, Train acc={train_acc:.4f}, "
            f"Val loss={avg_val_loss:.4f}, Val acc={val_acc:.4f}"
        )

    end_time = time()
    total_time = end_time - start_time
    print(f"Training took {total_time:.2f} s")
    return train_losses, val_losses, train_accs, val_accs


def plot_loss_acc(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def inference(
    model: nn.Module,
    data_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_outputs, all_labels


def evaluate_predictions(
    outputs: np.ndarray, labels: np.ndarray, classes, average: str = "macro"
):

    pred_labels = outputs.argmax(axis=1)
    acc = accuracy_score(labels, pred_labels)
    n_classes = outputs.shape[1]

    # Accuracy for each class
    class_accs = []
    for i in range(n_classes):
        idx = labels == i
        if idx.sum() > 0:
            class_acc = accuracy_score(labels[idx], pred_labels[idx])
        else:
            class_acc = float("nan")
        class_accs.append(class_acc)

    # AUC for each class
    try:
        y_true_onehot = np.eye(n_classes)[labels.astype(int)]
        aucs = roc_auc_score(y_true_onehot, outputs, average=None, multi_class="ovr")
    except Exception as e:
        aucs = [None] * n_classes

    print(f"Total Accuracy: {acc:.4f}")
    for i in range(n_classes):
        class_name = classes[i]
        auc_str = f"{aucs[i]:.4f}" if aucs[i] is not None else "N/A"
        acc_str = f"{class_accs[i]:.4f}" if not np.isnan(class_accs[i]) else "N/A"
        print(f"Class {i} ({class_name}): Accuracy={acc_str}, AUC={auc_str}")
    return acc, class_accs, aucs


# def save_model(
#   model: nn.Module,
#   model_path: str,
#   script_path: str,
#   state_path: str,
# ) -> None:
#     model.eval()

#     print(f'Model saved successfully (', end ='')


#     model_script = torch.jit.script(model)
#     model_script.save(script_path)

#     torch.save(model, model_path)

#     print(f'{model_path}, {script_path}, ', end='')

#     torch.save(
#         {'state_dict': model.state_dict()},
#         state_path,
#     )

#     print(f'{state_path})')


def show_config(
    num_particles: int,
    num_feats: int,
    batch_size: int,
    num_transformers: int,
    embbed_dim: int,
    num_heads: int,
    activation: str,
    normalization: str,
    dropout: float,
):
    dataset_name = (
        "hls4ml_lhc_jets_hlf"
        if num_particles == 1
        else f"{num_particles}_{num_feats}f jets"
    )
    print("-" * 15 + " Model configuration " + "-" * 15)
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Criterion: NLLLoss")
    print(f"# Transformers: {num_transformers}")
    print(f"Embedding dim: {embbed_dim}")
    print(f"# Attention heads: {num_heads}")
    print(f"Activation: {activation}")
    print(f"Normalization: {normalization}")
    print(f"Dropout: {dropout}")
    print("-" * 49)


def main(
    num_particles: int,
    num_feats: int,
    do_train: bool = False,
    # model_path: Optional[str] = None,
    # script_path: Optional[str] = None,
    # state_path: Optional[str] = None,
    # debug_path: Optional[str] = None,
    is_debug: bool = False,
    tiny_size: int = 1,
    num_epochs: int = 3,
    resume: bool = False,
    num_transformers: int = 3,
    embbed_dim: int = 64,
    num_heads: int = 2,
    activation: str = "ReLU",
    normalization: str = "Batch",
) -> None:
    # batch_size = 128
    batch_size = 256
    dropout = 0.0

    # Load dataset
    train_loader, val_loader, test_loader, tiny_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        tiny_size=tiny_size,
    )

    show_config(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        num_transformers=num_transformers,
        embbed_dim=embbed_dim,
        num_heads=num_heads,
        activation=activation,
        normalization=normalization,
        dropout=dropout,
    )

    # Initialize model
    if do_train:
        model = ConstituentNet(
            in_dim=num_feats,
            embbed_dim=embbed_dim,
            num_heads=num_heads,
            num_classes=len(classes),
            num_transformers=num_transformers,
            dropout=dropout,
            is_debug=is_debug,
            num_particles=num_particles,
            activation=activation,
            normalization=normalization,
        ).to(DEVICE)

        # Load model
        # if resume:
        #     state_dict = torch.load(state_path)['state_dict']
        #     model.load_state_dict(state_dict, strict=True)
        #     print(f'Model loaded from {state_path}')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        train_losses, val_losses, train_accs, val_accs = train_validate_loop(
            train_loader=train_loader,
            validate_loader=val_loader,
            model=model,
            criterion=torch.nn.NLLLoss(),
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_particles=num_particles,
            # model_path=model_path,
            # script_path=script_path,
            # state_path=state_path,
        )

        # Save model
        # save_model(...)
    # else:
    #     model = torch.load(model_path, map_location=DEVICE)
    #     print(f'Model loaded from {model_path}')
    #     model.to(DEVICE)
    #     model.eval()

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # plot_loss_acc(train_losses, val_losses, train_accs, val_accs)

    # Evaluate
    outputs, labels = inference(model, test_loader)
    evaluate_predictions(outputs, labels, classes)


if __name__ == "__main__":

    seed_everything(20)

    fetch = True
    num_particles = 30
    num_feats = 3

    # num_particles = 1
    # num_feats = 16

    if fetch:
        os.makedirs(
            os.path.join(DATASET_DIR, str(num_particles), "train"), exist_ok=True
        )
        os.makedirs(
            os.path.join(DATASET_DIR, str(num_particles), "test"), exist_ok=True
        )
        if num_particles == 1:
            fetch_hls4ml_dataset()
        else:
            fetch_N_dataset(num_particles=num_particles, num_feats=num_feats)

    main(
        num_particles=num_particles,
        num_feats=num_feats,
        do_train=True,
        # model_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.model.pth'),
        # script_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.script.pth'),
        # state_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.pth.tar'),
        # debug_path=os.path.join(DIR_NAME, 'layers_output.txt'),
        tiny_size=1,
        num_epochs=3,
        resume=False,
        num_transformers=3,
        embbed_dim=64,
        num_heads=2,
        activation="ReLU",
        normalization="Batch",
    )
