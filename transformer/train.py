import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Optional, List
from model.dataset import H5Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from build_dataset import customize_dataset, fetch_hls4ml_dataset

from tqdm import tqdm
from time import time
import random

from model.net import ConstituentNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data PATH
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def seed_everything(seed=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hls4ml_dataset(
    batch_size: int, val_ratio: float
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(PROCESSED_DIR, "1/train")
    test_dir = os.path.join(PROCESSED_DIR, "1/test")

    X_train_val = np.load(os.path.join(train_dir, f"X_train.npy"))
    X_test = np.ascontiguousarray(np.load(os.path.join(test_dir, f"X_test.npy")))
    y_train_val = np.load(os.path.join(train_dir, f"y_train.npy"))
    y_test = np.load(os.path.join(test_dir, f"y_test.npy"))

    classes = np.load(
        os.path.join(PROCESSED_DIR, "1", "classes.npy"), allow_pickle=True
    )

    # Split train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio
    )

    X_train_tensor, y_train_tensor = (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    X_val_tensor, y_val_tensor = (
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    X_test_tensor, y_test_tensor = (
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )

    print(
        f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}"
    )

    X_train_tensor = X_train_tensor.unsqueeze(dim=1)
    X_val_tensor = X_val_tensor.unsqueeze(dim=1)
    X_test_tensor = X_test_tensor.unsqueeze(dim=1)

    print(
        f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}"
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes


def load_N_dataset(
    num_particles: int, num_feats: int, batch_size: int, val_ratio: float
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(
        PROCESSED_DIR, str(num_particles), f"{num_feats}f", "train.h5"
    )
    test_dir = os.path.join(
        PROCESSED_DIR, str(num_particles), f"{num_feats}f", "test.h5"
    )

    classes = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]

    train_dataset = H5Dataset(train_dir)
    test_dataset = H5Dataset(test_dir)

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes


def load_dataset(
    num_particles: int, num_feats: int, batch_size: int = 128, val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    if num_particles == 1:
        return load_hls4ml_dataset(batch_size=batch_size, val_ratio=val_ratio)
    else:
        return load_N_dataset(
            num_particles=num_particles,
            num_feats=num_feats,
            batch_size=batch_size,
            val_ratio=val_ratio,
        )


def train_validate_loop(
    train_loader: DataLoader,
    validate_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
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
    num_epochs: int = 3,
    resume: bool = False,
    num_transformers: int = 3,
    embbed_dim: int = 64,
    num_heads: int = 2,
    activation: str = "ReLU",
    normalization: str = "Batch",
    batch_size: int = 128,
    dropout: float = 0.0,
) -> None:

    # Load dataset
    train_loader, val_loader, test_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
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

    build_dataset = False
    num_particles = 8
    num_feats = 3

    # num_particles = 1
    # num_feats = 16

    if build_dataset:
        if num_particles == 1:
            fetch_hls4ml_dataset()
        else:
            customize_dataset(
                num_particles=num_particles,
                name="train",
            )
            customize_dataset(
                num_particles=num_particles,
                name="test",
            )

    main(
        num_particles=num_particles,
        num_feats=num_feats,
        do_train=True,
        # model_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.model.pth'),
        # script_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.script.pth'),
        # state_path=os.path.join(DIR_NAME, quant_prefix + debug_prefix + 'best.pth.tar'),
        # debug_path=os.path.join(DIR_NAME, 'layers_output.txt'),
        num_epochs=3,
        resume=False,
        num_transformers=3,
        embbed_dim=64,
        num_heads=2,
        activation="ReLU",
        normalization="Batch",
        batch_size=256,
        dropout=0.0,
    )
