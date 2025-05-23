import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

from pathlib import Path
from typing import Tuple, Optional, List
from src.dataset import H5Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, Subset


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from build_dataset import customize_dataset, fetch_hls4ml_dataset

from tqdm import tqdm
from time import time
import random
import copy

from src.net import ConstituentNet

# TODO: add scaler for preprocessing
# add file path as argument: path='tmp/models', 'tmp/outputs'
# try 30 50 100 150 particles
# try tiny transformer for 1/8 particles (3f)
# calculate flops


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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


def _welford_mean_std(
    loader: DataLoader, max_batches: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = 0
    mean = None
    M2 = None

    if max_batches is None:
        max_batches = len(loader)
    max_batches = min(len(loader), max_batches)
    for i, (x, _) in enumerate(
        tqdm(islice(loader, max_batches), total=max_batches, desc="Estimating mean/std")
    ):
        x = x.view(-1, x.size(-1))  # Flatten: [B, P, F] -> [B*P, F]
        batch_n = x.size(0)

        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)

        if mean is None:
            mean = batch_mean
            M2 = batch_M2
            n = batch_n
        else:
            delta = batch_mean - mean
            total_n = n + batch_n
            mean = mean + delta * (batch_n / total_n)
            M2 = M2 + batch_M2 + (delta**2) * n * batch_n / total_n
            n = total_n

    std = torch.sqrt(M2 / (n - 1 + 1e-8))
    return mean, std


def _preprocess_h5dataset(
    train_dir: str, test_dir: str, val_ratio: float, num_workers: int = 4
) -> Tuple[Dataset, Dataset, Dataset]:
    raw_train_dataset = H5Dataset(train_dir)
    val_size = int(len(raw_train_dataset) * val_ratio)
    train_size = len(raw_train_dataset) - val_size
    # Randomly split dataset into non-overlapping train and validation subsets
    train_subset, val_subset = random_split(raw_train_dataset, [train_size, val_size])

    # Estimate mean/std on train subset
    print("Estimating mean and std from training data...")
    start_time = time()
    temp_loader = DataLoader(
        train_subset, batch_size=512, shuffle=False, num_workers=num_workers
    )
    mean, std = _welford_mean_std(temp_loader, max_batches=500)
    end_time = time()
    print(f"Time taken for preprocessing: {end_time - start_time:.2f} s")

    # Normalize datasets
    full_train_dataset = H5Dataset(train_dir, mean=mean, std=std)
    train_dataset = Subset(full_train_dataset, train_subset.indices)
    val_dataset = Subset(full_train_dataset, val_subset.indices)
    test_dataset = H5Dataset(test_dir, mean=mean, std=std)

    return train_dataset, val_dataset, test_dataset


def load_N_dataset(
    num_particles: int,
    num_feats: int,
    batch_size: int,
    val_ratio: float,
    num_workers: int = 4,
    prefetch_factor: int = 4,  # Number of batches to prefetch for each worker
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(
        PROCESSED_DIR, str(num_particles), f"{num_feats}f", "train.h5"
    )
    test_dir = os.path.join(
        PROCESSED_DIR, str(num_particles), f"{num_feats}f", "test.h5"
    )

    classes = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]

    train_dataset, val_dataset, test_dataset = _preprocess_h5dataset(
        train_dir=train_dir,
        test_dir=test_dir,
        val_ratio=val_ratio,
        num_workers=num_workers,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,  # Keep workers alive between epochs instead of restarting
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader, classes


def load_dataset(
    num_particles: int,
    num_feats: int,
    batch_size: int = 128,
    val_ratio: float = 0.1,
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
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    early_stopping_patience: int,
    num_particles: int,
    num_feats: int,
    model_config: dict,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time()

    best_val_loss = float("inf")
    # Minimum change to qualify as an improvement
    min_delta = 1e-4
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

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
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}: "
            f"Train loss={avg_train_loss:.4f}, Train acc={train_acc:.4f}, "
            f"Val loss={avg_val_loss:.4f}, Val acc={val_acc:.4f}, "
            f"LR={current_lr:.6f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            save_model(
                best_model_state,
                model_config=model_config,
                best_loss=best_val_loss,
                model_path=model_path,
            )

        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best model saved at epoch {best_epoch} to {model_path}")

    end_time = time()
    total_time = end_time - start_time
    print(f"Time taken for traing: {total_time:.2f} s")

    # Save loss and accuracy for training and validation
    save_loss_acc(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        num_particles,
        num_feats,
        output_path,
    )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, train_accs, val_accs


def save_model(
    best_model_state: dict,
    model_config: dict,
    best_loss: float,
    model_path: Optional[str] = None,
) -> None:
    if model_path is None:
        model_path = os.path.join(
            MODEL_DIR,
            f"{model_config['num_particles']}_{model_config['in_dim']}f.pth",
        )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_config": model_config,
            "best_loss": best_loss,
        },
        model_path,
    )


def load_model(
    model_class,
    num_particles: int,
    num_feats: int,
    device: torch.device = DEVICE,
    model_path: Optional[str] = None,
) -> nn.Module:
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"{num_particles}_{num_feats}f.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


def save_loss_acc(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    num_particles: int,
    num_feats: int,
    output_path: Optional[str],
):
    if output_path is None:
        output_path = os.path.join(
            OUTPUT_DIR, f"{num_particles}_{num_feats}f_loss_acc.npz"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_accs=np.array(train_accs),
        val_accs=np.array(val_accs),
    )
    print(f"Loss and accuracy saved to {output_path}")


def load_loss_acc(num_particles, num_feats):
    data = np.load(
        os.path.join(OUTPUT_DIR, f"{num_particles}_{num_feats}f_loss_acc.npz")
    )
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    train_accs = data["train_accs"]
    val_accs = data["val_accs"]
    return train_losses, val_losses, train_accs, val_accs


def plot_loss_acc(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    num_particles: int,
    num_feats: int,
    plot_path: Optional[str] = None,
):
    if plot_path is None:
        plot_path = os.path.join(OUTPUT_DIR, f"{num_particles}_{num_feats}f_plot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    epochs = np.arange(len(train_losses))

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.xticks(epochs)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print("Loss and accuracy plots saved to " f"{plot_path}")


def inference(
    model: nn.Module,
    data_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Inference", unit="batch"):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_outputs, all_labels


def evaluate(outputs: np.ndarray, labels: np.ndarray, classes, average: str = "macro"):

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
    print(f"Device: {DEVICE}")
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


def train(
    num_particles: int,
    num_feats: int,
    do_train: bool = True,
    is_debug: bool = False,
    val_ratio: float = 0.1,
    num_epochs: int = 10,
    early_stopping_patience: int = 2,
    num_transformers: int = 3,
    embbed_dim: int = 64,
    num_heads: int = 2,
    activation: str = "ReLU",
    normalization: str = "Batch",
    batch_size: int = 128,
    dropout: float = 0.0,
    model_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:

    # Load dataset
    train_loader, val_loader, test_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        val_ratio=val_ratio,
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

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": embbed_dim,
        "num_heads": num_heads,
        "num_classes": len(classes),
        "num_transformers": num_transformers,
        "dropout": dropout,
        "is_debug": is_debug,
        "num_particles": num_particles,
        "activation": activation,
        "normalization": normalization,
    }

    # Initialize model
    if do_train:
        model = ConstituentNet(**model_config).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-4
        )

        # Train
        train_losses, val_losses, train_accs, val_accs = train_validate_loop(
            train_loader=train_loader,
            validate_loader=val_loader,
            model=model,
            criterion=torch.nn.NLLLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            num_particles=num_particles,
            num_feats=num_feats,
            model_config=model_config,
            model_path=model_path,
            output_path=output_path,
        )

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    plot_loss_acc(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        num_particles,
        num_feats,
        plot_path=plot_path,
    )

    # Evaluate
    outputs, labels = inference(model, test_loader)
    evaluate(outputs, labels, classes)

    # Load model
    model_new = load_model(
        model_class=ConstituentNet,
        num_particles=num_particles,
        num_feats=num_feats,
        model_path=model_path,
    )

    outputs, labels = inference(model_new, test_loader)
    evaluate(outputs, labels, classes)


if __name__ == "__main__":

    seed_everything(20)

    build_dataset = False
    num_particles = 30
    num_feats = 16
    feats = range(16)
    # num_feats = 3
    # feats = [5, 8, 11]

    if build_dataset:
        if num_particles == 1:
            fetch_hls4ml_dataset()
        else:
            customize_dataset(
                num_particles=num_particles,
                feats=feats,
                name="train",
            )
            customize_dataset(
                num_particles=num_particles,
                feats=feats,
                name="test",
            )

    train(
        num_particles=num_particles,
        num_feats=num_feats,
        do_train=True,
        num_epochs=25,
        early_stopping_patience=4,
        num_transformers=3,
        embbed_dim=64,
        num_heads=2,
        activation="ReLU",
        normalization="Batch",
        batch_size=256,
        dropout=0.0,
        model_path=os.path.join(
            BASE_DIR, f"tmp/models/{num_particles}_{num_feats}f.pth"
        ),
        plot_path=os.path.join(
            BASE_DIR, f"tmp/outputs/{num_particles}_{num_feats}f_plot.png"
        ),
        output_path=os.path.join(
            BASE_DIR, f"tmp/outputs/{num_particles}_{num_feats}f_loss_acc.npz"
        ),
    )
