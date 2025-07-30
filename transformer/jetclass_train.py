import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from jetclass_dataset import JetClassDataset
import random
from tqdm import tqdm
import copy
from typing import Tuple, Optional, List
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from time import time
from fvcore.nn import FlopCountAnalysis
from train import (
    plot_loss_acc,
    evaluate,
    save_model,
    save_loss_acc,
    load_model,
)
from src.net import ConstituentNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def seed_everything(seed=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_jetclass_files(data_dir, name="train", n_per_class=3, seed=20):
    random.seed(seed)

    label_prefixes = {
        "label_Hbb": "HToBB",
        "label_Hcc": "HToCC",
        "label_Hgg": "HToGG",
        "label_Hqql": "HToWW2Q1L",
        "label_H4q": "HToWW4Q",
        "label_Tbqq": "TTBar",
        "label_Tbl": "TTBarLep",
        "label_Wqq": "WToQQ",
        "label_QCD": "ZJetsToNuNu",
        "label_Zqq": "ZToQQ",
    }

    selected_files = []

    for label, prefix in label_prefixes.items():
        matched = []
        pattern = os.path.join(data_dir, f"{prefix}_*.root")
        files = sorted(glob(pattern))
        if len(files) == 0:
            print(pattern, "not found")

        if name == "train":
            candidates = [f for f in files if _check_file_range(f, 0, 99)]
            matched += random.sample(candidates, min(n_per_class, len(candidates)))

        elif name == "val":
            candidates = [f for f in files if _check_file_range(f, 120, 124)]
            matched += random.sample(candidates, min(n_per_class, len(candidates)))

        elif name == "test":
            candidates = [f for f in files if _check_file_range(f, 100, 119)]
            matched += random.sample(candidates, min(n_per_class, len(candidates)))

        selected_files.extend(matched)

    return sorted(selected_files)


def _check_file_range(filepath, start, end):
    """Extract index like 087 from filename 'HToBB_087.root' and check if it's in [start, end]."""
    filename = os.path.basename(filepath)
    try:
        num = int(filename.split("_")[-1].replace(".root", ""))
        return start <= num <= end
    except Exception as e:
        return False


def load_multiple_files(file_list):
    all_x, all_y = [], []
    for f in tqdm(file_list, desc=f"Processing"):
        dataset = JetClassDataset(f)
        for i in range(len(dataset)):
            x, y = dataset[i]
            all_x.append(x)
            all_y.append(y)
    return torch.stack(all_x), torch.tensor(all_y)


def create_dataset(n_per_class=2, name="train"):
    base_dir = "../../particle_transformer/datasets/JetClass/Pythia"
    if name == "train":
        dir = os.path.join(base_dir, "train_100M")
    elif name == "val":
        dir = os.path.join(base_dir, "val_5M")
    elif name == "test":
        dir = os.path.join(base_dir, "test_20M")

    files = select_jetclass_files(data_dir=dir, name=name, n_per_class=n_per_class)

    # Save to npy files
    X, y = load_multiple_files(files)
    print(X.shape, y.shape)
    os.makedirs(f"JetClass/{n_per_class}M/{name}", exist_ok=True)
    np.save(f"JetClass/{n_per_class}M/train/X_{name}.npy", X.numpy())
    np.save(f"JetClass/{n_per_class}M/train/y_{name}.npy", y.numpy())


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path, mmap=True):
        self.mmap_mode = "r" if mmap else None
        # Use try-except to fallback if mmap causes issues
        try:
            self.X = np.load(x_path, mmap_mode=self.mmap_mode)
            self.y = np.load(y_path, mmap_mode=self.mmap_mode)
        except Exception as e:
            print(f"mmap_mode failed, fallback to RAM load. Reason: {e}")
            self.X = np.load(x_path)
            self.y = np.load(y_path)

        assert len(self.X) == len(self.y), "X and y must be the same length"

    def __getitem__(self, idx):
        # x = torch.from_numpy(self.X[idx]).float()
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor(self.y[idx]).long()
        return x, y

    def __len__(self):
        return len(self.y)


def get_dataloaders(base_dir, batch_size=256, num_workers=4, mmp=True):
    def load_npy(x_path, y_path):
        return NpyDataset(x_path, y_path, mmp)

    train_ds = load_npy(
        f"{base_dir}/train/X_train.npy", f"{base_dir}/train/y_train.npy"
    )
    val_ds = load_npy(f"{base_dir}/val/X_val.npy", f"{base_dir}/val/y_val.npy")
    test_ds = load_npy(f"{base_dir}/test/X_test.npy", f"{base_dir}/test/y_test.npy")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


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
    save: bool = True,
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
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
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
            for data, labels in tqdm(validate_loader, desc="Validate", unit="batch"):
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
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
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
            if save:
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

    if save:
        print(f"Best model saved at epoch {best_epoch} to {model_path}")

    end_time = time()
    total_time = end_time - start_time
    print(f"Time taken for traing: {total_time:.2f} s")

    # Save loss and accuracy for training and validation
    if save:
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


def count_flop_param(model: nn.Module, num_particles: int, num_feats: int) -> None:
    model.eval()
    # Dummy input for FLOPs calculation
    dummy_input = torch.randn(1, num_particles, num_feats).to(DEVICE)

    flops = FlopCountAnalysis(model, dummy_input)
    print("-" * 50)
    flops = flops.total()
    print(f"Total FLOPs: {flops / 1e6}M")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params / 1e6}M")
    print("-" * 50)


if __name__ == "__main__":

    seed_everything(20)
    n_per_class = 2
    # create_dataset(n_per_class=n_per_class, name="train")
    # create_dataset(n_per_class=n_per_class, name="val")
    # create_dataset(n_per_class=n_per_class, name="test")

    ############################### Train ##########################################

    num_particles = 128
    num_feats = 17
    # batch_size = 256
    batch_size = 128
    # num_transformers = 3
    num_transformers = 10
    embbed_dim = 128
    # num_heads = 4
    num_heads = 8
    activation = "ReLU"
    normalization = "Batch"
    dropout = 0

    # num_epochs = 30
    num_epochs = 500
    # early_stopping_patience = 4
    early_stopping_patience = 30
    save = True
    model_path = os.path.join(
        BASE_DIR,
        f"jetclass_results/{n_per_class}M/models/{num_transformers}_{embbed_dim}_{num_heads}_{dropout}.pth",
    )
    plot_path = os.path.join(
        BASE_DIR,
        f"jetclass_results/{n_per_class}M/outputs/{num_transformers}_{embbed_dim}_{num_heads}_{dropout}_plot.png",
    )
    output_path = os.path.join(
        BASE_DIR,
        f"jetclass_results/{n_per_class}M/outputs/{num_transformers}_{embbed_dim}_{num_heads}_{dropout}_loss_acc.npz",
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        base_dir=f"JetClass/{n_per_class}M",
        batch_size=batch_size,
        num_workers=4,
    )
    classes = [
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ]

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": embbed_dim,
        "num_heads": num_heads,
        "num_classes": len(classes),
        "num_transformers": num_transformers,
        "dropout": dropout,
        "num_particles": num_particles,
        "activation": activation,
        "normalization": normalization,
    }

    model = ConstituentNet(**model_config).to(DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

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
        save=save,
        model_path=model_path,
        output_path=output_path,
    )

    if save:
        plot_loss_acc(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            num_particles,
            num_feats,
            plot_path=plot_path,
        )

    # model = load_model(
    #     model_class=ConstituentNet,
    #     num_particles=num_particles,
    #     num_feats=num_feats,
    #     device=DEVICE,
    #     model_path=os.path.join(
    #         BASE_DIR,
    #         f"jetclass_results/{n_per_class}M/models/{num_transformers}_{embbed_dim}_{num_heads}.pth",
    #     ),
    # )[0]
    outputs, labels = inference(model, test_loader)
    evaluate(outputs, labels, classes)

    count_flop_param(model, num_particles, num_feats)
