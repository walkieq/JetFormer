# Use trial subprocess to speed up optuna tuning
import json
import sys
import os
import torch
import numpy as np
import random
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from train import train_validate_loop
from src.net import ConstituentNet
from src.dataset import H5Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fixed_dataset(num_particles=8, num_feats=3, batch_size=256):
    train_path = os.path.join(
        PROJECT_ROOT, "data/processed", str(num_particles), f"{num_feats}f", "train.h5"
    )

    # Load dataset indices and stats
    train_indices = np.load("split_data/train_indices.npy")
    val_indices = np.load("split_data/val_indices.npy")
    stats = np.load("split_data/dataset_stats.npz")
    mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])

    # Normalize
    full_dataset = H5Dataset(train_path, mean=mean, std=std)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )

    classes = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]
    return train_loader, val_loader, classes


def run_trial(param_file):
    with open(param_file, "r") as f:
        params = json.load(f)

    num_particles = params["num_particles"]
    num_feats = params["num_feats"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]

    train_loader, val_loader, classes = load_fixed_dataset(
        num_particles=num_particles, num_feats=num_feats, batch_size=batch_size
    )

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": params["embbed_dim"],
        "num_heads": params["num_heads"],
        "num_classes": len(classes),
        "num_transformers": params["num_transformers"],
        "dropout": params["dropout"],
        "is_debug": False,
        "num_particles": num_particles,
        "activation": "ReLU",
        "normalization": "Batch",
    }

    model = ConstituentNet(**model_config).to(DEVICE)
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

    model.train()
    _, _, _, val_accs = train_validate_loop(
        train_loader,
        val_loader,
        model,
        criterion=torch.nn.NLLLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        early_stopping_patience=4,
        num_particles=num_particles,
        num_feats=num_feats,
        model_config=model_config,
        save=False,
        model_path=None,
        output_path=None,
    )

    acc = max(val_accs)

    # FLOPs
    model.eval()
    dummy_input = torch.randn(1, num_particles, num_feats).to(DEVICE)
    try:
        flops = FlopCountAnalysis(model, dummy_input).total() / 1e6
    except Exception:
        flops = float("inf")

    result_path = sys.argv[2]
    with open(result_path, "w") as f:
        json.dump({"acc": acc, "flops": flops}, f)


if __name__ == "__main__":
    seed_everything(20)
    run_trial(sys.argv[1])
