from pathlib import Path
import logging
import random
import numpy as np
import torch
import optuna
import os
from fvcore.nn import FlopCountAnalysis
from train import load_dataset, train_validate_loop
from src.net import ConstituentNet

# Set up logging to terminal only
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global dataset cache
_dataset_cache = {}


def seed_everything(seed: int = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_once(num_particles, num_feats, batch_size, val_ratio):
    key = (num_particles, num_feats, batch_size, val_ratio)
    if key not in _dataset_cache:
        logging.info(f"[INFO] Loading dataset for key={key}")
        train_loader, val_loader, test_loader, classes = load_dataset(
            num_particles=num_particles,
            num_feats=num_feats,
            batch_size=batch_size,
            val_ratio=val_ratio,
        )
        _dataset_cache[key] = (train_loader, val_loader, test_loader, classes)
    return _dataset_cache[key]


def structure_objective(trial):
    seed_everything(20)
    num_transformers = trial.suggest_int("num_transformers", 1, 4)
    embbed_dim = trial.suggest_categorical("embbed_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05])

    num_particles, num_feats, batch_size, num_epochs = 30, 16, 256, 25
    train_loader, val_loader, _, classes = load_dataset_once(
        num_particles, num_feats, batch_size, val_ratio=0.1
    )

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": embbed_dim,
        "num_heads": num_heads,
        "num_classes": len(classes),
        "num_transformers": num_transformers,
        "dropout": dropout,
        "is_debug": False,
        "num_particles": num_particles,
        "activation": "ReLU",
        "normalization": "Batch",
    }

    model = ConstituentNet(**model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # FLOPs
    model.eval()
    dummy_input = torch.randn(1, num_particles, num_feats).to(DEVICE)
    try:
        flops = FlopCountAnalysis(model, dummy_input).total() / 1e6
    except Exception as e:
        logging.warning(f"[Warning] FLOP count failed: {e}")
        flops = float("inf")
    finally:
        model.train()

    _, _, _, val_accs = train_validate_loop(
        train_loader,
        val_loader,
        model,
        criterion=torch.nn.NLLLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        early_stopping_patience=3,
        num_particles=num_particles,
        num_feats=num_feats,
        model_config=model_config,
        save=False,
        model_path=None,
        output_path=None,
    )

    # Best val accuracy
    return flops, max(val_accs)


def optimization(n_trials: int = 30):
    Path("optuna_results").mkdir(exist_ok=True)
    logging.info("Stage 1: Architecture Search")

    structure_study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=20),
        study_name="structure_study",
        storage="sqlite:///optuna_results/structure_study.db",
        load_if_exists=True,
    )
    structure_study.optimize(structure_objective, n_trials=n_trials)


if __name__ == "__main__":
    seed_everything(20)
    optimization(n_trials=30)
