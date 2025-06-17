import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
import logging
import torch
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from train import welford_mean_std, H5Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def prepare_dataset(num_particles=8, num_feats=3, val_ratio=0.1):
    train_path = os.path.join(
        PROJECT_ROOT, "data/processed", str(num_particles), f"{num_feats}f", "train.h5"
    )
    raw_dataset = H5Dataset(train_path)
    total_len = len(raw_dataset)
    val_size = int(total_len * val_ratio)
    train_size = total_len - val_size

    torch.manual_seed(20)
    permuted_indices = torch.randperm(total_len)
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:]

    Path("split_data").mkdir(exist_ok=True)
    np.save("split_data/train_indices.npy", train_indices.numpy())
    np.save("split_data/val_indices.npy", val_indices.numpy())

    train_subset = Subset(raw_dataset, train_indices)
    loader_for_stats = DataLoader(
        train_subset, batch_size=512, shuffle=False, num_workers=4
    )
    mean, std = welford_mean_std(loader_for_stats)

    np.savez("split_data/dataset_stats.npz", mean=mean.numpy(), std=std.numpy())
    logging.info("Dataset split and stats saved.")


def structure_objective(trial):
    # Only for 8p3f parameter settings
    num_transformers = trial.suggest_int("num_transformers", 1, 6)
    dim_heads = trial.suggest_categorical(
        "dim_heads", ["8_2", "16_2", "32_2", "64_2", "64_4", "128_2", "128_4", "128_8"]
    )
    embbed_dim, num_heads = map(int, dim_heads.split("_"))
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    trial_params = {
        "num_particles": 8,
        "num_feats": 3,
        "batch_size": 256,
        "num_epochs": 25,
        "embbed_dim": embbed_dim,
        "num_heads": num_heads,
        "num_transformers": num_transformers,
        "dropout": dropout,
    }

    # Write params to temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(trial_params, f)
        f.flush()
        json_path = f.name

    # Save result
    with tempfile.NamedTemporaryFile(
        mode="r", delete=False, suffix=".json"
    ) as result_file:
        result_path = result_file.name

    # Call subprocess and run train_trial.py
    try:
        subprocess.run(
            ["python", "train_trial.py", json_path, result_path],
            check=True,  # Raise CalledProcessError on crash
        )
        with open(result_path, "r") as f:
            result = json.load(f)
        acc = result.get("acc", 0.0)
        flops = result.get("flops", float("inf"))
    except Exception as e:
        print("Subprocess failed:", e)
        acc = 0.0
        flops = float("inf")

    trial.set_user_attr("acc", acc)
    return flops, acc


def constraints_func(trial):
    # Constraint: acc >= 0.65 to be a pareto front
    acc = trial.user_attrs.get("acc", 0.0)
    return [0.65 - acc]


def optimization(n_trials: int = 30):
    Path("optuna_results").mkdir(exist_ok=True)
    logging.info("Architecture Search")

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=NSGAIISampler(seed=20, constraints_func=constraints_func),
        study_name="study_8p3f",
        storage="sqlite:///optuna_results/study_8p3f.db",
        load_if_exists=True,
    )
    study.optimize(structure_objective, n_trials=n_trials)


if __name__ == "__main__":
    # For 8 particle 3 feature dataset
    prepare_dataset(num_particles=8, num_feats=3)
    optimization(n_trials=80)
