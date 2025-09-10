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
from optuna.samplers import TPESampler
from optuna.integration import BoTorchSampler
from optuna_main import prepare_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from train import welford_mean_std, H5Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def structure_objective(trial):
    num_transformers = trial.suggest_int("num_transformers", 1, 6)
    dim_heads = trial.suggest_categorical(
        "dim_heads", ["8_2", "16_2", "32_2", "64_2", "64_4", "128_2", "128_4", "128_8"]
    )
    embbed_dim, num_heads = map(int, dim_heads.split("_"))
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05])

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

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(trial_params, f)
        f.flush()
        json_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="r", delete=False, suffix=".json"
    ) as result_file:
        result_path = result_file.name

    try:
        subprocess.run(
            ["python", "train_trial.py", json_path, result_path],
            check=True,
            env={**os.environ, "PYTHON_PARENT_SCRIPT": __file__},
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
    # Constraint: only acc >= 0.65 is feasible
    acc = trial.user_attrs.get("acc", 0.0)
    return [0.65 - acc]


def get_sampler(name):
    name = name.lower()
    if name == "nsga2":
        return NSGAIISampler(constraints_func=constraints_func)
    elif name == "tpe":
        return TPESampler(
            multivariate=True,
            constraints_func=constraints_func,
            n_startup_trials=20,
        )
    elif name == "botorch":
        return BoTorchSampler(
            constraints_func=constraints_func,
            n_startup_trials=20,  # Number of random trials run before switching to BoTorch search
        )
    else:
        raise ValueError(f"Unknown sampler: {name}")


def run_sampler(sampler_name, n_trials):
    Path("optuna_results").mkdir(exist_ok=True)
    study_name = f"study_{sampler_name}"
    db_path = f"sqlite:///optuna_results/{study_name}.db"

    sampler = get_sampler(sampler_name)
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=sampler,
        study_name=f"{study_name}",
        storage=db_path,
        load_if_exists=True,
    )

    logging.info(f"Running {sampler_name.upper()} optimization...")
    study.optimize(structure_objective, n_trials=n_trials)
    logging.info(f"Finished {sampler_name.upper()}: {len(study.trials)} trials.")


if __name__ == "__main__":
    # prepare_dataset(num_particles=8, num_feats=3, seed=20)
    # for sampler in ["nsga2", "tpe", "botorch"]:
    for sampler in ["nsga2"]:
        run_sampler(sampler_name=sampler, n_trials=3)
