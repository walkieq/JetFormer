import torch
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
# from src.net import ConstituentNet
from src.adjusted_model import ConstituentNet
from train import seed_everything, load_dataset, train_validate_loop, load_model
from prune import evaluate_model

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_single_model(
    train_loader,
    val_loader,
    model_config,
    model_index,
    num_particles,
    num_feats,
    num_epochs,
    save,
):
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
    train_validate_loop(
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
        device=DEVICE,
        save=save,
        model_path=f"tmp/models/model{model_index}.pth",
        output_path=f"tmp/outputs/loss_acc_model{model_index}.npz",
    )


def evaluate_best_models(model_indices, test_loader, device=DEVICE):
    for model_index in model_indices:
        model_path = f"tmp/models/model{model_index}.pth"
        model = load_model(
            model_class=ConstituentNet,
            num_particles=num_particles,
            num_feats=num_feats,
            device=DEVICE,
            model_path=model_path,
        )[0]
        acc, auc = evaluate_model(model, test_loader, device)
        print(f"Model {model_index} accuracy: {acc:.4f}, auc: {auc:.4f}")


def train_best_models(
    df, model_indices, num_particles, num_feats, batch_size, num_epochs, save=True
):

    train_loader, val_loader, test_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
    )

    selected = df.iloc[model_indices]
    param_list = selected.to_dict(orient="records")

    for i, param in enumerate(param_list):
        model_config = {
            "in_dim": num_feats,
            "embbed_dim": int(param["embedding_dim"]),
            "num_heads": int(param["num_heads"]),
            "num_classes": len(classes),
            "num_transformers": int(param["num_transformers"]),
            "dropout": float(param["dropout"]),
            "num_particles": num_particles,
            "activation": "ReLU",
            "normalization": "Batch",
        }

        print(f"Training model{model_indices[i]}...")
        print(
            f"Model params: {{'trial_id': {param['trial_id']}, 'num_transformers': {param['num_transformers']}, 'embedding_dim': {param['embedding_dim']}, 'num_heads': {param['num_heads']},  'dropout': {param['dropout']}}}"
        )
        train_single_model(
            train_loader,
            val_loader,
            model_config,
            model_indices[i],
            num_particles,
            num_feats,
            num_epochs,
            save,
        )

    evaluate_best_models(model_indices, test_loader)


if __name__ == "__main__":
    seed_everything(20)

    num_particles = 8
    num_feats = 3
    batch_size = 256
    num_epochs = 25

    df = pd.read_csv("../hpo/best_trials.csv")
    model_indices = [3]
    train_best_models(
        df, model_indices, num_particles, num_feats, batch_size, num_epochs, save=True
    )

    # Evaluation only
    # train_loader, val_loader, test_loader, classes = load_dataset(
    #     num_particles=num_particles,
    #     num_feats=num_feats,
    #     batch_size=batch_size,
    # )
    # evaluate_best_models(model_indices, test_loader)
