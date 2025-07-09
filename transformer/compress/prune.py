import torch
import torch.nn as nn
import torch_pruning as tp
from fvcore.nn import FlopCountAnalysis
import sys
import os
import copy
import logging
import numpy as np
import pandas as pd
import warnings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.net import ConstituentNet
from src.layer import SelfAttention
from train import seed_everything, load_dataset, train_validate_loop, load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataloader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def count_flop_param(model, num_particles=8, num_feats=3, device=DEVICE):
    dummy_inputs = torch.randn(1, num_particles, num_feats).to(device)
    flops = FlopCountAnalysis(model, dummy_inputs).total() / 1e6
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return flops, params


def get_percentage(base, pruned):
    return round((1 - pruned / base) * 100, 2)


def pruning_summary(
    base_acc, pruned_acc, base_params, pruned_params, base_flops, pruned_flops
):
    df = pd.DataFrame(
        {
            "Metric": ["FLOPs", "Params", "Accuracy"],
            "Before": [f"{base_flops}M", f"{base_params}M", base_acc],
            "After": [f"{pruned_flops}M", f"{pruned_params}M", pruned_acc],
            "Change (%)": [
                get_percentage(base_flops, pruned_flops),
                get_percentage(base_params, pruned_params),
                get_percentage(base_acc, pruned_acc),
            ],
        }
    )
    print("\nPruning Summary:")
    print(df.to_string(index=False))


def train(
    model_config,
    model_index,
    train_loader,
    val_loader,
    num_particles,
    num_feats,
    num_epochs,
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
        save=True,
        model_path=f"tmp/models/model{model_index}.pth",
        output_path=f"tmp/outputs/loss_acc_model{model_index}.npz",
    )


def compute_taylor_importance_gradients(model, dataloader, device, max_batches):
    # Compute importance score over minibatches
    model.train()
    model.zero_grad()

    criterion = torch.nn.NLLLoss()
    dataloader_iter = iter(dataloader)

    for i in range(max_batches):
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()


def prune(
    model_pruned,
    num_particles,
    num_feats,
    num_heads,
    train_loader,
    val_loader,
    test_loader,
    pruning_ratio=0.5,
    num_classes=5,
    iterative_steps=3,
    finetune_epochs=5,
    finetune=True,
    verbose=False,
):
    if verbose:
        print_linear_layer_shapes(model_pruned, "Before Pruning")
        print_attention_heads(model_pruned, "Before Pruning")

    example_inputs = torch.randn(1, num_particles, num_feats).to(DEVICE)
    imp = tp.importance.TaylorImportance()
    # imp = tp.importance.MagnitudeImportance(p=1)

    unwrapped_parameters = [(model_pruned.cls_token, 0)]
    # ignored_params = [model_pruned.cls_token]
    # Do not prune the final classifier layer
    ignored_layers = [
        m
        for m in model_pruned.modules()
        if isinstance(m, nn.Linear) and m.out_features == num_classes
    ]

    pruner = tp.pruner.MagnitudePruner(
        model_pruned,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        round_to=num_heads,  # round dim to the nearest multiple of num_headse to avoid errors
    )

    # Step-by-step pruning and fine-tuning
    for step in range(iterative_steps):
        print(f"\n[Pruning Step {step+1}]")

        if isinstance(imp, tp.importance.TaylorImportance):
            compute_taylor_importance_gradients(
                model=model_pruned,
                dataloader=train_loader,
                device=DEVICE,
                max_batches=50,
            )

        # Remove dims with least importance
        # groups = pruner.step(interactive=True)
        # for group in groups:
        #     group.prune()
        pruner.step()

        if verbose:
            print_linear_layer_shapes(model_pruned, "After Pruning")
            print_attention_heads(model_pruned, "After Pruning")

        if finetune:
            print(f"[Fine-tuning Step {step+1}]")
            optimizer = torch.optim.AdamW(
                model_pruned.parameters(), lr=1e-3, weight_decay=1e-2
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-3,
                steps_per_epoch=len(train_loader),
                epochs=finetune_epochs,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )
            train_validate_loop(
                train_loader,
                val_loader,
                model_pruned,
                criterion=torch.nn.NLLLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=finetune_epochs,
                early_stopping_patience=3,
                num_particles=num_particles,
                num_feats=num_feats,
                model_config=None,
                device=DEVICE,
                save=False,
                model_path=None,
                output_path=None,
            )

        # Evaludate pruned model
        pruned_acc = evaluate_model(model_pruned, test_loader)
        pruned_flops, pruned_params = count_flop_param(model_pruned)
        print(
            f"After step {step+1}: FLOPs = {pruned_flops}M, Params = {pruned_params}M, Accuracy = {pruned_acc:.4f}"
        )

    return pruned_flops, pruned_params, pruned_acc


def print_linear_layer_shapes(model, name):
    print(f"\n{name} - Linear Layer Output Shapes:")
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            print(
                f"{n}: in_features = {m.in_features}, out_features = {m.out_features}"
            )


def print_attention_heads(model, name):
    print(f"\n{name} - Attention Head Configuration:")
    for module_name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            q_weight_shape = module.q.weight.shape
            latent_dim = q_weight_shape[0]  # same as out_features of Linear
            num_heads = module.heads
            head_dim = latent_dim // num_heads if latent_dim % num_heads == 0 else "N/A"
            print(f"{module_name}:")
            print(f"  in_dim     = {module.in_dim}")
            print(f"  latent_dim = {latent_dim} (after linear projection)")
            print(f"  num_heads  = {num_heads}")
            print(f"  head_dim   = {head_dim}")
            print(f"  q_weight.shape = {tuple(q_weight_shape)}")
            print()


def load_pruned_model(model_index, pruning_ratio, device=DEVICE):
    model_path = f"tmp/pruned_models/pruned_model{model_index}_{pruning_ratio}.pth"
    model = torch.load(model_path, map_location=device)
    print("Pruned model loaded from", model_path)
    return model


def save_pruned_model(
    model,
    model_index,
    pruning_ratio,
):
    model_path = f"tmp/pruned_models/pruned_model{model_index}_{pruning_ratio}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.eval()
    torch.save(model, model_path)
    print(f"Pruned model saved to {model_path}")


def main(
    model_index, pruning_ratio, finetune=True, verbose=False, save=True, device=DEVICE
):
    df = pd.read_csv("../hpo/best_trials.csv")
    selected = df.iloc[model_index]
    param = selected.to_dict()
    print(
        f"Model{model_index} params: {{'trial_id': {param['trial_id']}, 'num_transformers': {param['num_transformers']}, 'embedding_dim': {param['embedding_dim']}, 'num_heads': {param['num_heads']},  'dropout': {param['dropout']}}}"
    )
    num_particles = 8
    num_feats = 3
    batch_size = 256
    num_classes = 5

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": int(param["embedding_dim"]),
        "num_heads": int(param["num_heads"]),
        "num_classes": num_classes,
        "num_transformers": int(param["num_transformers"]),
        "dropout": float(param["dropout"]),
        "num_particles": num_particles,
        "activation": "ReLU",
        "normalization": "Batch",
    }

    train_loader, val_loader, test_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
    )

    model_path = f"tmp/models/model{model_index}.pth"
    if not os.path.exists(model_path):
        ### Train origin model ###
        print("Training original model...")
        num_epochs = 25
        train(
            model_config=model_config,
            model_index=model_index,
            train_loader=train_loader,
            val_loader=val_loader,
            num_particles=num_particles,
            num_feats=num_feats,
            num_epochs=num_epochs,
        )

    ### Load origin model ###
    model = load_model(
        model_class=ConstituentNet,
        num_particles=num_particles,
        num_feats=num_feats,
        device=DEVICE,
        model_path=model_path,
    )[0]

    base_acc = evaluate_model(model, test_loader)
    base_flops, base_params = count_flop_param(model)
    print(
        f"Original: FLOPs = {base_flops}M, Params = {base_params}M, Accuracy = {base_acc:.4f}"
    )

    ### Prune ###
    iterative_steps = 3
    finetune_epochs = 5
    model_pruned = copy.deepcopy(model).to(device)

    pruned_flops, pruned_params, pruned_acc = prune(
        model_pruned=model_pruned,
        num_particles=num_particles,
        num_feats=num_feats,
        pruning_ratio=pruning_ratio,
        num_heads=model_config["num_heads"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        iterative_steps=iterative_steps,
        finetune_epochs=finetune_epochs,
        finetune=finetune,
        verbose=verbose,
    )

    if save:
        save_pruned_model(
            model=model_pruned, model_index=model_index, pruning_ratio=pruning_ratio
        )

    pruning_summary(
        base_acc, pruned_acc, base_params, pruned_params, base_flops, pruned_flops
    )


if __name__ == "__main__":
    seed_everything(20)

    model_index = 6
    pruning_ratio = 0.7
    main(model_index, pruning_ratio, save=True)

    # # Evaluation only
    # df = pd.read_csv("../hpo/best_trials.csv")
    # selected = df.iloc[model_index]
    # param = selected.to_dict()

    # num_particles = 8
    # num_feats = 3
    # batch_size = 256
    # num_classes = 5

    # train_loader, val_loader, test_loader, classes = load_dataset(
    #     num_particles=num_particles,
    #     num_feats=num_feats,
    #     batch_size=batch_size,
    # )

    # model_path = f"tmp/models/model{model_index}.pth"
    # model = load_model(
    #     model_class=ConstituentNet,
    #     num_particles=num_particles,
    #     num_feats=num_feats,
    #     device=DEVICE,
    #     model_path=model_path,
    # )[0]

    # # Base model
    # base_acc = evaluate_model(model, test_loader)
    # base_flops, base_params = count_flop_param(model)

    # # Pruned model
    # model_pruned = load_pruned_model(model_index, pruning_ratio)
    # pruned_acc = evaluate_model(model_pruned, test_loader)
    # pruned_flops, pruned_params = count_flop_param(model_pruned)

    # pruning_summary(
    #     base_acc, pruned_acc, base_params, pruned_params, base_flops, pruned_flops
    # )
