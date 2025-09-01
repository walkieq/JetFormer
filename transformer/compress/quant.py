import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_model import JetFormer
from train import seed_everything, load_dataset, train_validate_loop
from prune import evaluate_model, count_flop_param

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

seed_everything(20)

num_particles = 16
num_feats = 3
batch_size = 128
num_epochs = 80

num_transformers = 3
embbed_dim = 64
num_heads = 2
dropout = 0

model_config = {
    "in_dim": num_feats,
    "embbed_dim": embbed_dim,
    "num_heads": num_heads,
    "num_classes": 5,
    "num_transformers": num_transformers,
    "dropout": dropout,
    "num_particles": num_particles,
    "activation": "ReLU",
    "normalization": "Batch",
}
model = JetFormer(**model_config).to(DEVICE)

train_loader, val_loader, test_loader, classes = load_dataset(
    num_particles=num_particles,
    num_feats=num_feats,
    batch_size=batch_size,
)
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=1.5e-3, weight_decay=1e-2, betas=(0.9, 0.98)
# )
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=1.5e-2,
#     steps_per_epoch=len(train_loader),
#     epochs=num_epochs,
#     pct_start=0.35,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=1e4,
# )

steps_per_epoch = len(train_loader)
total_steps = num_epochs * steps_per_epoch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=8e-4,
    betas=(0.9, 0.98),
    weight_decay=0.01,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-4
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
    early_stopping_patience=12,
    num_particles=num_particles,
    num_feats=num_feats,
    model_config=model_config,
    device=DEVICE,
    save=False,
    # model_path=f"tmp/bit_models/model{num_particles}p.pth",
    # output_path=f"tmp/bit_outputs/loss_acc_model{num_particles}p.npz",
)

acc, auc = evaluate_model(model, test_loader, device=DEVICE)
# flops, params = count_flop_param(model, num_particles, num_feats, device=DEVICE)
print(f"acc: {acc:.4f}, auc: {auc:.4f}")
