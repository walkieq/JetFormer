import torch
import pandas as pd
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from train import seed_everything, load_dataset


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Particle_MLP(nn.Module):
    def __init__(self, in_dim):
        super(Particle_MLP, self).__init__()
        # self.fc1 = nn.Linear(16, 64)
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.output(x), dim=-1)
        x = self.output(x)
        return x


def train_validate_loop(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    num_epochs,
    patience,
    save,
    device,
):
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x = x.view(x.size(0), -1)
            output = model(x)

            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                output = model(x)
                loss = F.cross_entropy(output, y)

                val_loss += loss.item() * x.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = correct / total

        print(
            f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            if save:
                torch.save(best_model_state, "mlp_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if scheduler:
            scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if save:
        print(f"Best model saved at epoch {best_epoch} to 'mlp_best_model.pth'")


def evaluate_mlp(model, test_loader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            output = model(x)
            probs = F.softmax(output, dim=-1)

            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    accuracy = correct / total
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
    return accuracy, auc


def train_mlp(
    batch_size, num_epochs, num_particles, num_feats, save=True, device=DEVICE
):

    model = Particle_MLP(in_dim=num_particles * num_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loader, val_loader, test_loader, classes = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
    )

    train_validate_loop(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        num_epochs,
        patience,
        save,
        device,
    )

    evaluate_mlp(model, test_loader, device)


if __name__ == "__main__":
    seed_everything(20)

    num_particles = 8
    num_feats = 3
    batch_size = 128
    num_epochs = 25
    patience = 3

    # Training and evaluation
    train_mlp(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save=True,
    )

    # Evaluation only
    # train_loader, val_loader, test_loader, classes = load_dataset(
    #     num_particles=num_particles,
    #     num_feats=num_feats,
    #     batch_size=batch_size,
    # )

    # model = Particle_MLP(in_dim=num_particles * num_feats).to(DEVICE)
    # model.load_state_dict(torch.load("mlp_best_model.pth"))
    # evaluate_mlp(model, test_loader, DEVICE)
