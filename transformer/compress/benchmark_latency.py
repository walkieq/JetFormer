import torch
import numpy as np
import os
import sys
from time import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.net import JetFormer
from train import load_model, seed_everything
from prune import load_pruned_model

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def estimate_latency(model, example_inputs, repetitions):
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = np.zeros((repetitions, 1))

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = model(example_inputs)

    t = time()
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    latency = (time() - t) * 1000  # Convert to milliseconds
    print(latency / repetitions, "ms per repetition")

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


def print_latency_comparison(
    latency_mu_base, latency_std_base, latency_mu_pruned, latency_std_pruned
):
    print("=== Latency Comparison ===")
    print(f"Base Model   : {latency_mu_base:.3f} ± {latency_std_base:.3f} ms")
    print(f"Pruned Model : {latency_mu_pruned:.3f} ± {latency_std_pruned:.3f} ms")

    speedup = (latency_mu_base - latency_mu_pruned) / latency_mu_base * 100
    print(f"Latency Reduction: {speedup:.2f}%")


def compare_latency(
    model_index,
    pruning_ratio,
    repetitions=100,
    num_particles=8,
    num_feats=3,
    device=DEVICE,
):
    seed_everything(20)

    model_base = load_model(
        model_class=JetFormer,
        num_particles=num_particles,
        num_feats=num_feats,
        device=device,
        model_path=f"tmp/models/model{model_index}.pth",
    )[0]
    model_pruned = load_pruned_model(model_index, pruning_ratio, device=device)

    # Set larger batch size to get 100% GPU ustilization
    example_input = torch.randn(10240, num_particles, num_feats).to(device)
    latency_mu_base, latency_std_base = estimate_latency(
        model_base, example_input, repetitions
    )
    latency_mu_pruned, latency_std_pruned = estimate_latency(
        model_pruned, example_input, repetitions
    )

    print_latency_comparison(
        latency_mu_base, latency_std_base, latency_mu_pruned, latency_std_pruned
    )


if __name__ == "__main__":
    model_index = 2
    pruning_ratio = 0.5
    compare_latency(
        model_index,
        pruning_ratio,
        repetitions=500,
        num_particles=8,
        num_feats=3,
    )
