import torch
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.net import ConstituentNet
from train import load_model, seed_everything
from prune import load_pruned_model

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def trace_from_full_model(
    model_index, pruning_ratio, num_particles, num_feats, base_path, pruned_path, device
):
    model_base = load_model(
        model_class=ConstituentNet,
        num_particles=num_particles,
        num_feats=num_feats,
        device=device,
        model_path=f"tmp/models/model{model_index}.pth",
    )[0]
    model_pruned = load_pruned_model(model_index, pruning_ratio)

    model_base.eval()
    model_pruned.eval()
    dummy_input = torch.randn(1, num_particles, num_feats).to(device)
    model_base_script = torch.jit.trace(model_base, dummy_input)
    model_pruned_script = torch.jit.trace(model_pruned, dummy_input)
    torch.jit.save(model_base_script, base_path)
    torch.jit.save(model_pruned_script, pruned_path)
    print(f"Base model scripted and saved to {base_path}")
    print(f"Pruned model scripted and saved to {pruned_path}")


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

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

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
    # base_path = f"tmp/models/model{model_index}.pt"
    # pruned_path = f"tmp/pruned_models/pruned_model{model_index}_{pruning_ratio}.pt"
    # Convert models to scripted format to better estimate latency
    # if not os.path.exists(base_path) or not os.path.exists(pruned_path):
    #     os.makedirs(os.path.dirname(base_path), exist_ok=True)
    #     os.makedirs(os.path.dirname(pruned_path), exist_ok=True)

    #     trace_from_full_model(
    #         model_index,
    #         pruning_ratio,
    #         num_particles,
    #         num_feats,
    #         base_path,
    #         pruned_path,
    #         device,
    #     )

    model_base = load_model(
        model_class=ConstituentNet,
        num_particles=num_particles,
        num_feats=num_feats,
        device=device,
        model_path=f"tmp/models/model{model_index}.pth",
    )[0]
    model_pruned = load_pruned_model(model_index, pruning_ratio)

    # model_base_script = torch.jit.load(base_path, map_location=device)
    # model_pruned_script = torch.jit.load(pruned_path, map_location=device)

    example_input = torch.randn(32, num_particles, num_feats).to(device)
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
    model_index = 6
    pruning_ratio = 0.7
    compare_latency(
        model_index,
        pruning_ratio,
        repetitions=500,
        num_particles=8,
        num_feats=3,
    )
