import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, DataLoader
import allo
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
ALLO_T4P_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../../allo_t4p"))
LLVM_BUILD_DIR = os.path.join(ALLO_T4P_ROOT, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = LLVM_BUILD_DIR

from train import seed_everything, load_dataset
from train_mlp import evaluate_mlp

# source ~/xilinx_vitis.sh
# source /opt/xilinx/xrt/setup.sh


class Particle_MLP(nn.Module):
    def __init__(self, in_dim):
        super(Particle_MLP, self).__init__()
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


# LLVM
def llvm_emu(example_inputs):
    llvm_mod = allo.frontend.from_pytorch(
        model, example_inputs=example_inputs, verbose=False
    )
    golden = model(*example_inputs)
    np_inputs = [x.detach().numpy() for x in example_inputs]
    res = llvm_mod(*np_inputs)
    torch.testing.assert_close(res, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Passed!")

    return llvm_mod


# Vitis HLS
def vitis_emu(example_inputs, mode="sw_emu", project_name="mlp_emu.prj"):
    os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
    os.environ["XCL_EMULATION_MODE"] = "sw_emu"

    vitis_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=example_inputs,
        target="vitis_hls",
        mode=mode,
        project=project_name,
    )
    # print(vitis_mod.hls_code)

    golden = model(*example_inputs)
    # x_np = np.random.random((batch_size, num_feats)).astype(np.float32)
    x_np = example_inputs[0].detach().numpy()
    allo_out = np.zeros((batch_size, 5), dtype=np.float32)

    vitis_mod(x_np, allo_out)
    np.testing.assert_allclose(allo_out, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Passed!")

    return vitis_mod


def evaluate(model, mod, test_loader):
    def run_inference(m, x):
        # Pytorch model
        if isinstance(m, torch.nn.Module):
            return m(x)
        x_np = x.detach().numpy()
        # LLVM
        if isinstance(m, allo.backend.llvm.LLVMModule):
            out_np = m(x_np)
            return torch.from_numpy(out_np)
        # Vitis
        if isinstance(m, allo.backend.hls.HLSModule):
            batch_size = x_np.shape[0]
            num_classes = 5
            out_np = np.zeros((batch_size, num_classes), dtype=np.float32)
            m(x_np, out_np)
            return torch.from_numpy(out_np)
        raise ValueError("Unsupported model type")

    acc1, auc1 = _evaluate_single(model, test_loader, run_inference)
    acc2, auc2 = _evaluate_single(mod, test_loader, run_inference)

    print(
        f"PyTorch Model - Accuracy: {acc1:.4f}, AUC: {auc1:.4f}"
        f"\nAllo Model - Accuracy: {acc2:.4f}, AUC: {auc2:.4f}"
    )


def _evaluate_single(model, test_loader, run_inference):
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1)
            output = run_inference(model, x)
            probs = F.softmax(output, dim=-1)

            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs)
            all_labels.append(y)

    accuracy = correct / total
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return accuracy, auc


if __name__ == "__main__":
    batch_size = 32
    num_particles = 8
    num_feats = 3

    model = Particle_MLP(in_dim=num_particles * num_feats)
    model.load_state_dict(torch.load("mlp_best_model.pth"))
    model.eval()

    seed_everything(20)
    _, _, test_loader, _ = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
    )
    example_inputs = [next(iter(test_loader))[0].view(batch_size, -1)]

    # llvm_mod = llvm_emu(example_inputs)
    # print("Target: LLVM")
    # evaluate(model, llvm_mod, test_loader)

    # test sw_emu
    vitis_mod = vitis_emu(example_inputs, mode="sw_emu", project_name="test_sw.prj")
    print("Target: VITIS HLS")
    num_batches = 50
    subset_dataset = Subset(test_loader.dataset, range(num_batches * batch_size))
    subset_test_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
    evaluate(model, vitis_mod, subset_test_loader)
    print(
        f"{len(test_loader)} batches in total, {num_batches} batches tested, batch size: {batch_size}"
    )
