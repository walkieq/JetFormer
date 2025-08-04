import allo
import torch
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, DataLoader
from model import ConstituentNet, SliceFirstDim
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
ALLO_T4P_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../../allo_t4p"))
LLVM_BUILD_DIR = os.path.join(ALLO_T4P_ROOT, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = LLVM_BUILD_DIR

from train import seed_everything, load_dataset, load_model

# source ~/xilinx_vitis.sh
# source /opt/xilinx/xrt/setup.sh


# LLVM
def llvm_emu(example_inputs):
    llvm_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=example_inputs,
        leaf_modules=(SliceFirstDim,),
        verbose=False,
    )
    golden = model(*example_inputs)
    np_input = example_inputs[0].detach().numpy()
    res = llvm_mod(np_input)
    torch.testing.assert_close(res, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Test passed!")

    return llvm_mod


# Vitis HLS
def vitis_emu(example_inputs, mode="sw_emu", project_name="transformer_emu.prj"):
    os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
    os.environ["XCL_EMULATION_MODE"] = mode

    vitis_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=example_inputs,
        leaf_modules=(SliceFirstDim,),
        target="vitis_hls",
        mode=mode,
        project=project_name,
        verbose=False,
    )
    # print(vitis_mod.hls_code)

    golden = model(*example_inputs)
    np_input = example_inputs[0].detach().numpy()
    allo_out = np.zeros((batch_size, 5), dtype=np.float32)

    vitis_mod(np_input, allo_out)
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

    acc1, auc1 = _evaluate_single(
        model, test_loader, run_inference, name="PyTorch model"
    )
    acc2, auc2 = _evaluate_single(mod, test_loader, run_inference, name="Allo model")

    print(
        f"PyTorch Model - Accuracy: {acc1:.4f}, AUC: {auc1:.4f}"
        f"\nAllo Model - Accuracy: {acc2:.4f}, AUC: {auc2:.4f}"
    )


def _evaluate_single(model, test_loader, run_inference, name):
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Evaluating {name}"):
            output = run_inference(model, x)
            probs = output.exp()
            preds = probs.argmax(dim=1)
            # Last batch may have fewer samples
            if preds.shape[0] != y.shape[0]:
                preds = preds[: y.shape[0]]
                probs = probs[: y.shape[0]]
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs)
            all_labels.append(y)

    accuracy = correct / total
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        num_classes = all_probs.shape[1]
        all_labels_onehot = np.eye(num_classes)[all_labels.astype(int)]
        auc = roc_auc_score(all_labels_onehot, all_probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return accuracy, auc


if __name__ == "__main__":

    num_particles = 8
    num_feats = 3
    # TODO: try smaller batch size for hw emu
    # try pruned model3: need to reconstruct the model architecture + load state
    batch_size = 32

    # num_transformers = 4
    # embbed_dim = 16
    # num_heads = 2
    # dropout = 0

    # Load model
    # model_path = os.path.join(
    #     PROJECT_ROOT, f"compress/tmp/pruned_models/pruned_model6_0.5.pth"
    # )
    # model = torch.load(model_path, weights_only=False)
    model_path = os.path.join(PROJECT_ROOT, "compress/tmp/models/model3.pth")
    model = load_model(
        model_class=ConstituentNet,
        num_particles=num_particles,
        num_feats=num_feats,
        device="cpu",
        model_path=model_path,
    )[0]

    model.eval()

    seed_everything(20)
    _, _, test_loader, _ = load_dataset(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        num_workers=0,
    )

    example_inputs = [next(iter(test_loader))[0]]

    # LLVM
    # llvm_mod = llvm_emu(example_inputs)
    # print("Target: LLVM")
    # evaluate(model, llvm_mod, test_loader)

    # VITIS HLS
    mode = "hw_emu"
    project_name = "transformer_hw.prj"
    vitis_mod = vitis_emu(example_inputs, mode=mode, project_name=project_name)
    # Evaluation
    num_batches = 10
    subset_dataset = Subset(test_loader.dataset, range(num_batches * batch_size))
    subset_test_loader = DataLoader(
        subset_dataset, batch_size=batch_size, shuffle=False
    )
    evaluate(model, vitis_mod, subset_test_loader)
    print("Target: VITIS HLS")
    print(
        f"{len(test_loader)} batches in total, {num_batches} batches tested, batch size: {batch_size}"
    )
