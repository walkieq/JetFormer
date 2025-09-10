## Model Deployment

#### Environment Installation
Files in this folder should be run in the allo environment (`environment_allo.yml`). To install Allo, please follow the guidance here: https://cornell-zhang.github.io/allo/setup/index.html.


#### Run
Simply run `particle_mlp.py` and `particle_transformer.py` to load a pytorch model, and simulate it on `LLVM` or `Vitis HLS` backend.

#### Example
A small example to deploy the JetFormer:
```python
import allo
import os
import numpy as np
import torch
from src.adjusted_model import JetFormer, SliceClsToken
# Add other libraries if needed

num_particles = 8
in_dim = 3  # num_feats
batch_size = 2
num_classes = 5

num_transformers = 4
embbed_dim = 8
num_heads = 2
dropout = 0

# Either define or load a model
model = JetFormer(
    in_dim, embbed_dim, num_heads, num_classes, num_transformers
).eval()

example_inputs = [torch.randn(batch_size, num_particles, in_dim)]
golden = model(*example_inputs).detach().numpy()
np_input = example_inputs[0].detach().numpy()

# LLVM
llvm_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    leaf_modules=(SliceClsToken,),
    verbose=True,
)

out = llvm_mod(np_input)
np.testing.assert_allclose(out, golden, atol=1e-5, rtol=1e-5)
print("Passed!")

# VITIS HLS
mode = "sw_emu"
os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
os.environ["XCL_EMULATION_MODE"] = mode

allo_out = np.zeros((batch_size, num_classes), dtype=np.float32)

vitis_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=example_inputs,
        leaf_modules=(SliceClsToken,),
        target="vitis_hls",
        mode=mode,
        verbose=False,
    )
vitis_mod(np_input, allo_out)
np.testing.assert_allclose(allo_out, golden, rtol=1e-5, atol=1e-5)
print("Passed!")
```


