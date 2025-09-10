## Model Compression


#### Structured Pruning
Run `prune.py` for structured pruning. The following parameters can be experimented:
- `model_index`: Index of the model to prune (refers to best_trials.csv).
- `pruning_ratio`: Overall pruning ratio (0 < pruning_ratio < 1).
- `iterative_steps`: Number of iterative pruning steps.
- `finetune_epochs`: Number of epochs for fine-tuning after each pruning step.

`benchmark_latency.py` estimates the inferece time of the original and pruned models.

#### 1-Bit Quantization
Run `quant.py` for 1-bit quantization.
`estimatez_size.py` estimates the size of the original and quantized models.

All results are saved in `tmp/` folder.