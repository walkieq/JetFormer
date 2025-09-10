# Transformer4Physics

Transformer4Physics is a deep learning project based on the Transformer architecture for jet tagging.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/walkieq/Transformer4Physics.git
    cd Transformer4Physics
    ```
2. Install dependencies (for rtx 3090):
    ```bash
    conda env create -f environment_3090.yml -n [env_name]
    ```
3. Install dependencies for Allo deployment:
    ```bash
    conda env create -f environment_allo.yml -n [env_name]
    ```
    This environment is only for files in `transformer/hls` folder.


## Project Structure

```
Transformer4Physics/
├── transformer/
│   ├── compress/
│   │   ├── benchmark_latency.py     # Estimate latency for original and pruned models
│   │   ├── estimate_size.py         # Estimate size for original and quantized models
│   │   ├── prune.py                 # Structured pruning pipeline
│   │   ├── quant.py                 # 1-bit quantization pipeline
│   │   └── train_models.py          # Train the best trial models from hpo
│   ├── hls/
│   │   ├── particle_mlp.py          # Deploy particle mlp
│   │   ├── particle_transformer.py  # Deploy JetFormer
│   │   └── train_mlp.py             # Train a particle mlp example
│   ├── hpo/
│   │   ├── best_trials.csv          # Hyperparameter results from best trials
│   │   ├── optuna_main.py           # Hyperparameter optimization pipeline
│   │   ├── sampler_benchmark.py     # Comparison of three samplers
│   │   ├── train_trial.py           # Single search process (initiated by optuna_main or sampler_benchmark)
│   │   └── visualize_8p3f.ipynb
│   ├── data/
│   │   ├── processed/               # Customized datasets
│   │   ├── test/                    # Extracted hls4ml_LHCjet_150p_val.tar.gz
│   │   └── train/                   # Extracted hls4ml_LHCjet_150p_train.tar.gz
│   ├── JetClass/2M                  # Data needs to be downloaded from JetClass repo
│   │   ├── test/
│   │   └── train/
│   ├── models/                      # Saved models
│   ├── outputs/                     # Training plots
│   ├── src/                         # Model and Dataset definition
│   │   ├── layer.py
│   │   ├── net.py
│   │   ├── dataset.py
│   │   ├── adjusted_model.py        # Model used for Allo deployment
│   │   └── bit_model.py             # Model used for quantization
│   ├── build_dataset.py             # Customized dataset script
│   ├── train.py                     # Main training script for 150p dataset
│   ├── jetclass_dataset.py          # Proprocessing for JetClass dataset
│   └── jetclass_train.py            # Main training script for JetClass dataset
├── environment_3090.yml
├── environment_allo.yml
└── README.md
```


## Dataset
HLS4ML HLC Jets HLF dataset: https://www.openml.org/search?type=data&sort=runs&id=42468&status=active.  
No need to download this dataset manually. It can be automatically fetched by `--build_dataset`.

HLS4ML LHC Jet dataset (150 particles): https://zenodo.org/records/3602260.  
Please download the dataset and extract the two under `transformer/data/train/` and `transformer/data/test/` directories.

JetClass: https://github.com/jet-universe/particle_transformer.  
Please download the data from `data/JetClass/` by running `get_datasets.py` from the particle_transformer repository.


## Usage

### Model Training

#### Command-line Arguments for `train.py`
- `--num_particles`: Number of particles per sample (default: 30)
- `--num_feats`: Number of features per particle (choices: 3 or 16, default: 16)
- `--num_epochs`: Number of training epochs (default: 25)
- `--early_stopping_patience`: Early stopping patience (default: 4)
- `--num_transformers`: Number of transformer layers (default: 3)
- `--embbed_dim`: Embedding dimension (default: 64)
- `--num_heads`: Number of attention heads (default: 2)
- `--batch_size`: Batch size (default: 256)
- `--dropout`: Dropout rate (default: 0.0)
- `--seed`: Random seed (default: 20)
- `--save`: Save models and outputs
- `--build_dataset`: Build the dataset automatically

#### Train the model
To train the model on a dataset for the first time, add `--build_dataset`. It may take some time to create the dataset. The created dataset will be stored in `data/processed/` folder. If the dataset is already stored, no need to add `--build_dataset`.  

To run on the HLF dataset, use `--num_particles 1` and `--num_feats 16`:
```bash
cd transformer
python3 train.py --num_particles 1 --num_feats 16 --save
```

To run on a customized 150-particle dataset, if `num_feats` is
- 3: only Transverse momentum (pt), relative pseudorapidity (eta) and relative azimuthal angle (phi) are selected,
- 16: all features are selected.

For example, to train on 30-particle 16-feature dataset:
```bash
cd transformer
python3 train.py --num_particles 30 --num_feats 16 --save
```

### [Model Hyperparameter Optimization](transformer/hpo/README.md)

### [Model Compression](transformer/compress/README.md)

### [Model Deployment](transformer/hls/README.md)

