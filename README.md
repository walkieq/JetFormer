# Transformer4Physics

Transformer4Physics is a deep learning project based on the Transformer architecture for jet tagging.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Transformer4Physics.git
    cd Transformer4Physics
    ```
2. Install dependencies (for rtx 3090):
    ```bash
    conda env create -f environment_3090.yml -n [env_name]
    ```

## Project Structure

```
Transformer4Physics/
├── transformer/
│   ├── data/
│   │   ├── processed/             # Customized datasets
│   │   ├── test/                  # Extracted hls4ml_LHCjet_150p_val.tar.gz
│   │   └── train/                 # Extracted hls4ml_LHCjet_150p_train.tar.gz
│   ├── models/                    # Saved models
│   ├── outputs/                   # Training plots
│   ├── src/                       # Model and Dataset definition
│   │   ├── layer.py
│   │   ├── net.py
│   │   └── dataset.py 
│   ├── build_dataset.py           # Customized dataset script
│   └── train.py                   # Main training script
├── environment_3090.yml           # Conda environment file for RTX 3090
├── environment_4080.yml           # Conda environment file for RTX 4080
└── README.md                      # Project documentation
```

## Dataset
HLS4ML HLC Jets HLF dataset: https://www.openml.org/search?type=data&sort=runs&id=42468&status=active.  
No need to download this dataset manually. It can be automatically fetched by `--build_dataset`.

HLS4ML LHC Jet dataset (150 particles): https://zenodo.org/records/3602260.  
Please download the dataset and extract the two under `transformer/data/train/` and `transformer/data/test/` directories.

## Usage

### Command-line Arguments

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

### Train the model
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
