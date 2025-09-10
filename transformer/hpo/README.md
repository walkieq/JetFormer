## Model Hyperparameter Optimization


#### Hyperparameter optimization
To run the hyperparameter optimization pipeline, simply run `optuna_main.py`.  
The file calls two main functions:
1. `prepare_dataset()` to split the training dataset into train and validation subsets. This only needs to run once, and the data will be saved in the `split_data` folder.
2. `optimization()` to start the optimization process. This will call `train_trial.py`.

#### Sampler comparison
Run `sampler_benchmark.py` for sampler comparison experiment. If data is already split into train and validation sets, no need to run `prepare_dataset()` (comment this out).

All the HPO results are saved in `optuna_results` folder.