import numpy as np
import h5py
import os
import time
from tqdm import tqdm
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data PATH
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
# Train: 620000 jets, 150 constituents, 17 features, 5 classes
# Test: 260000 jets, 150 constituents, 17 features, 5 classes


def read_h5_files(name="train", batch_size=5000):
    if name == "train":
        input_path = os.path.join(DATA_DIR, "train")
        output_path = os.path.join(DATA_DIR, "merged_train.h5")
        file_list = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "test")
        output_path = os.path.join(DATA_DIR, "merged_test.h5")
        file_list = sorted(os.listdir(os.path.join(DATA_DIR, "test")))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()

    # Count total samples and shape
    total_samples = 0
    sample_shape = None
    target_shape = None
    for file in file_list:
        with h5py.File(os.path.join(input_path, file), "r") as data:
            jets = data["jetConstituentList"]
            targets = data["jets"][:, -6:-1]
            total_samples += jets.shape[0]
            if sample_shape is None:
                sample_shape = jets.shape[1:]
                target_shape = targets.shape[1:]

    # Initialize datasets
    with h5py.File(output_path, "w") as f_out:
        dset_X = f_out.create_dataset(
            "jetConstituentList",
            shape=(total_samples, *sample_shape),
            dtype=np.float32,
            compression="gzip",  # gzip for archive
            chunks=True,
        )
        dset_y = f_out.create_dataset(
            "jets",
            shape=(total_samples, *target_shape),
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )

        write_idx = 0

        with tqdm(file_list, desc="Merging h5 files") as t:
            for file in t:
                t.set_postfix(file=file)
                with h5py.File(os.path.join(input_path, file), "r") as data:
                    jets = data["jetConstituentList"][...]
                    targets = data["jets"][:, -6:-1]
                    num = jets.shape[0]
                    for start in range(0, num, batch_size):
                        end = min(start + batch_size, num)
                        dset_X[write_idx : write_idx + (end - start)] = jets[start:end]
                        dset_y[write_idx : write_idx + (end - start)] = targets[
                            start:end
                        ]
                        write_idx += end - start

        print("Final shape:", dset_X.shape, dset_y.shape)
        print(f"Saved merged result to {output_path}")

        end_time = time.time()
        print("Time taken:", end_time - start_time, "s")


def filter(name="train", batch_size=5000):
    if name == "train":
        input_path = os.path.join(DATA_DIR, "merged_train.h5")
        output_path = os.path.join(DATA_DIR, "filtered_train.h5")
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "merged_test.h5")
        output_path = os.path.join(DATA_DIR, "filtered_test.h5")

    pt_min = 2.0
    pt_index = 5

    start_time = time.time()

    with h5py.File(input_path, "r") as fin:
        X = fin["jetConstituentList"]
        y = fin["jets"]
        n_samples, n_constit, n_feat = X.shape

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset_X = fout.create_dataset(
                "jetConstituentList",
                shape=(0, n_constit, n_feat),
                maxshape=(None, n_constit, n_feat),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )
            dset_y = fout.create_dataset(
                "jets",
                shape=(0, y.shape[1]),
                maxshape=(None, y.shape[1]),
                dtype=np.float32,
                chunks=True,
                compression="lzf",  # lzf for speed
            )

            write_idx = 0
            for start in tqdm(range(0, n_samples, batch_size), desc="Filtering"):
                end = min(start + batch_size, n_samples)
                batch_X = X[start:end]  # shape: (batch, n_constit, n_feat)
                batch_y = y[start:end]

                mask = np.any(batch_X[:, :, pt_index] >= pt_min, axis=1)
                filtered_X = batch_X[mask]
                filtered_y = batch_y[mask]
                batch_len = filtered_X.shape[0]

                if batch_len == 0:
                    continue

                dset_X.resize(write_idx + batch_len, axis=0)
                dset_y.resize(write_idx + batch_len, axis=0)
                dset_X[write_idx : write_idx + batch_len] = filtered_X
                dset_y[write_idx : write_idx + batch_len] = filtered_y
                write_idx += batch_len

            print(f"Saved filtered result to {output_path}")
            print(f"Filtered jets shape: {dset_X.shape}")
            print(f"Filtered targets shape: {dset_y.shape}")

    end_time = time.time()
    print("Time taken:", end_time - start_time, "s")


def customize_dataset(num_particles, feats: list = [5, 8, 11], name="train"):

    assert num_particles <= 150, "num_particles should be less than or equal to 150"
    assert len(feats) <= 16, "feats should be less than or equal to 16"

    if name == "train":
        input_path = os.path.join(DATA_DIR, "filtered_train.h5")
        output_path = os.path.join(
            PROCESSED_DIR, str(num_particles), f"{len(feats)}f", "train.h5"
        )
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "filtered_test.h5")
        output_path = os.path.join(
            PROCESSED_DIR, str(num_particles), f"{len(feats)}f", "test.h5"
        )

    batch_size = 5000

    with h5py.File(input_path, "r") as fin:
        X = fin["jetConstituentList"]
        y = fin["jets"]
        n_samples, n_constit, n_feat = X.shape

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset_X = fout.create_dataset(
                "jetConstituentList",
                shape=(0, num_particles, len(feats)),
                maxshape=(None, num_particles, len(feats)),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )
            dset_y = fout.create_dataset(
                "jets",
                shape=(0, y.shape[1]),
                maxshape=(None, y.shape[1]),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )

            write_idx = 0
            for start in tqdm(range(0, n_samples, batch_size), desc="Cropping"):
                end = min(start + batch_size, n_samples)

                batch_X = X[start:end, :num_particles, :][:, :, feats]
                batch_y = y[start:end]

                batch_len = batch_X.shape[0]

                dset_X.resize(write_idx + batch_len, axis=0)
                dset_y.resize(write_idx + batch_len, axis=0)

                dset_X[write_idx : write_idx + batch_len] = batch_X
                dset_y[write_idx : write_idx + batch_len] = batch_y

                write_idx += batch_len

            print(f"JetConstituentList shape: {dset_X.shape}")
            print(f"Jets shape: {dset_y.shape}")

            # labels = dset_y[...]
            # labels = np.argmax(labels, axis=1)
            # unique, counts = np.unique(labels, return_counts=True)
            # for cls, cnt in zip(unique, counts):
            #     print(f"Class {cls}: {cnt} samples")

            print(f"Saved customized result to {output_path}")


def fetch_hls4ml_dataset(test_ratio: float = 0.2) -> None:
    # Train: 664000, test: 166000
    X, y = fetch_openml("hls4ml_lhc_jets_hlf", cache=True, return_X_y=True)
    assert len(X) == len(y)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = torch.from_numpy(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

    np.save(os.path.join(PROCESSED_DIR, "1/train", "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "1/test", "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "1/train", "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "1/test", "y_test.npy"), y_test)
    np.save(os.path.join(PROCESSED_DIR, "1/classes.npy"), le.classes_)

    print(f"Fetched hls4ml_lhc_jets_hlf dataset to {PROCESSED_DIR + '/1'}")

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    # unique, counts = np.unique(y_train, return_counts=True)
    print(f"Classes: {le.classes_}")


if __name__ == "__main__":

    feats = range(16)
    # customize_dataset(num_particles=16, feats=feats, name="test")
