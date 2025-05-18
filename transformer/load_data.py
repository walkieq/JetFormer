import numpy as np
import h5py
import os
import time
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data PATH
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
# 620000 jets, 150 constituents, 17 features, 5 classes


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
            "X",
            shape=(total_samples, *sample_shape),
            dtype=np.float32,
            compression="gzip",  # gzip for archive
            chunks=True,
        )
        dset_y = f_out.create_dataset(
            "y",
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


def filter(name="train", batch_size=5000):
    if name == "train":
        input_path = os.path.join(DATA_DIR, "merged_train.h5")
        output_path = os.path.join(DATA_DIR, "filtered_train_1.h5")
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "merged_test.h5")
        output_path = os.path.join(DATA_DIR, "filtered_test.h5")

    pt_min = 2.0
    pt_index = 5

    start_time = time.time()

    with h5py.File(input_path, "r") as fin:
        X = fin["X"]
        y = fin["y"]
        n_samples, n_constit, n_feat = X.shape

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset_X = fout.create_dataset(
                "X",
                shape=(0, n_constit, n_feat),
                maxshape=(None, n_constit, n_feat),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )
            dset_y = fout.create_dataset(
                "y",
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


def customize_dataset(num_particles, feats: list = [5, 8, 11]):

    # TODO
    # saved file name? processed/30/test/test_3f.h5
    # tqdm + batch?
    # eg. 30 particles, 16 feats, 100 particles?

    # input_path = "data/processed/filtered_train.h5"
    # output_path = f"data/processed/{num_particles}/train_{len(feats)}f.h5"
    input_path = "data/processed/filtered_test.h5"
    output_path = f"data/processed/{num_particles}/test_{len(feats)}f.h5"

    print(f"Reading in batches...")
    batch_size = 5000

    with h5py.File(input_path, "r") as fin:
        X = fin["X"]
        y = fin["y"]
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

            print("Jets shape:", dset_X.shape)
            print("Targets shape:", dset_y.shape)


if __name__ == "__main__":
    # customize_dataset(30)

    # h5_path = "/vol/bitbucket/rz1224/ml_project/Transformer4Physics/data/processed/30/train_3f.h5"

    # from h5py import File as hdf5_file

    # with hdf5_file(h5_path, "r") as f:
    #     jetConstituentList = np.array(f["jetConstituentList"])
    #     jets = np.array(f["jets"])
    #     target = np.argmax(jets, axis=1)

    # print(f"jetConstituentList shape: {jetConstituentList.shape}")
    # print(f"jets shape: {jets.shape}")
    # print(f"target shape: {target.shape}")
    # print(f"target class counts: {np.unique(target, return_counts=True)}")

    filter(name="train")
