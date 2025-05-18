import numpy as np
import h5py
import os
import time
from tqdm import tqdm


# Data PATH
TRAIN_PATH = "data/train/"
TEST_PATH = "data/test/"
# file_list = sorted(os.listdir(TRAIN_PATH))
file_list = sorted(os.listdir(TEST_PATH))

# 620000 jets, 150 constituents, 17 features, 5 classes

def read_h5_files():
    # output_path = "data/processed/merged_train.h5"
    output_path = "data/processed/merged_test.h5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f_out:
        dset_X = None
        dset_y = None

        for file in file_list:
            print(f"Appending {file} ...")
            # with h5py.File(TRAIN_PATH + file, "r") as data:
            with h5py.File(TEST_PATH + file, "r") as data:
                jets = data["jetConstituentList"]
                targets = data["jets"][:, -6:-1]

                batch = jets.shape[0]

                if dset_X is None:
                    dset_X = f_out.create_dataset(
                        "X",
                        shape=(batch, *jets.shape[1:]),
                        maxshape=(None, *jets.shape[1:]),
                        dtype=np.float32,
                        compression="gzip",
                        chunks=True
                    )
                    dset_y = f_out.create_dataset(
                        "y",
                        shape=(batch, targets.shape[1]),
                        maxshape=(None, targets.shape[1]),
                        dtype=np.float32,
                        compression="gzip",
                        chunks=True
                    )
                    dset_X[...] = jets[...]
                    dset_y[...] = targets[...]
                else:
                    dset_X.resize(dset_X.shape[0] + batch, axis=0)
                    dset_y.resize(dset_y.shape[0] + batch, axis=0)

                    dset_X[-batch:, ...] = jets[...]
                    dset_y[-batch:, ...] = targets[...]

        print("Final shape:", dset_X.shape, dset_y.shape)


def filter():
    # input_path = "data/processed/merged_train.h5"
    # output_path = "data/processed/filtered_train.h5"
    input_path = "data/processed/merged_test.h5"
    output_path = "data/processed/filtered_test.h5"
    pt_min = 2.0
    pt_index = 5


    with h5py.File(input_path, "r") as fin:
        X = fin["X"]
        y = fin["y"]
        n_samples, n_constit, n_feat = X.shape
        batch_size = 5000

        jets_buffer = []
        targets_buffer = []

        for start in tqdm(range(0, n_samples, batch_size), desc="Filtering"):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]  # shape: (batch, n_constit, n_feat)
            batch_y = y[start:end]

            for i in range(batch_X.shape[0]):
                jet = batch_X[i]
                if np.any(jet[:, pt_index] >= pt_min):
                    jets_buffer.append(jet)
                    targets_buffer.append(batch_y[i])

        jets_out = np.stack(jets_buffer)  # shape: (n_valid, n_constit, n_feat)
        targets_out = np.stack(targets_buffer)

        start_time = time.time()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            fout.create_dataset("X", data=jets_out, compression="lzf", chunks=True)
            fout.create_dataset("y", data=targets_out, compression="lzf", chunks=True)

        print(f"Saved filtered result to {output_path}")
        print(f"Filtered jets shape: {jets_out.shape}")
        print(f"Filtered targets shape: {targets_out.shape}")

    end_time = time.time()
    print("time:", end_time - start_time, "s")


def customize_dataset(num_particles, feats: list=[5, 8, 11]):
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
                compression="lzf"
            )
            dset_y = fout.create_dataset(
                "jets",
                shape=(0, y.shape[1]),
                maxshape=(None, y.shape[1]),
                dtype=np.float32,
                chunks=True,
                compression="lzf"
            )

            write_idx = 0
            for start in tqdm(range(0, n_samples, batch_size), desc="Cropping"):
                end = min(start + batch_size, n_samples)

                batch_X = X[start:end, :num_particles, :][:, :, feats]
                batch_y = y[start:end]

                batch_len = batch_X.shape[0]

                dset_X.resize(write_idx + batch_len, axis=0)
                dset_y.resize(write_idx + batch_len, axis=0)

                dset_X[write_idx:write_idx + batch_len] = batch_X
                dset_y[write_idx:write_idx + batch_len] = batch_y

                write_idx += batch_len

            print("Jets shape:", dset_X.shape)
            print("Targets shape:", dset_y.shape)





if __name__ == "__main__":
    # customize_dataset(30)

    h5_path = "/vol/bitbucket/rz1224/ml_project/Transformer4Physics/data/processed/30/train_3f.h5"

    from h5py import File as hdf5_file

    with hdf5_file(h5_path, 'r') as f:
        jetConstituentList = np.array(f['jetConstituentList'])
        jets = np.array(f['jets'])
        target = np.argmax(jets, axis=1)

    print(f"jetConstituentList shape: {jetConstituentList.shape}")
    print(f"jets shape: {jets.shape}")
    print(f"target shape: {target.shape}")
    print(f"target class counts: {np.unique(target, return_counts=True)}")
