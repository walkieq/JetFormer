import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, "r")
        self.X = self.h5_file["jetConstituentList"]
        self.y = self.h5_file["jets"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx, :].argmax()
        return torch.from_numpy(x).float(), torch.tensor(y).long()

    def __del__(self):
        self.h5_file.close()


if __name__ == "__main__":
    train_dataset = H5Dataset("../data/filtered_train.h5")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(len(train_loader))
