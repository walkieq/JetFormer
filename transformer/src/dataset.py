import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, h5_path, mean=None, std=None):
        self.h5_path = h5_path
        # Lazy loading, only open file when needed
        self._h5_file = None
        self.mean = mean
        self.std = std

    def _get_file(self):
        # Each DataLoader worker opens the file separately
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx):
        f = self._get_file()
        x = f["jetConstituentList"][idx]
        y = f["jets"][idx].argmax()

        # Normalize the data
        x = torch.from_numpy(x).float()
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        return x, torch.tensor(y).long()

    def __len__(self):
        return self._get_file()["jetConstituentList"].shape[0]

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    train_dataset = H5Dataset("../data/filtered_train.h5")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(len(train_loader))
