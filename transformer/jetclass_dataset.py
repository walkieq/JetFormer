import torch
from torch.utils.data import Dataset
import numpy as np
import uproot
import awkward as ak


def _pad(a, maxlen, value=0, dtype="float32"):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, : len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def build_features_and_labels(tree, transform_features=True):
    a = tree.arrays(filter_name=["part_*", "jet_pt", "jet_energy", "label_*"])

    # Derived features
    a["part_mask"] = ak.ones_like(a["part_energy"])
    a["part_pt"] = np.hypot(a["part_px"], a["part_py"])
    a["part_pt_log"] = np.log(a["part_pt"])
    a["part_e_log"] = np.log(a["part_energy"])
    a["part_logptrel"] = np.log(a["part_pt"] / a["jet_pt"])
    a["part_logerel"] = np.log(a["part_energy"] / a["jet_energy"])
    a["part_deltaR"] = np.hypot(a["part_deta"], a["part_dphi"])
    a["part_d0"] = np.tanh(a["part_d0val"])
    a["part_dz"] = np.tanh(a["part_dzval"])

    if transform_features:
        a["part_pt_log"] = (a["part_pt_log"] - 1.7) * 0.7
        a["part_e_log"] = (a["part_e_log"] - 2.0) * 0.7
        a["part_logptrel"] = (a["part_logptrel"] - (-4.7)) * 0.7
        a["part_logerel"] = (a["part_logerel"] - (-4.7)) * 0.7
        a["part_deltaR"] = (a["part_deltaR"] - 0.2) * 4.0
        a["part_d0err"] = _clip(a["part_d0err"], 0, 1)
        a["part_dzerr"] = _clip(a["part_dzerr"], 0, 1)

    feature_list = [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_d0",
        "part_d0err",
        "part_dz",
        "part_dzerr",
        "part_deta",
        "part_dphi",
    ]
    pf_features = np.stack(
        [_pad(a[n], maxlen=128).to_numpy() for n in feature_list], axis=2
    )  # (N, 128, 17)
    pf_mask = _pad(a["part_mask"], maxlen=128).to_numpy().astype(np.float32)  # (N, 128)

    label_list = [
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ]
    labels = np.stack(
        [a[n].to_numpy().astype("int") for n in label_list], axis=1
    )  # (N, 10)
    labels = labels.argmax(axis=1)  # Convert to class index

    return pf_features, pf_mask, labels


class JetClassDataset(Dataset):
    def __init__(self, root_file_path, transform_features=True):
        tree = uproot.open(root_file_path)["tree"]
        self.features, self.masks, self.labels = build_features_and_labels(
            tree, transform_features
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # shape: (128, 17)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # scalar class label
        return x, y
