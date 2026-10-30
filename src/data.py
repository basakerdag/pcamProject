import os
import h5py, numpy as np, torch
from torch.utils.data import Dataset

BASE = "/kaggle/input/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon"
XNAME = {"train": "training_split.h5", "valid": "validation_split.h5", "test": "test_split.h5"}


def _paths(root, split):
    if os.path.isdir(BASE):
        return (f"{BASE}/pcam/{XNAME[split]}",
                f"{BASE}/Labels/Labels/camelyonpatch_level_2_split_{split}_y.h5")
    return (f"{root}/camelyonpatch_level_2_split_{split}_x.h5",
            f"{root}/camelyonpatch_level_2_split_{split}_y.h5")


class PCamH5(Dataset):
    def __init__(self, root, split, transform=None, indices=None):
        self.x_path, self.y_path = _paths(root, split)
        self.transform = transform
        self._x = None
        with h5py.File(self.y_path, "r") as f:
            self.labels = f["y"][:].reshape(-1).astype(np.float32)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._x is None:
            self._x = h5py.File(self.x_path, "r")["x"]
        idx = self.indices[i]
        img = self._x[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(self.labels[idx])


def load_split(root, split, max_n=None):
    x_path, y_path = _paths(root, split)
    with h5py.File(y_path, "r") as f:
        y = f["y"][:].reshape(-1).astype(np.float32)
    with h5py.File(x_path, "r") as f:
        d = f["x"]
        n = d.shape[0] if max_n is None else min(max_n, d.shape[0])
        X = np.empty((n, 96, 96, 3), dtype=np.uint8)
        for i in range(0, n, 8192):
            j = min(i + 8192, n)
            X[i:j] = d[i:j]
    return X, y[:n]