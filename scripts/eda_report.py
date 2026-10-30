import os, sys
sys.path.insert(0, os.getcwd())
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import _paths

ROOT, OUT = "data", "figures_en"
os.makedirs(OUT, exist_ok=True)


def labels(split):
    with h5py.File(_paths(ROOT, split)[1], "r") as f:
        return f["y"][:].reshape(-1)


def load_idx(split, idx):
    idx = np.sort(idx)
    with h5py.File(_paths(ROOT, split)[0], "r") as f:
        return f["x"][idx]


for split in ["train", "valid", "test"]:
    y = labels(split)
    n, p = len(y), int(y.sum())
    print(f"{split:6s} n={n:6d} | positive {p:6d} ({p / n:.4f}) | negative {n - p:6d} ({(n - p) / n:.4f})")

rng = np.random.RandomState(0)

ytr = labels("train")
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for r, c in enumerate([0, 1]):
    pick = rng.choice(np.where(ytr == c)[0], 8, replace=False)
    X = load_idx("train", pick)
    for j in range(8):
        axes[r, j].imshow(X[j])
        axes[r, j].axis("off")
    axes[r, 0].set_title("negative" if c == 0 else "positive", fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_sample_patches.png", dpi=130)
plt.close()

cols = ["tab:red", "tab:green", "tab:blue"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for ax, split in zip(axes, ["train", "valid", "test"]):
    n = len(labels(split))
    X = load_idx(split, rng.choice(n, 4000, replace=False))
    m = X.reshape(len(X), -1, 3).mean(axis=1)
    for ch, col, nm in zip(range(3), cols, "RGB"):
        ax.hist(m[:, ch], bins=50, range=(0, 255), alpha=0.5, color=col, label=nm)
    ax.set_title(f"{split} (n=4000)")
    ax.set_xlabel("per-image mean channel value")
    print(split, "mean RGB:", np.round(m.mean(0), 1), "| std:", np.round(m.std(0), 1))
axes[0].legend()
axes[0].set_ylabel("number of images")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_color_histogram.png", dpi=130)
plt.close()