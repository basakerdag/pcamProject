import json, h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import PCamH5

ROOT, OUT = "data", "results"
rng = np.random.default_rng(0)

def read(split, idx):
    idx = np.sort(idx)
    with h5py.File(f"{ROOT}/camelyonpatch_level_2_split_{split}_x.h5", "r") as f:
        return f["x"][idx]

train = PCamH5(ROOT, "train")
y = train.labels

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for r, (cls, name) in enumerate([(0, "Negative"), (1, "Positive")]):
    idx = rng.choice(np.where(y == cls)[0], 8, replace=False)
    imgs = read("train", idx)
    for c in range(8):
        ax = axes[r, c]
        ax.imshow(imgs[c]); ax.axis("off")
        ax.add_patch(plt.Rectangle((32, 32), 32, 32, fill=False, ec="lime", lw=1.2))
    axes[r, 0].set_title(name, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_samples.png", dpi=150); plt.close()

n, s, s2 = 0, np.zeros(3), np.zeros(3)
with h5py.File(f"{ROOT}/camelyonpatch_level_2_split_train_x.h5", "r") as f:
    X = f["x"]
    for i in range(0, X.shape[0], 4096):
        b = X[i:i+4096].astype(np.float64) / 255.0
        n += b.shape[0] * 96 * 96
        s += b.sum(axis=(0, 1, 2)); s2 += (b ** 2).sum(axis=(0, 1, 2))
mean = s / n; std = np.sqrt(s2 / n - mean ** 2)
print("mean (RGB):", mean.round(4), "std (RGB):", std.round(4))

idx = np.sort(rng.choice(len(y), 4000, replace=False))
imgs, lab = read("train", idx), y[idx]
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for c, col in enumerate(["red", "green", "blue"]):
    for cls, ls in [(0, "-"), (1, "--")]:
        h, _ = np.histogram(imgs[lab == cls][..., c], bins=64, range=(0, 255), density=True)
        axes[c].plot(np.linspace(0, 255, 64), h, color=col, ls=ls, label=["Negative", "Positive"][cls])
    axes[c].set_title(f"{col} channel"); axes[c].legend()
m = imgs.reshape(len(imgs), -1, 3).mean(1)
axes[3].scatter(m[:, 0], m[:, 2], c=lab, s=3, cmap="coolwarm", alpha=0.5)
axes[3].set_xlabel("Mean R"); axes[3].set_ylabel("Mean B"); axes[3].set_title("Per-image mean colour")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_color.png", dpi=150); plt.close()

stats = {"n": {s_: len(PCamH5(ROOT, s_)) for s_ in ["train", "valid", "test"]},
         "pos_rate": {s_: float(PCamH5(ROOT, s_).labels.mean()) for s_ in ["train", "valid", "test"]},
         "mean": mean.tolist(), "std": std.tolist()}
json.dump(stats, open(f"{OUT}/dataset_stats.json", "w"), indent=2)