import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

os.makedirs("figures_en", exist_ok=True)
MODELS = [("Logistic Regression", "logreg_none_relu_adam_seed0"),
          ("MLP", "mlp_none_relu_adam_seed0"),
          ("Small CNN (geo)", "smallcnn_geo_relu_adam_seed0"),
          ("ResNet18", "resnet18_geo_relu_adam_seed0"),
          ("EfficientNet-B0", "efficientnet_b0_geo_relu_adam_seed0")]
R = {n: json.load(open(f"results/{t}.json", encoding="utf-8")) for n, t in MODELS}

plt.figure(figsize=(5.6, 5.2))
for n, t in MODELS:
    d = np.load(f"results/preds/{t}.npz")
    fpr, tpr, _ = roc_curve(d["ytest"], d["test"])
    plt.plot(fpr, tpr, label=f"{n} (AUC {R[n]['test']['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k:", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves, test set (seed 0)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("figures_en/roc_test_seed0.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
for ax, (n, t) in zip(axes, MODELS):
    te = R[n]["test"]
    cm = np.array([[te["tn"], te["fp"]], [te["fn"], te["tp"]]])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["negative", "positive"]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_title(f"{n}\nthreshold {te['threshold']:.2f}", fontsize=9)
axes[0].set_ylabel("True")
plt.suptitle("Confusion matrices, test set (seed 0, row percentages)", y=1.04)
plt.tight_layout(); plt.savefig("figures_en/confusion_test_seed0.png", dpi=150, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n, t in MODELS:
    h = R[n]["history"]; ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], marker="o", ms=3, label=n)
    ax[1].plot(ep, [x["val_auc"] for x in h], marker="o", ms=3, label=n)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Training loss (seed 0)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation AUC"); ax[1].set_title("Validation AUC (seed 0)")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures_en/training_curves_seed0.png", dpi=150); plt.close()

df = pd.read_csv("tablolar/all_runs.csv"); ll = df[df.group == "low_label"]
names = {"scratch": "Scratch (small CNN)", "ae": "AE init. (small CNN)", "imagenet": "ImageNet (ResNet18)"}
plt.figure(figsize=(5.8, 4.4))
for m, lab in names.items():
    g = ll[ll.method == m].groupby("frac")["auc"].agg(["mean", lambda x: x.std(ddof=1)])
    g.columns = ["mean", "std"]
    plt.errorbar([1, 10], g["mean"].values, yerr=g["std"].values, marker="o", capsize=4, label=lab)
plt.xscale("log"); plt.xticks([1, 10], ["1%", "10%"])
plt.xlabel("Fraction of labels used"); plt.ylabel("Test ROC-AUC")
plt.title("Low-label experiment (3 seeds, mean ± std)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures_en/low_label_auc.png", dpi=150); plt.close()
print("Figures:", sorted(os.listdir("figures_en")))