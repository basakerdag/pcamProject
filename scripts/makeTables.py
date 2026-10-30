import glob, json, os
import numpy as np, pandas as pd

os.makedirs("tablolar", exist_ok=True)
rows = []
for f in glob.glob("results/*.json"):
    name = os.path.basename(f)
    if name == "dataset_stats.json" or "_n20000" in name or "_smoke" in name:
        continue
    r = json.load(open(f, encoding="utf-8")); a = r["args"]; t = r["test"]
    if "_lf" in name: g = "low_label"
    elif name.startswith(("resnet18_", "efficientnet_b0_")): g = "imagenet"
    elif name.startswith(("logreg_", "mlp_")): g = "baseline"
    elif "_n50000" in name: g = "ablation"
    elif name.startswith("smallcnn_"): g = "smallcnn_full"
    else: continue
    method = None
    if g == "low_label":
        method = "imagenet" if "_imagenet" in name else ("ae" if "_ae" in name else "scratch")
    rows.append(dict(group=g, model=a["model"], aug=a["aug"], act=a["act"], opt=a["opt"],
                     seed=a["seed"], frac=a.get("label_frac", 0), method=method,
                     val_auc=r["best_val_auc"], thr=t["threshold"], bal_acc=t["balanced_acc"],
                     precision=t["precision"], recall=t["recall"], f1=t["f1"], auc=t["roc_auc"],
                     epochs_run=len(r["history"])))
df = pd.DataFrame(rows)
print("runs read from JSON:", len(df), "| groups:", df.group.value_counts().to_dict())

tx = pd.read_csv("scripts/augmentation_results.txt")
sc = df[df.group == "smallcnn_full"].copy()
print("\nSeed-0 consistency check (text file - JSON, largest absolute difference):")
for _, row in tx[tx.seed == 0].iterrows():
    j = sc[(sc.aug == row["mod"]) & (sc.seed == 0)].iloc[0]
    d = max(abs(j[k] - row[k2]) for k, k2 in [("bal_acc", "bal_acc"), ("precision", "precision"),
            ("recall", "recall"), ("f1", "f1"), ("auc", "auc"), ("thr", "esik")])
    print(f"  {row['mod']}: {d:.5f}")
extra = []
for _, row in tx[tx.seed != 0].iterrows():
    extra.append(dict(group="smallcnn_full", model="smallcnn", aug=row["mod"], act="relu", opt="adam",
                      seed=int(row["seed"]), frac=0, method=None, val_auc=np.nan, thr=row["esik"],
                      bal_acc=row["bal_acc"], precision=row["precision"], recall=row["recall"],
                      f1=row["f1"], auc=row["auc"], epochs_run=np.nan))
sc_all = pd.concat([sc, pd.DataFrame(extra)], ignore_index=True)

M = ["bal_acc", "precision", "recall", "f1", "auc"]
def ms(x):
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f}"
def summ(d, label):
    return {"Model": label, "n": len(d), **{m: ms(d[m]) for m in M}}

t1 = pd.DataFrame([
    summ(df[df.model == "logreg"], "Logistic Regression"),
    summ(df[df.model == "mlp"], "MLP"),
    summ(sc_all[sc_all.aug == "geo"], "Small CNN (geo)"),
    summ(df[(df.group == "imagenet") & (df.model == "resnet18")], "ResNet18"),
    summ(df[(df.group == "imagenet") & (df.model == "efficientnet_b0")], "EfficientNet-B0"),
])
t2 = pd.DataFrame([summ(sc_all[sc_all.aug == a], a) for a in ("none", "geo", "geo_color")])
ab = df[df.group == "ablation"]
t3 = pd.DataFrame([{"act": a, "opt": o, "n": len(g), **{m: ms(g[m]) for m in ["auc", "bal_acc", "f1", "val_auc"]}}
                   for (a, o), g in ab.groupby(["act", "opt"])])
ll = df[df.group == "low_label"]
t4 = pd.DataFrame([{"frac": fr, "method": me, "n": len(g), **{m: ms(g[m]) for m in ["auc", "bal_acc", "f1", "val_auc"]}}
                   for (fr, me), g in ll.groupby(["frac", "method"])])

for nm, t in [("table1_main_result", t1), ("table2_augmentation", t2), ("table3_ablation", t3), ("table4_low_label", t4)]:
    t.to_csv(f"tablolar/{nm}.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== {nm} ===")
    print(t.to_string(index=False))
df.to_csv("tablolar/all_runs.csv", index=False, encoding="utf-8-sig")