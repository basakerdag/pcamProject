import argparse, json, time, copy, os
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from src.utils import set_seed, get_device
from src.data import load_split
from src.metrics import pick_threshold, compute_metrics
from src.models import build_model

ROOT, OUT = "data", "results"

def to_gpu(xb, dev):
    return torch.from_numpy(xb).to(dev).permute(0, 3, 1, 2).float().div_(255).contiguous()

def augment(x, mode):
    """Augment a batch x of shape (B,3,H,W) in [0,1] on the GPU. mode: none | geo | geo_color"""
    if mode == "none":
        return x
    B = x.shape[0]
    dev = x.device
    hf = torch.rand(B, device=dev) < 0.5
    x = torch.where(hf[:, None, None, None], x.flip(3), x)
    vf = torch.rand(B, device=dev) < 0.5
    x = torch.where(vf[:, None, None, None], x.flip(2), x)
    k = torch.randint(0, 4, (B,), device=dev)
    for r in (1, 2, 3):
        x = torch.where((k == r)[:, None, None, None], torch.rot90(x, r, (2, 3)), x)
    if mode == "geo_color":
        u = lambda lo, hi: (torch.rand(B, 1, 1, 1, device=dev) * (hi - lo) + lo)
        x = x * u(0.8, 1.2)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mu) * u(0.8, 1.2) + mu
        gray = x.mean(dim=1, keepdim=True)
        x = gray + (x - gray) * u(0.8, 1.2)
        x = x + (torch.rand(B, 3, 1, 1, device=dev) - 0.5) * 0.10
        x = x.clamp(0, 1)
    return x

@torch.no_grad()
def predict(model, X, dev, mean, std, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = (to_gpu(X[i:i + bs], dev) - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="smallcnn")
    p.add_argument("--act", default="relu")
    p.add_argument("--opt", default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aug", default="none")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", type=int, default=0, help="0 = use the whole training set")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--norm", default="stats", choices=["stats", "imagenet"])
    p.add_argument("--freeze", type=int, default=0, help="keep the backbone frozen for the first N epochs")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--suffix", default="")
    p.add_argument("--label_frac", type=float, default=0.0, help="fraction of labels used (class-balanced subset)")
    p.add_argument("--init_ae", default="", help="path to the autoencoder encoder weights")
    a = p.parse_args()

    set_seed(a.seed)
    dev = get_device()
    use_amp = dev.type == "cuda"
    st = json.load(open(f"{OUT}/dataset_stats.json"))
    m_, s_ = (([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if a.norm == "imagenet"
              else (st["mean"], st["std"]))
    mean = torch.tensor(m_, device=dev).view(1, 3, 1, 1).float()
    std = torch.tensor(s_, device=dev).view(1, 3, 1, 1).float()

    print("Loading data...")
    Xtr, ytr = load_split(ROOT, "train")
    if a.subset:
        idx = np.random.RandomState(a.seed).choice(len(ytr), a.subset, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if a.label_frac:
        rs = np.random.RandomState(1000 + a.seed)
        sel = []
        for c in (0.0, 1.0):
            ci = np.where(ytr == c)[0]
            sel.append(rs.choice(ci, int(round(len(ci) * a.label_frac)), replace=False))
        idx = np.sort(np.concatenate(sel))
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"label fraction {a.label_frac}: {len(ytr)} samples, positive rate {ytr.mean():.3f}", flush=True)
    Xva, yva = load_split(ROOT, "valid")
    print(f"train {Xtr.shape}, valid {Xva.shape}")

    model = build_model(a.model, a.act).to(dev)
    if a.init_ae:
        model.features.load_state_dict(torch.load(a.init_ae, map_location=dev))
        print("AE encoder loaded:", a.init_ae, flush=True)
    head = None
    if a.freeze > 0:
        head = model.m.fc if a.model == "resnet18" else model.m.classifier
        head_ids = {id(q) for q in head.parameters()}
        body = [q for q in model.parameters() if id(q) not in head_ids]
        groups = [{"params": body, "lr": a.lr}, {"params": list(head.parameters()), "lr": a.lr_head}]
        for q in body:
            q.requires_grad_(False)
    else:
        groups = [{"params": list(model.parameters()), "lr": a.lr}]
    opt = (torch.optim.Adam(groups) if a.opt == "adam"
           else torch.optim.SGD(groups, momentum=0.9))
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tag = f"{a.model}_{a.aug}_{a.act}_{a.opt}_seed{a.seed}" + (f"_n{a.subset}" if a.subset else "") + a.suffix
    os.makedirs(f"{OUT}/ckpt", exist_ok=True); os.makedirs(f"{OUT}/preds", exist_ok=True)
    rng = np.random.RandomState(a.seed)
    best_auc, best_state, bad, hist = -1, None, 0, []

    for ep in range(1, a.epochs + 1):
        if head is not None and ep == a.freeze + 1:
            for q in model.parameters():
                q.requires_grad_(True)
            opt.param_groups[1]["lr"] = a.lr
            print("backbone unfrozen", flush=True)
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = augment(to_gpu(Xtr[idx], dev), a.aug)
            xb = (xb - mean) / std
            yb = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach() * len(idx)
        tot = float(tot)
        t_train = time.time() - t0
        pv = predict(model, Xva, dev, mean, std)
        auc = roc_auc_score(yva, pv)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_auc": float(auc)})
        print(f"ep {ep:02d} | loss {tot/len(perm):.4f} | val AUC {auc:.4f} | "
              f"train {t_train:.0f}s | total {time.time()-t0:.0f}s")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, f"{OUT}/ckpt/{tag}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping."); break

    model.load_state_dict(best_state)
    pv = predict(model, Xva, dev, mean, std)
    thr = pick_threshold(yva, pv)
    del Xtr
    Xte, yte = load_split(ROOT, "test")
    pt = predict(model, Xte, dev, mean, std)
    res = {"tag": tag, "args": vars(a), "best_val_auc": float(best_auc),
           "val": compute_metrics(yva, pv, thr), "test": compute_metrics(yte, pt, thr), "history": hist}
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    np.savez_compressed(f"{OUT}/preds/{tag}.npz", val=pv, yval=yva, test=pt, ytest=yte)
    t = res["test"]
    print(f"\nTEST (threshold {thr:.2f}): bal.acc {t['balanced_acc']:.4f} | prec {t['precision']:.4f} | "
          f"rec {t['recall']:.4f} | F1 {t['f1']:.4f} | AUC {t['roc_auc']:.4f}")

if __name__ == "__main__":
    main()