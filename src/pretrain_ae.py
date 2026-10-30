import argparse, json, time, os
import numpy as np, torch
from src.utils import set_seed, get_device
from src.data import load_split
from src.models import AutoEncoder

OUT = "results"

def to_gpu(xb, dev):
    return torch.from_numpy(xb).to(dev).permute(0, 3, 1, 2).float().div_(255).contiguous()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--act", default="relu")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--suffix", default="")
    a = p.parse_args()

    set_seed(a.seed)
    dev = get_device()
    use_amp = dev.type == "cuda"
    st = json.load(open(f"{OUT}/dataset_stats.json"))
    mean = torch.tensor(st["mean"], device=dev).view(1, 3, 1, 1).float()
    std = torch.tensor(st["std"], device=dev).view(1, 3, 1, 1).float()

    print("Loading data...", flush=True)
    Xtr, _ = load_split("data", "train")
    if a.subset:
        Xtr = Xtr[np.random.RandomState(a.seed).choice(len(Xtr), a.subset, replace=False)]
    Xva, _ = load_split("data", "valid", max_n=8192)
    print(f"train {Xtr.shape}, valid (monitoring) {Xva.shape}", flush=True)

    model = AutoEncoder(a.act).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.RandomState(a.seed)
    os.makedirs(f"{OUT}/ae", exist_ok=True)
    tag = f"ae_seed{a.seed}{a.suffix}"
    best, hist = 1e9, []

    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = (to_gpu(Xtr[idx], dev) - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xb)
            loss = ((out.float() - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach() * len(idx)
        tr = float(tot) / len(perm)
        model.eval(); vt = 0.0
        with torch.no_grad():
            for i in range(0, len(Xva), 512):
                xb = (to_gpu(Xva[i:i + 512], dev) - mean) / std
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(xb)
                vt += float(((out.float() - xb) ** 2).mean()) * len(xb)
        va = vt / len(Xva)
        hist.append({"epoch": ep, "train_mse": tr, "val_mse": va})
        print(f"ep {ep:02d} | train MSE {tr:.4f} | val MSE {va:.4f} | {time.time()-t0:.0f}s", flush=True)
        if va < best:
            best = va
            torch.save(model.encoder.state_dict(), f"{OUT}/ae/{tag}_encoder.pt")
    json.dump({"tag": tag, "args": vars(a), "best_val_mse": best, "history": hist},
              open(f"{OUT}/ae/{tag}.json", "w"), indent=2)
    print(f"Saved: {OUT}/ae/{tag}_encoder.pt (best val MSE {best:.4f})", flush=True)

if __name__ == "__main__":
    main()