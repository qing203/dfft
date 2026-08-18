import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cmapss_experiment import Model, WinDS, build_hi, read_train, split_units

OUT = Path("cmapss_futureops_results")
OUT.mkdir(exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mode_fops(hist, fops, mode):
    if mode == "true_future":
        return fops
    if mode == "hold_current":
        return hist[:, -1:, :3].repeat(1, fops.shape[1], 1)
    if mode == "no_future":
        return torch.zeros_like(fops)
    raise ValueError(mode)


def d_loss(raw, y, h0):
    true_cum = y - h0[:, None]
    pred = h0[:, None] + raw
    return ((raw - true_cum) ** 2).mean(), pred


def evaluate(model, loader, mode, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for hist, fops, y, h0 in loader:
            hist, fops, y, h0 = [x.to(device) for x in (hist, fops, y, h0)]
            raw = model(hist, mode_fops(hist, fops, mode))
            _, pred = d_loss(raw, y, h0)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    e = (p - y) * 100
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "LAST_MAE": float(np.mean(np.abs(e[:, -1]))),
    }


def train_one(df, tr, va, te, L, K, mode, seed, opsc, device):
    seed_all(seed)
    train = WinDS(df, tr, L, K, opsc)
    val = WinDS(df, va, L, K, opsc)
    test = WinDS(df, te, L, K, opsc)

    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(train, 256, shuffle=True, generator=gen)
    vl = DataLoader(val, 512)
    tl = DataLoader(test, 512)

    model = Model(32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = 1e99
    state = None
    bad = 0

    for ep in range(40):
        model.train()
        for hist, fops, y, h0 in dl:
            hist, fops, y, h0 = [x.to(device) for x in (hist, fops, y, h0)]
            opt.zero_grad()
            raw = model(hist, mode_fops(hist, fops, mode))
            loss, _ = d_loss(raw, y, h0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for hist, fops, y, h0 in vl:
                hist, fops, y, h0 = [x.to(device) for x in (hist, fops, y, h0)]
                raw = model(hist, mode_fops(hist, fops, mode))
                loss, _ = d_loss(raw, y, h0)
                vals.append(loss.item())
        v = float(np.mean(vals))
        if v < best - 1e-6:
            best = v
            state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 6:
            break

    model.load_state_dict(state)
    metrics = evaluate(model, tl, mode, device)
    metrics.update({"epochs": ep + 1, "n_train": len(train), "n_val": len(val), "n_test": len(test)})
    return metrics


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for fd in ["FD002", "FD004"]:
        raw = read_train(fd)
        tr, va, te = split_units(raw)
        df, _ = build_hi(raw, tr)
        opsc = StandardScaler().fit(df[df.unit.isin(tr)][["op1", "op2", "op3"]])

        for K in [3, 5, 10]:
            for mode in ["true_future", "hold_current", "no_future"]:
                metrics = train_one(df, tr, va, te, 20, K, mode, 11, opsc, device)
                row = {"dataset": fd, "K": K, "mode": mode, "seed": 11, **metrics}
                rows.append(row)
                print(row, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pilot_seed11.csv", index=False)
    print("\nFUTURE OPS ABLATION\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
