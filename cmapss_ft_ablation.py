import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cmapss_experiment import WinDS, build_hi, read_train, split_units

OUT = Path("cmapss_ft_results")
OUT.mkdir(exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ft(hist):
    """Four causal degradation features from the historical HI window only.

    hist[..., 3] is normalized HI in [0, 1]. Features:
    1) net HI drop over the window;
    2) mean downward HI change magnitude;
    3) longest consecutive-decline run ratio;
    4) short-term HI-change fluctuation (std of first differences).
    """
    hi = hist[:, :, 3]
    dhi = hi[:, 1:] - hi[:, :-1]

    net_drop = hi[:, 0] - hi[:, -1]
    mean_down = torch.relu(-dhi).mean(dim=1)

    run = torch.zeros_like(net_drop)
    best = torch.zeros_like(net_drop)
    zeros = torch.zeros_like(net_drop)
    for j in range(dhi.shape[1]):
        run = torch.where(dhi[:, j] < 0, run + 1.0, zeros)
        best = torch.maximum(best, run)
    longest_ratio = best / float(max(dhi.shape[1], 1))

    fluct = dhi.std(dim=1, unbiased=False)
    return torch.stack([net_drop, mean_down, longest_ratio, fluct], dim=1)


class FtModel(nn.Module):
    """Same architecture/parameter count for both modes.

    Decoder always receives 3 future operating-condition variables plus 4 Ft slots.
    In no_ft mode the 4 Ft slots are zero, so only information content changes.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.GRU(4, hidden, batch_first=True)
        self.dec = nn.GRU(7, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, hist, fops, mode):
        _, h = self.enc(hist)
        if mode == "with_ft":
            ft = build_ft(hist)
        elif mode == "no_ft":
            ft = torch.zeros(hist.shape[0], 4, dtype=hist.dtype, device=hist.device)
        else:
            raise ValueError(mode)
        ft_seq = ft[:, None, :].expand(-1, fops.shape[1], -1)
        dec_in = torch.cat([fops, ft_seq], dim=2)
        z, _ = self.dec(dec_in, h)
        return self.head(z).squeeze(-1)


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
            raw = model(hist, fops, mode)
            _, pred = d_loss(raw, y, h0)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    e = (p - y) * 100.0
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

    model = FtModel(32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = 1e99
    state = None
    bad = 0

    for ep in range(40):
        model.train()
        for hist, fops, y, h0 in dl:
            hist, fops, y, h0 = [x.to(device) for x in (hist, fops, y, h0)]
            opt.zero_grad()
            raw = model(hist, fops, mode)
            loss, _ = d_loss(raw, y, h0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for hist, fops, y, h0 in vl:
                hist, fops, y, h0 = [x.to(device) for x in (hist, fops, y, h0)]
                raw = model(hist, fops, mode)
                loss, _ = d_loss(raw, y, h0)
                vals.append(loss.item())
        v = float(np.mean(vals))
        if v < best - 1e-6:
            best = v
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 6:
            break

    model.load_state_dict(state)
    m = evaluate(model, tl, mode, device)
    m["epochs"] = ep + 1
    m["n_train"] = len(train)
    m["n_val"] = len(val)
    m["n_test"] = len(test)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["FD002", "FD004"])
    ap.add_argument("--k", type=int, required=True, choices=[3, 5, 10])
    ap.add_argument("--seed", type=int, required=True, choices=[11, 22, 33])
    args = ap.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = read_train(args.dataset)
    tr, va, te = split_units(raw)
    df, _ = build_hi(raw, tr)
    opsc = StandardScaler().fit(df[df.unit.isin(tr)][["op1", "op2", "op3"]])

    rows = []
    for mode in ["no_ft", "with_ft"]:
        m = train_one(df, tr, va, te, 20, args.k, mode, args.seed, opsc, device)
        row = {
            "dataset": args.dataset,
            "K": args.k,
            "seed": args.seed,
            "mode": mode,
            **m,
        }
        rows.append(row)
        print(row, flush=True)

    out = pd.DataFrame(rows)
    path = OUT / f"{args.dataset}_K{args.k}_seed{args.seed}.csv"
    out.to_csv(path, index=False)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
