import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from cmapss_experiment import build_hi, read_train, split_units, train_one

OUT = Path("cmapss_pilot_results")
OUT.mkdir(exist_ok=True)


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    meta = {}

    for fd in ["FD002", "FD004"]:
        raw = read_train(fd)
        tr, va, te = split_units(raw)
        df, hmeta = build_hi(raw, tr)
        opsc = StandardScaler().fit(df[df.unit.isin(tr)][["op1", "op2", "op3"]])
        meta[fd] = {
            "n_rows": len(raw),
            "n_units": int(raw.unit.nunique()),
            "train_units": len(tr),
            "val_units": len(va),
            "test_units": len(te),
            "hi_meta": hmeta,
        }

        for K in [3, 5, 10]:
            for kind in ["B", "C", "D", "E"]:
                m = train_one(df, tr, va, te, 20, K, kind, 11, opsc, device)
                row = {"dataset": fd, "K": K, "model": kind, "seed": 11, **m}
                rows.append(row)
                print(row, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pilot_seed11.csv", index=False)
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nPILOT SUMMARY\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
