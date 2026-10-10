"""Export the latest run to a static dashboard (static/), for a free Hugging Face Static Space.

Run after experimenter.py:  uv run export_static.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import experimenter as ex
import prepare

OUT = Path(__file__).parent / "static"


def main():
    items, orders = prepare.load()
    _, test = ex.targets(orders)
    ct = ex.Context(items, orders, test)
    best = json.loads(ex.BEST.read_text())
    tops = {name: np.argsort(-ex.score(cfg, ct), axis=1)[:, :10] for name, cfg in [("popularity", ex.make("pop")), ("winner", best)]}
    cut = test.set_index("user")["hour"]

    def venue(i, r=None):
        it = items.iloc[i]
        v = {"id": int(i), "cat": it["category"], "lat": round(float(it["lat"]), 5), "lng": round(float(it["lng"]), 5)}
        if r is not None:
            v["near"] = round(float(ct.near[r, np.searchsorted(ct.Q, i)]), 2)
        return v

    users = []
    for r, u in enumerate(ct.U):
        hist = orders[(orders["user"] == u) & (orders["hour"] < cut[u])]
        seen = hist.drop_duplicates("item")
        truth = int(ct.Q[ct.truth[r]])
        users.append({
            "user": int(u),
            "counts": hist["vertical"].value_counts().to_dict(),
            "seen": [dict(venue(i), v=v) for i, v in zip(seen["item"], seen["vertical"])],
            "truth": dict(venue(truth, r), repeat=bool(truth in set(hist["item"]))),
            **{name: [venue(int(ct.Q[q]), r) for q in t[r]] for name, t in tops.items()},
        })

    log = pd.read_csv(ex.RESULTS, sep="\t")
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({
        "log": log.to_dict("records"),
        "report": ex.REPORT.read_text(),
        "users": users,
    }, separators=(",", ":")))
    print(f"wrote {OUT / 'data.json'} ({(OUT / 'data.json').stat().st_size / 1e6:.1f} MB, {len(users)} users)")


if __name__ == "__main__":
    main()
