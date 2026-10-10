"""Synthetic super-app order log: Food, Quik (quick commerce), and Shops across 12 Dubai zones.

Dummy data with a known generative story, so we can check whether the experimenter finds the
signals that really drive behaviour:

- one taste vector per user, shared across verticals  -> cross-vertical transfer should help
- users in the same zone have similar tastes, and some Quik items are local specialties
                                                     -> zone-level popularity should help
- people re-order what they bought before            -> repeat boosting should help
- the next order leans towards the previous one, in any vertical (sushi dinner -> Asian groceries)
                                                     -> attention over recent orders should help

Writes data/users.csv, data/items.csv, data/orders.csv. Run: uv run simulate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ZONES = [
    ("Dubai Marina", 25.080, 55.140), ("JLT", 25.069, 55.145), ("Downtown", 25.197, 55.274),
    ("Business Bay", 25.186, 55.262), ("Deira", 25.271, 55.308), ("Bur Dubai", 25.254, 55.297),
    ("Al Barsha", 25.112, 55.198), ("Jumeirah", 25.208, 55.247), ("Mirdif", 25.218, 55.421),
    ("Silicon Oasis", 25.121, 55.377), ("Al Quoz", 25.140, 55.230), ("Palm Jumeirah", 25.112, 55.139),
]
N_USERS, D, DAYS = 4000, 6, 90
N_ITEMS = {"food": 400, "quik": 500, "shops": 300}
HOURS = {"food": ([13, 20], 1.5), "quik": ([10, 19], 3.0), "shops": ([15], 4.0)}  # peaks, spread
# Mean orders per user over 90 days (gamma shape, scale). Quik is skewed: many users barely use it.
RATES = {"food": (2.0, 6.0), "quik": (0.8, 6.0), "shops": (1.0, 2.0)}
W_TASTE, W_POP, W_REPEAT, W_INTENT, W_LOCAL = 1.2, 0.8, 1.5, 1.0, 1.0


def main(seed=42):
    rng = np.random.default_rng(seed)
    nz = len(ZONES)

    verticals = np.concatenate([[v] * n for v, n in N_ITEMS.items()])
    vec = rng.normal(0, 1, (len(verticals), D))
    bias = rng.normal(0, 1, len(verticals))
    local = np.where((verticals == "quik") & (rng.random(len(verticals)) < 0.3), rng.integers(0, nz, len(verticals)), -1)

    zone_taste = rng.normal(0, 0.8, (nz, D))
    zone = rng.choice(nz, N_USERS, p=rng.dirichlet(np.full(nz, 5.0)))
    taste = zone_taste[zone] + rng.normal(0, 1, (N_USERS, D))

    rows = []
    for u in range(N_USERS):
        events = []
        for v, (shape, scale) in RATES.items():
            n = rng.poisson(rng.gamma(shape, scale))
            peaks, spread = HOURS[v]
            hour = (rng.choice(peaks, n) + rng.normal(0, spread, n)) % 24
            events += [(d * 24 + h, v) for d, h in zip(rng.integers(0, DAYS, n), hour)]
        bought = np.zeros(len(verticals))
        prev, prev_t = None, -1e9
        for t, v in sorted(events):
            cand = np.flatnonzero(verticals == v)
            logit = W_TASTE * vec[cand] @ taste[u] / np.sqrt(D) + W_POP * bias[cand] + W_REPEAT * np.log1p(bought[cand])
            if prev is not None and t - prev_t < 72:  # short-term intent carries over for ~3 days
                logit += W_INTENT * vec[cand] @ vec[prev] / np.sqrt(D)
            logit += W_LOCAL * (local[cand] == zone[u])
            p = np.exp(logit - logit.max())
            item = rng.choice(cand, p=p / p.sum())
            bought[item] += 1
            prev, prev_t = item, t
            rows.append((u, item, v, round(t, 2)))

    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"user": range(N_USERS), "zone": [ZONES[z][0] for z in zone],
                  "lat": [ZONES[z][1] for z in zone], "lng": [ZONES[z][2] for z in zone]}).to_csv(out / "users.csv", index=False)
    pd.DataFrame({"item": range(len(verticals)), "vertical": verticals,
                  "local_zone": [ZONES[z][0] if z >= 0 else "" for z in local]}).to_csv(out / "items.csv", index=False)
    orders = pd.DataFrame(rows, columns=["user", "item", "vertical", "hour"])
    orders.to_csv(out / "orders.csv", index=False)
    print(orders["vertical"].value_counts().to_string(), f"\n{len(orders)} orders, {N_USERS} users")


if __name__ == "__main__":
    main()
