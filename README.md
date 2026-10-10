# Autonomous Experimenter: super-app personalization

Careem AI Challenge, option 2: an AI that runs experiments, compares models, and summarizes the results without a human in the loop.

The problem it works on is the one in the Personalization role: recommend a user's **next Quik (quick-commerce) order** using what they do in **Food, Quik, and Shops**, across Dubai zones, including users who have barely used Quik.

## Summary (100 words)

An autonomous experimenter for cross-vertical personalization, built on the autoresearch loop (modify, verify, keep or discard). On a simulated Dubai super-app log (Food, Quik, Shops; 75K orders), it tunes a next-Quik-order recommender: each iteration changes one thing (base model such as graph-based RP3beta, or one signal: cross-vertical history, zone popularity, repeat buying, attention over recent orders). A paired bootstrap over users keeps a change only if the gain isn't noise. Kept changes become git commits; results.tsv is the proposer's memory. On held-out orders: +142% NDCG@10 over popularity, +113% for cold-start users, 95% CI excluding zero.

## Run it

```bash
uv run experimenter.py                     # heuristic proposer, no API key, a few seconds
uv run --extra llm experimenter.py --llm   # Claude proposes each change and writes the conclusion
uv run simulate.py                         # regenerate the data (seeded)
```

Outputs: `results.tsv` (every experiment), `best_config.json`, `report.md`, and one git commit per kept change (`git log --oneline`).

## The data (dummy, by design)

No public dataset combines food delivery, quick commerce, and retail with location (the Delivery Hero dataset's download is access-restricted). So `simulate.py` generates one with a known story. That is useful here: we can check that the experimenter finds the signals that really drive behaviour.

| File | Contents |
|---|---|
| `data/users.csv` | 4,000 users, home zone (12 Dubai zones with lat/lng) |
| `data/items.csv` | 400 Food, 500 Quik, 300 Shops items; 30% of Quik items are local to one zone |
| `data/orders.csv` | 75,583 orders over 90 days (48K Food, 19.5K Quik, 8K Shops) |

Planted signals: one taste vector per user shared across verticals; neighbourhood taste plus local assortment; repeat buying; and short-term intent that carries from one order to the next, in any vertical. Quik use is skewed, so many users are cold in Quik but active in Food.

## How the loop works

```
propose one change ─► score on validation ─► paired bootstrap vs current best
        ▲                                             │
        │                          P(gain > 0) ≥ 0.90 │ otherwise
        │                                keep ◄───────┴──────► discard
        └──────── results.tsv + git log (memory) ◄──────────────┘
```

| Part | What it does |
|---|---|
| Task and split | Users with 2+ Quik orders (2,584). Second-to-last Quik order is validation, last is test. Only history before each order is visible |
| Metric | NDCG@10 over the Quik catalogue; also NDCG for cold users (≤1 earlier Quik order), Recall@10, coverage |
| Base models | Popularity, ItemKNN, EASE, RP3beta (a random walk on the user-item graph, i.e. graph-based retrieval) |
| Signals | `cross`: weight of Food + Shops history in the user vector · `geo`: zone-level Quik popularity · `repeat`: re-order boost · `seq`: dot-product attention over the 5 most recent orders, any vertical |
| Proposer | Heuristic: try each base model, then change one knob of the current best. `--llm`: Claude reads the log and proposes the next change with a hypothesis |
| Verify | A fixed evaluation harness plus a paired bootstrap over users, so small noisy gains are rejected |
| Memory | Kept configs are git commits; every run, including discards, is in `results.tsv` |
| Honest final score | Baseline and winner scored **once** on the test orders, which the loop never saw |

## Result (heuristic proposer, seed 0, 40 iterations)

| | NDCG@10 | NDCG@10, cold users | Recall@10 | Quik coverage |
|---|---|---|---|---|
| Popularity baseline | 0.1608 | 0.0959 | 0.3042 | 2.0% |
| RP3beta + cross 0.2, geo 0.4, repeat 1.0, seq 0.2 | 0.3895 | 0.2046 | 0.5077 | 98.2% |

+142% NDCG@10 (95% CI of the delta +0.215 to +0.243), +113% for the 439 cold users. 13 changes kept, 26 discarded. Full trail in [`report.md`](report.md).

What the loop found, in order: the graph model beats popularity overall but hurts cold users (validation cold NDCG 0.121 → 0.064). Zone popularity fixes that (0.064 → 0.182) and is the biggest single gain. Repeat boosting helps returning users. Cross-vertical history is the change that moves cold users most (0.183 → 0.196): Food behaviour tells you what a new Quik user will buy. Attention over recent orders adds a little on top.

## Limitations and next steps

- The data is simulated, so this shows the loop recovers planted signals. It says nothing about Careem's users. On real data the next step is a user-randomised A/B test on Quik: orders per user as the primary metric, with coverage, latency, and Food cannibalisation as guardrails.
- The attention signal uses fixed co-occurrence embeddings. A trained sequential transformer (SASRec/BERT4Rec style) is the natural replacement at scale, as are two-tower retrieval for big catalogues and online updates within a session.
- Re-using one validation set over 40 iterations overfits it a little. The single test run measures how much.

Inspired by [autoresearch](https://github.com/uditgoenka/autoresearch) (Udit Goenka) and Karpathy's original autoresearch idea.
