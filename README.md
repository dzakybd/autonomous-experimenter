# Autonomous Experimenter: cross-vertical personalization

An AI that runs experiments, compares models, and summarizes the results without a human in the loop.

**Live demo (saved run, interactive map):** https://huggingface.co/spaces/dzakybd/autonomous-experimenter · **Code:** https://github.com/dzakybd/autonomous-experimenter

The problem: recommend a user's next **Mart** (quick-commerce) purchase using what they do in **Food, Mart, and Shops**, with location, including users who have barely used Mart.

## Summary (100 words)

An autonomous experimenter for cross-vertical personalization, built on the autoresearch loop (modify, verify, keep or discard). On public Foursquare NYC check-ins mapped to Food, Mart convenience retail, and Shops, it tunes a next-Mart-visit recommender. Each iteration changes one thing: a base model (e.g. graph-based RP3beta) or one signal (cross-vertical history, proximity, repeat visits, attention over recent check-ins). A paired bootstrap over users keeps a change only if the gain isn't noise. Kept changes become git commits; results.tsv is the proposer's memory. On held-out visits: NDCG@10 0.074 → 0.558 over popularity; cold-start users 0.143 → 0.330.

## Run it

```bash
uv run experimenter.py                     # heuristic proposer, no API key, a few seconds
uv run --extra llm experimenter.py --llm   # Claude proposes each change and writes the conclusion
uv run --extra app app.py                  # Gradio app: run the loop live in your browser
uv run export_static.py                    # rebuild static/ (the Hugging Face dashboard) from the latest run
```

Outputs: `results.tsv` (every experiment), `best_config.json`, `report.md`, and one git commit per kept change (`git log --oneline`). The data downloads on first run.

## The data

[Foursquare NYC check-ins](https://huggingface.co/datasets/yyh11/FourSquare_checkin) (Yang et al., IEEE TSMC 2014): 227K check-ins by 1,083 users, Apr 2012 – Feb 2013, with venue category and coordinates. `prepare.py` maps categories to verticals and keeps venues with 5+ check-ins:

| Vertical | Venue categories | Venues | Check-ins |
|---|---|---|---|
| Mart | Deli / Bodega, Food & Drink Shop, Drugstore / Pharmacy, Convenience Store, Market | 705 | 10,642 |
| Food | Restaurants, cafés, bakeries, dessert | 2,764 | 32,511 |
| Shops | Other retail and personal services | 655 | 9,768 |

Check-ins are in-person visits, not deliveries. They are a public proxy for where people buy, so treat results as evidence about which *kinds* of signal matter, not as an estimate for any real app.

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
| Task and split | Users with 2+ Mart visits (721 after the split). Second-to-last Mart visit is validation, last is test. Only history before each visit is visible |
| Metric | NDCG@10 over the 705 Mart venues; also NDCG for cold users (≤1 earlier Mart visit), Recall@10, coverage |
| Base models | Popularity, ItemKNN, EASE, RP3beta (a random walk on the user–venue graph, i.e. graph-based retrieval) |
| Signals | `cross`: weight of Food + Shops history in the user vector · `geo`: proximity to the user's usual area · `repeat`: revisit boost · `seq`: dot-product attention over the 5 most recent check-ins, any vertical |
| Proposer | Heuristic: try each base model, then change one knob of the current best. `--llm`: Claude reads the log and proposes the next change with a hypothesis |
| Verify | A fixed evaluation harness plus a paired bootstrap over users, so small noisy gains are rejected |
| Memory | Kept configs are git commits; every run, including discards, is in `results.tsv` |
| Honest final score | Baseline and winner scored **once** on the test visits, which the loop never saw |

## Result (heuristic proposer, seed 0, 40 iterations)

| | NDCG@10 | NDCG@10, cold users | Recall@10 | Mart coverage |
|---|---|---|---|---|
| Popularity baseline | 0.0740 | 0.1429 | 0.1512 | 1.4% |
| RP3beta (α 0.9, β 0.5) + cross 0.2, geo 0.6, repeat 2.0 | 0.5579 | 0.3299 | 0.7087 | 100% |

Delta 95% CI +0.449 to +0.520 over 721 users; 118 cold users. 9 changes kept, 30 discarded. Full trail in [`report.md`](report.md).

What the loop found:
- The graph model (RP3beta) is the biggest jump, but on its own it barely helps cold users (validation cold NDCG 0.123 → 0.127).
- Proximity is what fixes cold users (0.127 → 0.216). Where you are tells you which bodega you'll use.
- Repeat visits matter: 74% of Mart visits are to a venue the user has been to before.
- Cross-vertical history helps cold users further (0.226 → 0.241). Food and Shops behaviour says which Mart venues a new user will pick.
- Attention over recent check-ins was tried five times and **rejected every time**. On this data, recency adds noise once location and repeat are in.

## Limitations and next steps

- Proxy data: visits, not deliveries, and one city in 2012. On real order data the next step is a user-randomised A/B test on Mart: orders per user as the primary metric, with coverage, latency, and Food cannibalisation as guardrails.
- The attention signal uses fixed co-occurrence embeddings. A trained sequential transformer (SASRec/BERT4Rec style) is the natural replacement at scale, along with two-tower retrieval for big catalogues and online updates within a session.
- Re-using one validation set over 40 iterations overfits it a little. The single test run measures how much.

Inspired by [autoresearch](https://github.com/uditgoenka/autoresearch) (Udit Goenka) and Karpathy's original autoresearch idea.
