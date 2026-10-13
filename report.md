# Autonomous Experimenter report

Next Mart venue (bodega, deli, pharmacy, convenience store) for each user with 2+ such visits, using their Food, Mart, and Shops check-ins (Foursquare NYC, public). Loop decisions used each user's second-to-last Mart visit; the numbers below are one final run on the last visit, which the loop never saw.

## Result

| | config | NDCG@10 | NDCG@10, cold users | Recall@10 | Mart coverage |
|---|---|---|---|---|---|
| baseline | `{"blend": {"cross": 0.0, "geo": 0.0, "here": 0.0, "repeat": 0.0, "seq": 0.0}, "model": "pop", "params": {}}` | 0.0740 | 0.1429 | 0.1512 | 1.4% |
| winner | `{"blend": {"cross": 0.0, "geo": 0.2, "here": 0.2, "repeat": 1.0, "seq": 0.0}, "model": "rp3beta", "params": {"alpha": 1.0, "beta": 0.5}}` | 0.5629 | 0.3349 | 0.7129 | 100.0% |

Lift: +660.9% NDCG@10 overall (+134.3% for the 118 cold users with at most one earlier Mart visit). Delta 95% CI +0.4536..+0.5260; P(delta>0) = 1.000 (paired bootstrap over 721 users).
Experiments: 29 run, 6 kept, 22 discarded.

## Conclusion

The loop replaced the popularity baseline with **rp3beta** plus the geo, here, repeat signal(s), lifting held-out NDCG@10 by +660.9% (0.0740 -> 0.5629); the 95% CI of the delta excludes zero. Cold users changed by +134.3%. Check-ins are visits, not deliveries, so treat this as evidence about which signal families matter, not an estimate for any real app. Next step on real data: a user-randomised A/B test on Mart with orders per user as the primary metric, and coverage, latency, and Food cannibalisation as guardrails.

## Path of kept changes

- iter 0: popularity -> val NDCG@10 0.0734 (cold 0.1232)
- iter 1: try base model itemknn at defaults -> val NDCG@10 0.2439 (cold 0.0236)
- iter 3: try base model rp3beta at defaults -> val NDCG@10 0.4541 (cold 0.1273)
- iter 4: repeat 0.0 -> 0.5 -> val NDCG@10 0.4897 (cold 0.1273)
- iter 5: here 0.0 -> 0.2 -> val NDCG@10 0.5255 (cold 0.2298)
- iter 6: geo 0.0 -> 0.2 -> val NDCG@10 0.5332 (cold 0.2460)
- iter 9: repeat 0.5 -> 1.0 -> val NDCG@10 0.5409 (cold 0.2460)

## Best validation NDCG@10 per base model

| model | NDCG@10 |
|---|---|
| rp3beta | 0.5421 |
| itemknn | 0.2439 |
| ease | 0.2351 |
| pop | 0.0734 |

Full log: `results.tsv`. Kept changes are git commits (`git log --oneline`).
