# Autonomous Experimenter report

Next Quik order for each user with 2+ Quik orders, using their Food, Quik, and Shops history (simulated super-app data, see `simulate.py`). Loop decisions used each user's second-to-last Quik order; the numbers below are one final run on the last order, which the loop never saw.

## Result

| | config | NDCG@10 | NDCG@10, cold users | Recall@10 | Quik catalogue coverage |
|---|---|---|---|---|---|
| baseline | `{"blend": {"cross": 0.0, "geo": 0.0, "repeat": 0.0, "seq": 0.0}, "model": "pop", "params": {}}` | 0.1608 | 0.0959 | 0.3042 | 2.0% |
| winner | `{"blend": {"cross": 0.2, "geo": 0.4, "repeat": 1.0, "seq": 0.2}, "model": "rp3beta", "params": {"alpha": 0.84, "beta": 0.18}}` | 0.3895 | 0.2046 | 0.5077 | 98.2% |

Lift: +142.2% NDCG@10 overall (+113.4% for the 439 cold users with at most one earlier Quik order). Delta 95% CI +0.2146..+0.2425; P(delta>0) = 1.000 (paired bootstrap over 2584 users).
Experiments: 40 run, 13 kept, 26 discarded.

## Conclusion

The loop replaced the popularity baseline with **rp3beta** plus the cross, geo, repeat, seq signal(s), lifting held-out NDCG@10 by +142.2% (0.1608 -> 0.3895); the 95% CI of the delta excludes zero. Cold users changed by +113.4%. This is simulated data, so the result shows the loop finds planted signals, not what Careem's users do. Next step on real data: a user-randomised A/B test on Quik with orders per user as the primary metric, and coverage, latency, and Food cannibalisation as guardrails.

## Path of kept changes

- iter 0: popularity -> val NDCG@10 0.1601 (cold 0.1206)
- iter 3: try base model rp3beta at defaults -> val NDCG@10 0.1869 (cold 0.0643)
- iter 5: geo 0.0 -> 0.2 -> val NDCG@10 0.2416 (cold 0.1817)
- iter 6: beta 0.5 -> 0.38 -> val NDCG@10 0.2553 (cold 0.1825)
- iter 7: alpha 1.0 -> 0.84 -> val NDCG@10 0.2782 (cold 0.1828)
- iter 9: geo 0.2 -> 0.6 -> val NDCG@10 0.3062 (cold 0.1830)
- iter 10: geo 0.6 -> 0.8 -> val NDCG@10 0.3110 (cold 0.1830)
- iter 12: repeat 0.0 -> 0.5 -> val NDCG@10 0.3310 (cold 0.1830)
- iter 15: cross 0.0 -> 0.2 -> val NDCG@10 0.3374 (cold 0.1958)
- iter 18: repeat 0.5 -> 1.0 -> val NDCG@10 0.3393 (cold 0.1958)
- iter 19: seq 0.0 -> 0.2 -> val NDCG@10 0.3418 (cold 0.2001)
- iter 20: beta 0.38 -> 0.26 -> val NDCG@10 0.3446 (cold 0.1972)
- iter 22: geo 0.8 -> 0.4 -> val NDCG@10 0.3468 (cold 0.1969)
- iter 23: beta 0.26 -> 0.18 -> val NDCG@10 0.3488 (cold 0.1990)

## Best validation NDCG@10 per base model

| model | NDCG@10 |
|---|---|
| rp3beta | 0.3492 |
| pop | 0.1601 |
| ease | 0.1576 |
| itemknn | 0.1134 |

Full log: `results.tsv`. Kept changes are git commits (`git log --oneline`).
