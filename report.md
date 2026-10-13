# Autonomous Experimenter report

Next Mart venue (bodega, deli, pharmacy, convenience store) for each user with 2+ such visits, using their Food, Mart, and Shops check-ins (Foursquare NYC, public). Loop decisions used each user's second-to-last Mart visit; the numbers below are one final run on the last visit, which the loop never saw.

## Result

| | config | NDCG@10 | NDCG@10, cold users | Recall@10 | Mart coverage |
|---|---|---|---|---|---|
| baseline | `{"blend": {"cross": 0.0, "geo": 0.0, "here": 0.0, "repeat": 0.0, "seq": 0.0}, "model": "pop", "params": {}}` | 0.0740 | 0.1429 | 0.1512 | 1.4% |
| winner | `{"blend": {"cross": 0.2, "geo": 0.6, "here": 0.0, "repeat": 2.0, "seq": 0.0}, "model": "rp3beta", "params": {"alpha": 0.8, "beta": 0.5}}` | 0.5587 | 0.3333 | 0.7115 | 100.0% |

Lift: +655.3% NDCG@10 overall (+133.2% for the 118 cold users with at most one earlier Mart visit). Delta 95% CI +0.4500..+0.5210; P(delta>0) = 1.000 (paired bootstrap over 721 users).
Experiments: 32 run, 2 kept, 29 discarded.

## Conclusion

The loop replaced the popularity baseline with **rp3beta** plus the cross, geo, repeat signal(s), lifting held-out NDCG@10 by +655.3% (0.0740 -> 0.5587); the 95% CI of the delta excludes zero. Cold users changed by +133.2%. Check-ins are visits, not deliveries, so treat this as evidence about which signal families matter, not an estimate for any real app. Next step on real data: a user-randomised A/B test on Mart with orders per user as the primary metric, and coverage, latency, and Food cannibalisation as guardrails.

## Path of kept changes

- iter 0: popularity -> val NDCG@10 0.0734 (cold 0.1232)
- iter 1: resume from a previous best -> val NDCG@10 0.5404 (cold 0.2404)
- iter 12: alpha 0.9 -> 0.8 -> val NDCG@10 0.5423 (cold 0.2391)

## Best validation NDCG@10 per base model

| model | NDCG@10 |
|---|---|
| rp3beta | 0.5445 |
| itemknn | 0.2439 |
| ease | 0.2351 |
| pop | 0.0734 |

Full log: `results.tsv`. Kept changes are git commits (`git log --oneline`).
