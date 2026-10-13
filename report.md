# Autonomous Experimenter report

Next Mart venue (bodega, deli, pharmacy, convenience store) for each user with 2+ such visits, using their Food, Mart, and Shops check-ins (Foursquare NYC, public). Loop decisions used each user's second-to-last Mart visit; the numbers below are one final run on the last visit, which the loop never saw.

## Result

| | config | NDCG@10 | NDCG@10, cold users | Recall@10 | Mart coverage |
|---|---|---|---|---|---|
| baseline | `{"blend": {"cross": 0.0, "geo": 0.0, "repeat": 0.0, "seq": 0.0}, "model": "pop", "params": {}}` | 0.0740 | 0.1429 | 0.1512 | 1.4% |
| winner | `{"blend": {"cross": 0.2, "geo": 0.6, "repeat": 2.0, "seq": 0.0}, "model": "rp3beta", "params": {"alpha": 0.9, "beta": 0.5}}` | 0.5579 | 0.3299 | 0.7087 | 100.0% |

Lift: +654.2% NDCG@10 overall (+130.8% for the 118 cold users with at most one earlier Mart visit). Delta 95% CI +0.4490..+0.5204; P(delta>0) = 1.000 (paired bootstrap over 721 users).
Experiments: 40 run, 9 kept, 30 discarded.

## Conclusion

The loop replaced the popularity baseline with **rp3beta** plus the cross, geo, repeat signal(s), lifting held-out NDCG@10 by +654.2% (0.0740 -> 0.5579); the 95% CI of the delta excludes zero. Cold users changed by +130.8%. Check-ins are visits, not deliveries, so treat this as evidence about which signal families matter, not a Careem estimate. Next step on real data: a user-randomised A/B test on Mart with orders per user as the primary metric, and coverage, latency, and Food cannibalisation as guardrails.

## Path of kept changes

- iter 0: popularity -> val NDCG@10 0.0734 (cold 0.1232)
- iter 1: try base model itemknn at defaults -> val NDCG@10 0.2439 (cold 0.0236)
- iter 3: try base model rp3beta at defaults -> val NDCG@10 0.4541 (cold 0.1273)
- iter 5: geo 0.0 -> 0.2 -> val NDCG@10 0.4898 (cold 0.2161)
- iter 9: geo 0.2 -> 0.6 -> val NDCG@10 0.4984 (cold 0.2255)
- iter 12: repeat 0.0 -> 0.5 -> val NDCG@10 0.5219 (cold 0.2255)
- iter 16: cross 0.0 -> 0.2 -> val NDCG@10 0.5313 (cold 0.2405)
- iter 19: repeat 0.5 -> 1.0 -> val NDCG@10 0.5350 (cold 0.2405)
- iter 23: alpha 1.0 -> 0.9 -> val NDCG@10 0.5370 (cold 0.2404)
- iter 31: repeat 1.0 -> 2.0 -> val NDCG@10 0.5404 (cold 0.2404)

## Best validation NDCG@10 per base model

| model | NDCG@10 |
|---|---|
| rp3beta | 0.5406 |
| itemknn | 0.2439 |
| ease | 0.2351 |
| pop | 0.0734 |

Full log: `results.tsv`. Kept changes are git commits (`git log --oneline`).
