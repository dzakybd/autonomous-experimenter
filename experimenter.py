"""Autonomous Experimenter: an autoresearch-style loop for cross-vertical personalization.

Task: recommend each user's next Mart venue (bodega, deli, pharmacy, convenience store),
using their Food, Mart, and Shops history from the public Foursquare NYC check-in data
(see prepare.py). The question is which signals transfer: other verticals, location,
repeat visits, or the intent behind the most recent check-ins.

Loop (after github.com/uditgoenka/autoresearch): propose ONE change -> run it ->
verify mechanically (validation NDCG@10 + paired bootstrap) -> keep (git commit)
or discard -> log to results.tsv -> repeat. The log is the agent's memory: the
proposer reads it to avoid repeats and to climb from the current best.

Each user's second-to-last Mart visit is validation; the last is test. The loop only
sees validation. At the end the baseline and the winner are scored once on test.

Usage:
    uv run experimenter.py --iters 40          # heuristic proposer, no API key
    uv run --extra llm experimenter.py --llm   # Claude proposes and narrates
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import prepare

ROOT = Path(__file__).parent
RESULTS = ROOT / "results.tsv"
BEST = ROOT / "best_config.json"
REPORT = ROOT / "report.md"
K = 10
KEEP_PROB = 0.90  # keep only if P(delta NDCG > 0) >= 0.90 under paired bootstrap
GEO_KM = 1.5      # distance scale of the proximity signal

# Base retrieval models: param -> (low, high, log_scale, is_int)
SPACE = {
    "pop": {},
    "itemknn": {"k": (10, 400, True, True), "shrink": (0, 200, False, False)},
    "ease": {"lam": (10, 5000, True, False)},
    "rp3beta": {"alpha": (0.2, 1.5, False, False), "beta": (0.0, 1.0, False, False)},  # graph random walk
}
DEFAULTS = {"pop": {}, "itemknn": {"k": 100, "shrink": 10.0}, "ease": {"lam": 250.0}, "rp3beta": {"alpha": 1.0, "beta": 0.5}}
# Signals blended on top of the base model: name -> (high, step). All start at 0 (off).
BLEND = {
    "cross": (1.0, 0.2),   # weight of Food + Shops history in the user vector (cross-vertical transfer)
    "geo": (1.0, 0.2),     # proximity of the Mart venue to the user's usual area
    "here": (1.0, 0.2),    # proximity to where the user is now (their latest check-in, any vertical)
    "repeat": (2.0, 0.5),  # boost Mart venues the user visited before
    "seq": (1.0, 0.2),     # attention over the user's 5 most recent check-ins, any vertical
}


# ---------------------------------------------------------------- data
def targets(orders):
    """Per user with 2+ Mart visits: second-to-last Mart visit -> val, last -> test."""
    q = orders[orders["vertical"] == "mart"].copy()
    q["n_before"] = q.groupby("user").cumcount()
    q["pos"] = q.groupby("user").cumcount(ascending=False)
    q = q[q["user"].isin(q.loc[q["pos"] == 1, "user"])]
    return q[q["pos"] == 1], q[q["pos"] == 0]


class Context:
    """Everything known before each target user's cutoff (other users: their full history)."""

    def __init__(self, items, orders, target):
        cut = orders["user"].map(target.set_index("user")["hour"]).fillna(np.inf)
        hist = orders[orders["hour"] < cut]
        n_users, n_items = orders["user"].max() + 1, len(items)
        self.items = items
        self.X = sp.csr_matrix((np.ones(len(hist)), (hist["user"], hist["item"])), shape=(n_users, n_items))
        self.Xb = (self.X > 0).astype(float)
        self.is_mart = items["vertical"].to_numpy() == "mart"
        self.Q = np.flatnonzero(self.is_mart)
        self.U = target["user"].to_numpy()
        self.truth = np.searchsorted(self.Q, target["item"].to_numpy())
        self.cold = target["n_before"].to_numpy() <= 1  # at most one earlier Mart visit
        by_user = hist.groupby("user")["item"]
        recent = by_user.apply(lambda s: s.to_numpy()[-5:])
        self.recent = [recent.get(u, np.array([], int)) for u in self.U]
        lat, lng = items["lat"].to_numpy(), items["lng"].to_numpy()
        anchor = by_user.apply(lambda s: (np.median(lat[s]), np.median(lng[s])))  # the user's usual area
        a = np.array([anchor.get(u, (np.nan, np.nan)) for u in self.U])
        dy = (lat[self.Q][None] - a[:, :1]) * 111.0
        dx = (lng[self.Q][None] - a[:, 1:]) * 111.0 * np.cos(np.radians(40.7))
        self.near = np.nan_to_num(np.exp(-np.hypot(dx, dy) / GEO_KM))
        last = by_user.apply(lambda s: s.to_numpy()[-1])  # latest check-in before the cutoff
        h = np.array([last.get(u, -1) for u in self.U])
        dy = (lat[self.Q][None] - lat[h][:, None]) * 111.0
        dx = (lng[self.Q][None] - lng[h][:, None]) * 111.0 * np.cos(np.radians(40.7))
        self.here = np.where(h[:, None] >= 0, np.exp(-np.hypot(dx, dy) / GEO_KM), 0.0)
        self._cos = {}

    def cosine(self, cross):
        if cross not in self._cos:
            self._cos = {cross: cosine(self.weighted(cross))}  # keep one cached matrix
        return self._cos[cross]

    def weighted(self, cross):
        return self.Xb @ sp.diags(np.where(self.is_mart, 1.0, cross))


# ---------------------------------------------------------------- models
def rownorm(S):
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return np.where(hi > lo, (S - lo) / np.where(hi > lo, hi - lo, 1), 0.0)


def cosine(Xw, shrink=0.0):
    G = (Xw.T @ Xw).toarray()
    n = np.sqrt(np.diag(G))
    sim = G / (np.outer(n, n) + shrink + 1e-9)
    np.fill_diagonal(sim, 0)
    return sim


def score(cfg, c: Context):
    m, p, b = cfg["model"], cfg["params"], cfg["blend"]
    Xw = c.weighted(b["cross"])
    if m == "pop":
        S = np.tile(np.asarray(c.Xb[:, c.Q].sum(0)), (len(c.U), 1))
    elif m == "itemknn":
        sim = cosine(Xw, p["shrink"])[:, c.Q]
        sim[sim < np.partition(sim, -p["k"], axis=0)[-p["k"]]] = 0  # top-k neighbours per Mart venue
        S = Xw[c.U] @ sim
    elif m == "ease":  # Steck 2019, closed form
        P = np.linalg.inv((Xw.T @ Xw).toarray() + p["lam"] * np.eye(Xw.shape[1]))
        B = -P[:, c.Q] / np.diag(P)[c.Q]
        B[c.Q, np.arange(len(c.Q))] = 0
        S = Xw[c.U] @ B
    elif m == "rp3beta":  # 3-step walk user -> venue -> user -> venue, popularity-penalised
        def walk(M):
            d = np.asarray(M.sum(1)).ravel()
            return (sp.diags(1 / np.where(d > 0, d, 1)) @ M).power(p["alpha"])
        Pui, Piu = walk(Xw), walk(Xw.T.tocsr())
        W = (Piu @ Pui[:, c.Q]).toarray() / (np.asarray(c.Xb[:, c.Q].sum(0)) + 1) ** p["beta"]
        S = Pui[c.U] @ W
    else:
        raise ValueError(m)
    S = rownorm(np.asarray(S, dtype=float))

    if b["repeat"]:
        S += b["repeat"] * rownorm(np.log1p(c.X[c.U][:, c.Q].toarray()))
    if b["geo"]:
        S += b["geo"] * c.near
    if b["here"]:
        S += b["here"] * c.here
    if b["seq"]:
        sim = c.cosine(b["cross"])
        A = np.zeros_like(S)
        for r, h in enumerate(c.recent):
            if len(h):
                a = np.exp(sim[h[-1], h] / 0.1)  # query = latest check-in, keys = recent check-ins
                A[r] = (a / a.sum()) @ sim[np.ix_(h, c.Q)]
        S += b["seq"] * rownorm(A)
    return S + np.random.default_rng(0).uniform(0, 1e-9, S.shape)  # random tie-break


# ---------------------------------------------------------------- evaluation
def evaluate(cfg, c: Context):
    S = score(cfg, c)
    rank = (S > S[np.arange(len(c.U)), c.truth][:, None]).sum(1)
    ndcg = np.where(rank < K, 1 / np.log2(rank + 2), 0.0)
    top = np.argpartition(-S, K, axis=1)[:, :K]
    return {
        "ndcg": float(ndcg.mean()),
        "ndcg_cold": float(ndcg[c.cold].mean()),
        "recall": float((rank < K).mean()),
        "coverage": len(np.unique(top)) / len(c.Q),
        "per_user": ndcg,
    }


def p_improve(new, old, n=2000, seed=0):
    """Paired bootstrap over users: P(mean NDCG delta > 0) and a 95% CI of the delta."""
    d = new - old
    means = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float((means > 0).mean()), np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------- proposers
def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def make(model, params=None, blend=None):
    cfg = {"model": model, "params": dict(DEFAULTS[model], **(params or {})), "blend": dict.fromkeys(BLEND, 0.0)}
    cfg["blend"].update(blend or {})
    for name, (lo, hi, _, is_int) in SPACE[model].items():
        v = min(max(cfg["params"][name], lo), hi)
        cfg["params"][name] = int(round(v)) if is_int else round(float(v), 2)
    cfg["params"] = {n: cfg["params"][n] for n in SPACE[model]}
    cfg["blend"] = {n: round(min(max(float(cfg["blend"][n]), 0.0), hi), 2) for n, (hi, _) in BLEND.items()}
    return cfg


def propose_heuristic(history, best, rng):
    """Try each base model once, then change one knob of the current best."""
    tried = {h["key"] for h in history}
    for m in SPACE:
        cfg = make(m)
        if key(cfg) not in tried:
            return cfg, f"try base model {m} at defaults"
    for _ in range(300):
        params, blend = dict(best["params"]), dict(best["blend"])
        knobs = list(SPACE[best["model"]]) + list(BLEND)
        knob = knobs[rng.integers(len(knobs))]
        if knob in BLEND:
            old, step = blend[knob], BLEND[knob][1]
            blend[knob] = old + rng.choice([-2, -1, 1, 2]) * step if old else rng.choice([1, 2, 3]) * step
        else:
            lo, hi, log, _ = SPACE[best["model"]][knob]
            old, f = params[knob], rng.choice([0.5, 0.7, 1.4, 2.0])
            params[knob] = old * f if log else old + (f - 1) * (hi - lo) / 4
        cfg = make(best["model"], params, blend)
        new = cfg["blend"][knob] if knob in BLEND else cfg["params"][knob]
        if key(cfg) not in tried:
            return cfg, f"{knob} {old} -> {new}"
    return None, "every one-step change around the best has been tried (local optimum)"


def propose_llm(history, best):
    import anthropic
    from typing import Literal, Optional
    from pydantic import BaseModel

    class Proposal(BaseModel):
        model: Literal["pop", "itemknn", "ease", "rp3beta"]
        k: Optional[int] = None
        shrink: Optional[float] = None
        lam: Optional[float] = None
        alpha: Optional[float] = None
        beta: Optional[float] = None
        cross: float
        geo: float
        repeat: float
        seq: float
        hypothesis: str

    log = "\n".join(f"{h['iter']}\t{h['status']}\t{h['ndcg']:.4f}\t{h['ndcg_cold']:.4f}\t{h['key']}" for h in history)
    prompt = (
        "You are tuning a next-visit recommender for Mart venues (bodegas, delis, pharmacies, convenience "
        "stores) from Foursquare NYC check-ins. Users also visit Food and Shops venues. Metric: NDCG@10 on "
        "each user's held-out validation Mart visit.\n"
        f"Base models and params (low, high, log, int): {json.dumps(SPACE)}\n"
        f"Blend weights (max, step): {json.dumps(BLEND)} -- cross = weight of Food+Shops history, geo = proximity "
        "to the user's usual area, here = proximity to the user's latest check-in, repeat = revisit boost, seq = attention over the 5 most recent check-ins.\n"
        f"Current best: {key(best)}\nHistory (iter, status, ndcg, ndcg_cold, config):\n{log}\n\n"
        "Propose exactly ONE new config that differs from the best by a single atomic change and is not in "
        "the history. Use the history to decide what is likely to help. Give a one-line hypothesis."
    )
    resp = anthropic.Anthropic().messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Proposal,
    )
    p = resp.parsed_output
    cfg = make(p.model, {n: getattr(p, n) for n in SPACE[p.model] if getattr(p, n) is not None},
               {n: getattr(p, n) for n in BLEND})
    if key(cfg) in {h["key"] for h in history}:
        return None, "LLM proposed a duplicate"
    return cfg, p.hypothesis


def narrate_llm(text: str) -> str:
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": (
            "Write a 120-word, data-driven conclusion for a personalization lead from these recommender "
            "experiments (next Mart venue visit, Foursquare NYC). State the winner, the test lift with its "
            "CI, which signals drove it, the effect on cold-start users, one risk, and a concrete online A/B test "
            f"as next step. Use only numbers given.\n\n{text}")}],
    )
    return next(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- the loop
def run(iters=40, use_llm=False, seed=0, on_keep=None, start=None):
    """Run the loop. Yields one dict per experiment, then a final dict with the test results.

    start: a config to resume from (e.g. a previous run's best). It is proposed right after the
    popularity baseline and goes through the same keep rule; the baseline stays popularity.
    """
    rng = np.random.default_rng(seed)
    items, orders = prepare.load()
    val, test = targets(orders)
    cv = Context(items, orders, val)

    history, best, best_res = [], None, None
    RESULTS.write_text("iter\tstatus\tndcg@10\tndcg_cold\trecall@10\tcoverage\tdelta\tp_improve\tconfig\tdescription\n")
    for it in range(iters):
        if best is None:
            cfg, desc = make("pop"), "popularity"
        elif start is not None and it == 1:
            cfg, desc = make(start["model"], start["params"], start["blend"]), "resume from a previous best"
        elif use_llm:
            cfg, desc = propose_llm(history, best)
        else:
            cfg, desc = propose_heuristic(history, best, rng)
        if cfg is None:
            yield {"stop": desc}
            break

        res = evaluate(cfg, cv)
        if best is None:
            status, delta, prob = "baseline", 0.0, 1.0
        else:
            delta = res["ndcg"] - best_res["ndcg"]
            prob, _ = p_improve(res["per_user"], best_res["per_user"])
            status = "keep" if delta > 0 and prob >= KEEP_PROB else "discard"

        row = {"iter": it, "status": status, "ndcg": res["ndcg"], "ndcg_cold": res["ndcg_cold"],
               "recall": res["recall"], "coverage": res["coverage"], "delta": delta, "p": prob,
               "key": key(cfg), "desc": desc}
        history.append(row)
        with RESULTS.open("a") as f:
            f.write(f"{it}\t{status}\t{res['ndcg']:.4f}\t{res['ndcg_cold']:.4f}\t{res['recall']:.4f}\t"
                    f"{res['coverage']:.3f}\t{delta:+.4f}\t{prob:.2f}\t{key(cfg)}\t{desc}\n")
        if status in ("baseline", "keep"):
            best, best_res = cfg, res
            BEST.write_text(json.dumps(cfg, indent=2) + "\n")
            if on_keep:
                on_keep(f"{status}: {desc} (val ndcg@10 {res['ndcg']:.4f})")
        yield row

    # Final, one-shot check on each user's last Mart visit, which the loop never scored.
    ct = Context(items, orders, test)
    base_cfg = json.loads(history[0]["key"])
    base_t, best_t = evaluate(base_cfg, ct), evaluate(best, ct)
    prob, ci = p_improve(best_t["per_user"], base_t["per_user"])
    write_report(base_cfg, best, base_t, best_t, prob, ci, ct, use_llm)
    yield {"final": True, "base": base_t, "best": best_t, "best_cfg": best, "ci": ci, "p": prob}


def git_commit(msg):
    if (ROOT / ".git").exists():
        subprocess.run(["git", "add", BEST.name, RESULTS.name], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--llm", action="store_true", help="use Claude to propose configs and narrate results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, help="config JSON to continue from, e.g. a previous best_config.json")
    args = ap.parse_args()
    start = json.loads(args.resume.read_text()) if args.resume else None
    for r in run(args.iters, args.llm, args.seed, on_keep=git_commit, start=start):
        if "stop" in r:
            print(f"stop: {r['stop']}")
        elif "final" in r:
            print(f"\ntest ndcg@10: baseline {r['base']['ndcg']:.4f} -> best {r['best']['ndcg']:.4f} "
                  f"(95% CI of delta {r['ci'][0]:+.4f}..{r['ci'][1]:+.4f}). Report: {REPORT.name}")
        else:
            print(f"[{r['iter']:02d}] {r['status']:8s} ndcg={r['ndcg']:.4f} cold={r['ndcg_cold']:.4f} "
                  f"({r['delta']:+.4f}, p={r['p']:.2f})  {r['desc']}")


def write_report(base_cfg, best, base_t, best_t, prob, ci, ct, use_llm):
    log = pd.read_csv(RESULTS, sep="\t").rename(columns={"ndcg@10": "ndcg"})
    kept = log[log["status"].isin(["baseline", "keep"])]
    lift = best_t["ndcg"] / base_t["ndcg"] - 1
    cold_lift = best_t["ndcg_cold"] / base_t["ndcg_cold"] - 1
    table = (
        "| | config | NDCG@10 | NDCG@10, cold users | Recall@10 | Mart coverage |\n|---|---|---|---|---|---|\n"
        f"| baseline | `{key(base_cfg)}` | {base_t['ndcg']:.4f} | {base_t['ndcg_cold']:.4f} | {base_t['recall']:.4f} | {base_t['coverage']:.1%} |\n"
        f"| winner | `{key(best)}` | {best_t['ndcg']:.4f} | {best_t['ndcg_cold']:.4f} | {best_t['recall']:.4f} | {best_t['coverage']:.1%} |\n\n"
        f"Lift: {lift:+.1%} NDCG@10 overall ({cold_lift:+.1%} for the {ct.cold.sum()} cold users with at most one "
        f"earlier Mart visit). Delta 95% CI {ci[0]:+.4f}..{ci[1]:+.4f}; P(delta>0) = {prob:.3f} "
        f"(paired bootstrap over {len(ct.U)} users).\n"
        f"Experiments: {len(log)} run, {len(kept) - 1} kept, {(log['status'] == 'discard').sum()} discarded.\n"
    )
    steps = "\n".join(f"- iter {r.iter}: {r.description} -> val NDCG@10 {r.ndcg:.4f} (cold {r.ndcg_cold:.4f})"
                      for r in kept.itertuples())
    model = log["config"].map(lambda c: json.loads(c)["model"])
    fam = "\n".join(f"| {m} | {v:.4f} |" for m, v in log.groupby(model)["ndcg"].max().sort_values(ascending=False).items())
    on = [n for n, v in best["blend"].items() if v]
    narrative = narrate_llm(table + steps) if use_llm else (
        f"The loop replaced the popularity baseline with **{best['model']}** plus the "
        f"{', '.join(on) if on else 'no extra'} signal(s), lifting held-out NDCG@10 by {lift:+.1%} "
        f"({base_t['ndcg']:.4f} -> {best_t['ndcg']:.4f}); the 95% CI of the delta "
        f"{'excludes zero' if ci[0] > 0 else 'includes zero, so the lift is not conclusive'}. Cold users changed "
        f"by {cold_lift:+.1%}. Check-ins are visits, not deliveries, so treat this as evidence about which signal "
        "families matter, not an estimate for any real app. Next step on real data: a user-randomised A/B test on Mart with "
        "orders per user as the primary metric, and coverage, latency, and Food cannibalisation as guardrails."
    )
    REPORT.write_text(
        "# Autonomous Experimenter report\n\n"
        "Next Mart venue (bodega, deli, pharmacy, convenience store) for each user with 2+ such visits, "
        "using their Food, Mart, and Shops check-ins (Foursquare NYC, public). Loop decisions used each user's "
        "second-to-last Mart visit; the numbers below are one final run on the last visit, which the loop never saw.\n\n"
        f"## Result\n\n{table}\n## Conclusion\n\n{narrative}\n\n"
        f"## Path of kept changes\n\n{steps}\n\n"
        f"## Best validation NDCG@10 per base model\n\n| model | NDCG@10 |\n|---|---|\n{fam}\n\n"
        "Full log: `results.tsv`. Kept changes are git commits (`git log --oneline`).\n"
    )


if __name__ == "__main__":
    main()
