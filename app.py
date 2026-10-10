"""Gradio demo: run the autonomous experimenter live, and inspect recommendations for one user."""

import json
import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import experimenter as ex
import prepare

ITEMS, ORDERS = prepare.load()
VAL, TEST = ex.targets(ORDERS)
CT = ex.Context(ITEMS, ORDERS, TEST)
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
COLORS = {"quik": "#16a34a", "food": "#f97316", "shops": "#6366f1"}
COLUMNS = ["iter", "status", "NDCG@10", "cold NDCG@10", "delta", "P(gain>0)", "change"]


def log_table(rows):
    return pd.DataFrame([[r["iter"], r["status"], round(r["ndcg"], 4), round(r["ndcg_cold"], 4),
                          round(r["delta"], 4), round(r["p"], 2), r["desc"]] for r in rows], columns=COLUMNS)


def saved_rows():
    if not ex.RESULTS.exists():
        return []
    d = pd.read_csv(ex.RESULTS, sep="\t")
    return [{"iter": r["iter"], "status": r["status"], "ndcg": r["ndcg@10"], "ndcg_cold": r["ndcg_cold"],
             "delta": r["delta"], "p": r["p_improve"], "desc": r["description"]} for _, r in d.iterrows()]


def chart(rows):
    fig = go.Figure()
    if rows:
        it = [r["iter"] for r in rows]
        best = np.maximum.accumulate([r["ndcg"] if r["status"] != "discard" else 0 for r in rows])
        for status, color in [("keep", "#16a34a"), ("discard", "#9ca3af"), ("baseline", "#111827")]:
            pts = [r for r in rows if r["status"] == status]
            fig.add_scatter(x=[r["iter"] for r in pts], y=[r["ndcg"] for r in pts], mode="markers", name=status,
                            marker=dict(color=color, size=9), text=[r["desc"] for r in pts])
        fig.add_scatter(x=it, y=best, mode="lines", line=dict(shape="hv", color="#16a34a"), name="best so far")
    fig.update_layout(xaxis_title="iteration", yaxis_title="validation NDCG@10", height=380,
                      margin=dict(l=40, r=10, t=10, b=40), legend=dict(orientation="h"))
    return fig


def run_loop(iters, use_llm):
    rows = []
    for r in ex.run(int(iters), use_llm=use_llm and HAS_KEY):
        if "stop" in r:
            continue
        if "final" in r:
            yield log_table(rows), chart(rows), ex.REPORT.read_text()
            return
        rows.append(r)
        yield log_table(rows), chart(rows), f"Running... {len(rows)} experiments so far."


SCORES = {}


def top10(cfg):
    k = ex.key(cfg)
    if k not in SCORES:
        SCORES[k] = np.argsort(-ex.score(cfg, CT), axis=1)[:, :10]
    return SCORES[k]


def explore(user):
    r = int(np.flatnonzero(CT.U == int(user))[0])
    best = json.loads(ex.BEST.read_text()) if ex.BEST.exists() else ex.make("pop")
    truth = CT.Q[CT.truth[r]]
    hist = ORDERS[(ORDERS["user"] == int(user)) & (ORDERS["hour"] < TEST.set_index("user").loc[int(user), "hour"])]
    seen = hist.merge(ITEMS, on="item").drop_duplicates("item")

    fig = go.Figure()
    for v, g in seen.groupby("vertical_x"):
        fig.add_scattermap(lat=g["lat"], lon=g["lng"], mode="markers", name=f"visited: {v}",
                           marker=dict(size=8, color=COLORS[v], opacity=0.5), text=g["category"])
    rows = []
    for name, cfg, symbol_color in [("popularity", ex.make("pop"), "#9ca3af"), ("winner", best, "#dc2626")]:
        rec = ITEMS.iloc[CT.Q[top10(cfg)[r]]]
        fig.add_scattermap(lat=rec["lat"], lon=rec["lng"], mode="markers", name=f"top-10: {name}",
                           marker=dict(size=13, color=symbol_color), text=rec["category"])
        for rank, (_, it) in enumerate(rec.iterrows(), 1):
            rows.append([name, rank, f"#{it['item']}", it["category"], round(float(CT.near[r, np.searchsorted(CT.Q, it["item"])]), 2),
                         "HIT" if it["item"] == truth else ""])
    t = ITEMS.iloc[truth]
    fig.add_scattermap(lat=[t["lat"]], lon=[t["lng"]], mode="markers", name="actual next Quik visit",
                       marker=dict(size=20, color="#facc15"), text=[t["category"]])
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=t["lat"], lon=t["lng"]), zoom=12),
                      height=480, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
    counts = hist["vertical"].value_counts().reindex(["food", "quik", "shops"], fill_value=0)
    info = (f"**User {user}**: {counts['food']} Food, {counts['quik']} Quik, {counts['shops']} Shops check-ins "
            f"before the target. Actual next Quik visit: **{t['category']}** "
            f"({'repeat' if truth in set(hist['item']) else 'new venue'}).")
    table = pd.DataFrame(rows, columns=["model", "rank", "venue", "category", "proximity (0-1)", "hit"])
    return fig, info, table


INTRO = """# Autonomous Experimenter: cross-vertical personalization

An autoresearch-style loop (**modify, verify, keep or discard**) that tunes a recommender on its own. Task:
predict each user's next **Quik-like** visit (bodega, deli, pharmacy, convenience store) from their **Food,
Quik, and Shops** history, using the public [Foursquare NYC check-ins](https://huggingface.co/datasets/yyh11/FourSquare_checkin)
(1,080 users, 4,124 venues). Every iteration changes one thing; a paired bootstrap keeps it only if the gain
is not noise; the winner is scored once on held-out visits. Code: [GitHub](https://github.com/dzakybd/autonomous-experimenter)
"""

with gr.Blocks(title="Autonomous Experimenter") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Run the loop"):
        with gr.Row():
            iters = gr.Slider(5, 60, value=40, step=1, label="Iterations")
            llm = gr.Checkbox(label="Claude proposes changes" + ("" if HAS_KEY else " (needs ANTHROPIC_API_KEY)"),
                              value=False, interactive=HAS_KEY)
            go_btn = gr.Button("Run experiments", variant="primary")
        rows0 = saved_rows()
        plot = gr.Plot(chart(rows0))
        report = gr.Markdown(ex.REPORT.read_text() if ex.REPORT.exists() else "")
        log = gr.Dataframe(log_table(rows0), label="results.tsv", wrap=True)
        go_btn.click(run_loop, [iters, llm], [log, plot, report], concurrency_limit=1)
    with gr.Tab("Explore a user"):
        users = sorted(CT.U.tolist())
        best_cfg = json.loads(ex.BEST.read_text()) if ex.BEST.exists() else ex.make("pop")
        win = (top10(best_cfg) == CT.truth[:, None]).any(1) & ~(top10(ex.make("pop")) == CT.truth[:, None]).any(1)
        first = int(CT.U[np.flatnonzero(win)[0]]) if win.any() else users[0]  # a user the winner gets right
        user = gr.Dropdown(users, value=first, label="User (held-out test visit)")
        info = gr.Markdown()
        user_map = gr.Plot()
        recs = gr.Dataframe(label="Top-10 Quik venues: popularity vs winner")
        user.change(explore, user, [user_map, info, recs])
        demo.load(explore, user, [user_map, info, recs])

if __name__ == "__main__":
    demo.launch()
