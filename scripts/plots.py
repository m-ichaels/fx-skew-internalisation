#!/usr/bin/env python3
"""Figures from results/*.json.   python scripts/plots.py [results_dir] [fig_dir]"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"
F = sys.argv[2] if len(sys.argv) > 2 else "results/figures"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
LABEL = {"AS": "Avellaneda-Stoikov", "GLFT": "GLFT closed form", "QH": "quadratic-Hamiltonian", "QH_adverse": "QH + adverse selection",
         "HJB_2023": "BBG 2023 (numerical)", "HJB_noread": "+ adverse selection", "HJB": "+ adverse selection + price reading",
         "HJB_tox": "HJB + toxicity routing", "HJB_nwz": "HJB + anticipatory hedging", "HJB_noadv_noread": "BBG 2023 (numerical)", "HJB_noadv": "readers only"}


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fig_quotes():
    q = load("quotes.json")
    if not q:
        return
    tiers = [t["name"] for t in q["tiers"]["tiers"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    want = ["HJB_2023", "HJB_noread", "HJB", "QH", "GLFT"]
    for name in want:
        e = next((x for x in q["quoters"] if x["quoter"] == name or (name == "HJB_2023" and x["quoter"] == "HJB_noadv_noread")), None)
        if not e: continue
        qs = [r["q"] for r in e["rows"]]
        ax[0].plot(qs, [r["bid"][0] for r in e["rows"]], label=LABEL.get(name, name))
        ax[1].plot(qs, [r["bid"][3] for r in e["rows"]], label=LABEL.get(name, name))
        ax[2].plot(qs, [r["hedge"] for r in e["rows"]], label=LABEL.get(name, name))
    ax[0].set_title(f"bid offset shown to '{tiers[0]}' (pips)"); ax[0].set_xlabel("inventory q (M)"); ax[0].legend(fontsize=6)
    ax[1].set_title(f"bid offset shown to '{tiers[3]}' (readers)"); ax[1].set_xlabel("inventory q (M)")
    ax[2].set_title("hedge at review (M): the internalisation band"); ax[2].set_xlabel("inventory q (M)"); ax[2].axhline(0, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "quotes.png")); plt.close(fig)
    # skew per tier: bid - ask over q, with and without readers
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for k, name in enumerate(["HJB_noread", "HJB"]):
        e = next((x for x in q["quoters"] if x["quoter"] == name), None)
        if not e: continue
        qs = [r["q"] for r in e["rows"]]
        for i, t in enumerate(tiers): ax[k].plot(qs, [0.5 * (r["bid"][i] - r["ask"][i]) for r in e["rows"]], label=t)
        ax[k].set_title(LABEL[name] + ": skew (bid-ask)/2 by tier"); ax[k].set_xlabel("inventory q (M)"); ax[k].legend(fontsize=7)
    ax[0].set_ylabel("skew, pips (+ = wants to sell)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "skew_by_tier.png")); plt.close(fig)


def fig_frontier():
    f = load("frontier.json")
    if not f:
        return
    cells = f["cells"]; quoters = list(dict.fromkeys(c["quoter"] for c in cells))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for qn in quoters:
        cs = sorted([c for c in cells if c["quoter"] == qn], key=lambda c: c["gamma"])
        x = [c["pnl_1min_std_usd"] for c in cs]; y = [c["pnl_usd_mean"] for c in cs]; lo = [c["pnl_usd_mean"] - c["pnl_usd_lo"] for c in cs]; hi = [c["pnl_usd_hi"] - c["pnl_usd_mean"] for c in cs]
        ax[0].errorbar(x, y, yerr=[lo, hi], fmt="-o", ms=3, capsize=2, label=LABEL.get(qn, qn))
        ax[1].plot([c["gamma"] for c in cs], [c["pnl_per_unit_var"] for c in cs], "-o", ms=3, label=LABEL.get(qn, qn))
    ax[0].set_xscale("log"); ax[0].set_xlabel("std of 1-minute P&L ($, log)"); ax[0].set_ylabel("daily P&L ($, mean over days x seeds, 95% CI)"); ax[0].set_title("Risk-return frontier over risk aversion"); ax[0].legend(fontsize=6)
    ax[1].set_xscale("log"); ax[1].set_xlabel("risk aversion gamma"); ax[1].set_ylabel("P&L per unit of 1-min variance"); ax[1].set_title("P&L per unit variance"); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(F, "frontier.png")); plt.close(fig)
    # decomposition at the middle gamma
    gam = sorted(set(c["gamma"] for c in cells))[len(set(c["gamma"] for c in cells)) // 2]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    comps = ["spread", "inventory", "adverse_selection", "hedge_execution", "hedge_impact"]
    bp = np.zeros(len(quoters)); bn = np.zeros(len(quoters))
    for comp in comps:
        v = np.array([next(c["pnl_decomposition_usd"][comp] for c in cells if c["quoter"] == qn and c["gamma"] == gam) for qn in quoters])
        pos = np.where(v > 0, v, 0); neg = np.where(v < 0, v, 0)
        ax.bar(range(len(quoters)), pos, bottom=bp, label=comp); ax.bar(range(len(quoters)), neg, bottom=bn, color=ax.patches[-1].get_facecolor()); bp += pos; bn += neg
    tot = [next(c["pnl_usd_mean"] for c in cells if c["quoter"] == qn and c["gamma"] == gam) for qn in quoters]
    ax.plot(range(len(quoters)), tot, "k_", ms=18, mew=2, label="total")
    ax.set_xticks(range(len(quoters))); ax.set_xticklabels([LABEL.get(q, q) for q in quoters], rotation=20, ha="right", fontsize=7); ax.set_ylabel("$ per day"); ax.set_title(f"P&L decomposition at gamma = {gam} (sums by construction)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "decomposition.png")); plt.close(fig)


def fig_signatures():
    r = load("run_HJB.json")
    if not r:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for t in r["tier_markouts"]:
        ax[0].plot([m["horizon_s"] for m in t["markouts"]], [m["markout_pips_client_favour"] for m in t["markouts"]], "-o", ms=3, label=f"{t['tier']} (n={t['n']})")
    ax[0].set_xscale("log"); ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("horizon after trade (s)"); ax[0].set_ylabel("mid move in the client's favour (pips)"); ax[0].set_title("Price signatures by tier (Oomen 2019)"); ax[0].legend(fontsize=7)
    ht = r.get("holding_times_s", [])
    if ht:
        ax[1].hist(np.clip(ht, 0, 1200), bins=40); ax[1].set_xlabel("time from opening a position to flat (s)"); ax[1].set_title(f"Internalisation holding times (median {np.median(ht):.0f} s)")
    tr = r.get("trades", [])
    if tr:
        for tier in sorted(set(t["tier"] for t in tr)):
            s = [t["tox"] for t in tr if t["tier"] == tier]
            ax[2].hist(s, bins=30, alpha=0.5, label=r["tier_markouts"][tier]["tier"])
        ax[2].set_xlabel("toxicity score at trade time"); ax[2].set_title("Online classifier scores by tier"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "signatures.png")); plt.close(fig)


def fig_regime_sens():
    rg = load("regime.json"); se = load("sensitivity.json"); ll = load("lastlook.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    if rg:
        cells = rg["cells"]; x = np.arange(len(cells))
        ax[0].bar(x - 0.2, [c["pnl_before_usd"] for c in cells], 0.4, yerr=[[c["pnl_before_usd"] - c["pnl_before_lo"] for c in cells], [c["pnl_before_hi"] - c["pnl_before_usd"] for c in cells]], capsize=2, label="before the switch")
        ax[0].bar(x + 0.2, [c["pnl_after_usd"] for c in cells], 0.4, yerr=[[c["pnl_after_usd"] - c["pnl_after_lo"] for c in cells], [c["pnl_after_hi"] - c["pnl_after_usd"] for c in cells]], capsize=2, label=f"after (intensity x{rg['lambda_mult']:g}, toxicity x{rg['mu_mult']:g})")
        ax[0].set_xticks(x); ax[0].set_xticklabels([LABEL.get(c["quoter"], c["quoter"]) for c in cells], rotation=25, ha="right", fontsize=6); ax[0].set_ylabel("$ per half day"); ax[0].set_title("Client-flow regime change mid-episode"); ax[0].legend(fontsize=7); ax[0].axhline(0, color="k", lw=0.8)
    if se:
        for qn in dict.fromkeys(e["quoter"] for e in se["latency"]):
            es = [e for e in se["latency"] if e["quoter"] == qn]
            ax[1].errorbar([e["latency_ms"] for e in es], [e["pnl_usd_mean"] for e in es], yerr=[[e["pnl_usd_mean"] - e["pnl_usd_lo"] for e in es], [e["pnl_usd_hi"] - e["pnl_usd_mean"] for e in es]], fmt="-o", ms=3, capsize=2, label=LABEL.get(qn, qn))
        ax[1].set_xlabel("quote-update latency (ms)"); ax[1].set_ylabel("$ per day"); ax[1].set_title("Sensitivity to latency"); ax[1].legend(fontsize=6)
        for qn in dict.fromkeys(e["quoter"] for e in se["toxicity_calibration"]):
            es = [e for e in se["toxicity_calibration"] if e["quoter"] == qn]
            ax[2].errorbar([e["mu_mult"] for e in es], [e["pnl_usd_mean"] for e in es], yerr=[[e["pnl_usd_mean"] - e["pnl_usd_lo"] for e in es], [e["pnl_usd_hi"] - e["pnl_usd_mean"] for e in es]], fmt="-o", ms=3, capsize=2, label=LABEL.get(qn, qn))
        ax[2].set_xlabel("true toxicity / calibrated toxicity"); ax[2].set_ylabel("$ per day"); ax[2].set_title("Sensitivity to the toxicity calibration"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(F, "regime_sensitivity.png")); plt.close(fig)
    if ll:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        labels = [f"hold {e['hold_ms']:g} ms\ntol {e['tolerance_pips']:g}" if e["tolerance_pips"] > 0 else "no last look" for e in ll]
        ax[0].bar(range(len(ll)), [e["pnl_usd_mean"] for e in ll], yerr=[[e["pnl_usd_mean"] - e["pnl_usd_lo"] for e in ll], [e["pnl_usd_hi"] - e["pnl_usd_mean"] for e in ll]], capsize=2); ax[0].set_xticks(range(len(ll))); ax[0].set_xticklabels(labels, fontsize=6); ax[0].set_ylabel("$ per day"); ax[0].set_title("Last look as a control: P&L")
        ax[1].bar(range(len(ll)), [100 * e["rejection_rate"] for e in ll]); ax[1].set_xticks(range(len(ll))); ax[1].set_xticklabels(labels, fontsize=6); ax[1].set_ylabel("rejection rate (%)"); ax[1].set_title("Rejection rate")
        fig.tight_layout(); fig.savefig(os.path.join(F, "lastlook.png")); plt.close(fig)


if __name__ == "__main__":
    fig_quotes(); fig_frontier(); fig_signatures(); fig_regime_sens(); print("figures written to", F)
