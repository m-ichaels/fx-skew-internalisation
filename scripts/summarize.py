#!/usr/bin/env python3
"""Key numbers from results/*.json as Markdown (results/summary.md)."""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


out = []; P = out.append
P("## Reference price data (histdata.com EURUSD ticks, April 2025)\n")
P("| day | ticks | median spread (pips) | σ 1 s (pips/√s) | σ from 1-min (pips/√s) |"); P("|---|---|---|---|---|")
for d in ("14", "15", "16", "17", "18"):
    t = load(f"ticks_2025-04-{d}.json")
    if t: P(f"| 2025-04-{d}{' (Good Friday, excluded)' if d == '18' else ''} | {t['ticks']:,} | {t['spread_p50_pips']:.2f} | {t['sigma_1s_pips']:.3f} | {t['sigma_1min_per_sqrt_s']:.3f} |")

q = load("quotes.json")
if q:
    tiers = [t["name"] for t in q["tiers"]["tiers"]]
    P("\n## Quoting policies (σ = %.2f pips/√s, γ = %g, ψ = %g pips/M)\n" % (q["market"]["sigma"], q["market"]["gamma"], q["market"]["psi"]))
    P("| quoter | offsets at q = 0 by tier (pips) | skew slope by tier (pips per M, from q = ±10) | hedge band (M) |"); P("|---|---|---|---|")
    for e in q["quoters"]:
        rows = {r["q"]: r for r in e["rows"]}
        off = "/".join(f"{rows[0]['bid'][i]:.2f}" for i in range(len(tiers)))
        slope = "/".join(f"{(rows[10]['bid'][i] - rows[-10]['bid'][i]) / 20:.3f}" for i in range(len(tiers)))
        band = e.get("band_M", 0)
        P(f"| {e['quoter']} | {off} | {slope} | {band:.1f} |")
    P(f"\ntiers: {', '.join(tiers)}")
    mb = os.path.join(R, "mbt_gym_check.txt")
    if os.path.exists(mb): P("\nmbt_gym check: " + open(mb).read().strip().replace("\n", "; "))

f = load("frontier.json")
if f:
    P(f"\n## Frontier ({len(f['days'])} days × {f['seeds']} seeds per cell)\n")
    P("| quoter | γ | daily P&L $ (95% CI) | 1-min P&L std $ | P&L / variance | internalisation | holding time p50 (s) | max |q| (M) | spread | inventory | adverse | hedge exec | hedge impact |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in f["cells"]:
        d = c["pnl_decomposition_usd"]
        P(f"| {c['quoter']} | {c['gamma']:g} | {c['pnl_usd_mean']:,.0f} [{c['pnl_usd_lo']:,.0f}, {c['pnl_usd_hi']:,.0f}] | {c['pnl_1min_std_usd']:,.0f} | {c['pnl_per_unit_var']:.3f} | {c['internalisation_ratio']:.2f} | {(c['holding_time_s_p50'] or 0):.0f} | {c['max_abs_q']:.1f} | {d['spread']:,.0f} | {d['inventory']:,.0f} | {d['adverse_selection']:,.0f} | {d['hedge_execution']:,.0f} | {d['hedge_impact']:,.0f} |")

r = load("run_HJB.json")
if r:
    P("\n## One day of the full model (HJB with adverse selection and price reading, 2025-04-14, seed 3)\n")
    p = r["pnl_pips_M"]
    P(f"- P&L ${r['pnl_usd']:,.0f} = spread {100*p['spread']:,.0f} + inventory {100*p['inventory']:,.0f} + adverse selection {100*p['adverse_selection']:,.0f} + hedge execution {100*p['hedge_execution']:,.0f} + hedge impact {100*p['hedge_impact']:,.0f} (residual {p['decomposition_residual']:.1e})")
    P(f"- {r['n_trades']:,} client trades ({r['client_volume_M']:,.0f} M), {r['n_hedges']:,} hedges ({r['hedged_volume_M']:,.0f} M), internalisation ratio {r['internalisation_ratio']:.2f}, holding time p50 {(r['holding_time_s_p50'] or 0):.0f} s / mean {(r['holding_time_s_mean'] or 0):.0f} s, |q| mean {r['mean_abs_q']:.1f} M max {r['max_abs_q']:.1f} M")
    P("\n| tier | trades | mean offset paid (pips) | mark-out 0.1 s | 1 s | 10 s | 30 s | 60 s (pips, client's favour) |"); P("|---|---|---|---|---|---|---|---|")
    for t in r["tier_markouts"]:
        P(f"| {t['tier']} | {t['n']:,} | {t['mean_offset_pips']:.2f} | " + " | ".join(f"{m['markout_pips_client_favour']:+.2f}" for m in t["markouts"]) + " |")
    c = r["classifier"]
    P(f"\nclassifier: {c['n_labelled']:,} labelled trades, {c['n_toxic_labels']:,} toxic labels; weights {['%.2f' % w for w in c['weights']]}")

rg = load("regime.json")
if rg:
    P(f"\n## Regime change at mid-day (client intensity ×{rg['lambda_mult']:g}, toxicity ×{rg['mu_mult']:g})\n")
    P("| quoter | P&L before ($, 95% CI) | P&L after | survives (CI > 0) |"); P("|---|---|---|---|")
    for c in rg["cells"]: P(f"| {c['quoter']} | {c['pnl_before_usd']:,.0f} [{c['pnl_before_lo']:,.0f}, {c['pnl_before_hi']:,.0f}] | {c['pnl_after_usd']:,.0f} [{c['pnl_after_lo']:,.0f}, {c['pnl_after_hi']:,.0f}] | {'yes' if c['survives'] else 'no'} |")

se = load("sensitivity.json")
if se:
    P("\n## Sensitivity\n")
    P("| quoter | latency 0 ms | 50 ms | 200 ms | 1000 ms |"); P("|---|---|---|---|---|")
    for qn in dict.fromkeys(e["quoter"] for e in se["latency"]):
        P(f"| {qn} | " + " | ".join(f"{e['pnl_usd_mean']:,.0f}" for e in se["latency"] if e["quoter"] == qn) + " |")
    P("\n| quoter | true toxicity ×0 | ×0.5 | ×1 | ×2 | ×4 |"); P("|---|---|---|---|---|---|")
    for qn in dict.fromkeys(e["quoter"] for e in se["toxicity_calibration"]):
        P(f"| {qn} | " + " | ".join(f"{e['pnl_usd_mean']:,.0f}" for e in se["toxicity_calibration"] if e["quoter"] == qn) + " |")

ll = load("lastlook.json")
if ll:
    P("\n## Last look\n")
    P("| hold (ms) | tolerance (pips) | P&L $ (95% CI) | rejection rate | 10 s mark-out of accepted trades (client's favour, pips) |"); P("|---|---|---|---|---|")
    for e in ll: P(f"| {e['hold_ms']:g} | {e['tolerance_pips'] if e['tolerance_pips'] > 0 else 'none'} | {e['pnl_usd_mean']:,.0f} [{e['pnl_usd_lo']:,.0f}, {e['pnl_usd_hi']:,.0f}] | {100*e['rejection_rate']:.1f}% | {e['accepted_markout_10s_client_favour']:+.3f} |")

ts = os.path.join(R, "tests.txt")
if os.path.exists(ts): P("\n## Tests\n\n```\n" + open(ts).read().strip() + "\n```")
open(os.path.join(R, "summary.md"), "w", encoding="utf-8").write("\n".join(out))
print("written", os.path.join(R, "summary.md"))
