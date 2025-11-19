#!/usr/bin/env python3
"""Check fxmm's Avellaneda-Stoikov quotes against mbt_gym's AvellanedaStoikovAgent (Jerome et al., ICAIF 2023).

    python tools/mbt_gym_check.py results/quotes.json [third_party/mbt_gym]

fxmm fits A e^{-k delta} to each tier's logistic response and averages k by intensity; the same fit
is redone here, an mbt_gym TradingEnvironment is built with (sigma, k, T), and the agent's
half-spreads are compared with fxmm's AS ladder over the inventory grid.
"""
import json
import os
import sys

import numpy as np

qfile = sys.argv[1] if len(sys.argv) > 1 else "results/quotes.json"
root = sys.argv[2] if len(sys.argv) > 2 else "third_party/mbt_gym"
sys.path.insert(0, root)
try:
    from mbt_gym.agents.BaselineAgents import AvellanedaStoikovAgent
    from mbt_gym.gym.TradingEnvironment import TradingEnvironment
    from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
    from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
    from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
    from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
except Exception as e:  # noqa: BLE001
    print("mbt_gym not importable:", e); sys.exit(2)

q = json.load(open(qfile))
tiers = q["tiers"]["tiers"]
sigma, gamma = q["market"]["sigma"], q["market"]["gamma"]
H = 60.0  # fxmm AS_horizon

# same exponential fit as fxmm: log f(delta) ~ a - k delta on [0, 2] pips, intensity-weighted mean of k
ks, ws = [], []
for t in tiers:
    d = np.arange(0, 2.0001, 0.05); f = 1 / (1 + np.exp(t["alpha"] + t["beta"] * d))
    k = -np.polyfit(d, np.log(np.maximum(f, 1e-12)), 1)[0]; ks.append(k); ws.append(t["lambda"])
k = float(np.average(ks, weights=ws))

env = TradingEnvironment(terminal_time=H, n_steps=int(H), model_dynamics=LimitOrderModelDynamics(
    midprice_model=BrownianMotionMidpriceModel(volatility=sigma, terminal_time=H, step_size=1.0),
    arrival_model=PoissonArrivalModel(intensity=np.array([1.0, 1.0]), step_size=1.0),
    fill_probability_model=ExponentialFillFunction(fill_exponent=k, step_size=1.0)))
agent = AvellanedaStoikovAgent(risk_aversion=gamma, env=env)

rows = next(x for x in q["quoters"] if x["quoter"] == "AS")["rows"]
maxerr = 0.0
for r in rows:
    inv = r["q"]
    spread = agent._get_spread(0.0); adj = agent._get_price_adjustment(inv, 0.0)
    bid_ref, ask_ref = max(spread / 2 + adj, 0.0), max(spread / 2 - adj, 0.0)
    err = max(abs(r["bid"][0] - bid_ref), abs(r["ask"][0] - ask_ref)); maxerr = max(maxerr, err)
print(f"k = {k:.3f}, gamma = {gamma}, sigma = {sigma}, H = {H}: max |fxmm AS - mbt_gym AS| over q in [-Q, Q] = {maxerr:.2e} pips")
print("PASS" if maxerr < 1e-6 else "FAIL")
sys.exit(0 if maxerr < 1e-6 else 1)
