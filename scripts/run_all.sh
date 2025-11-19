#!/usr/bin/env bash
# Full pipeline: ticks -> quoters -> experiments -> figures -> summary -> report.  Usage: scripts/run_all.sh
set -euo pipefail
B=./build/fxmm
DAYS=data/ticks/EURUSD_2025-04-14.csv,data/ticks/EURUSD_2025-04-15.csv,data/ticks/EURUSD_2025-04-16.csv,data/ticks/EURUSD_2025-04-17.csv
PSI=0.15   # interbank half spread + fee (pips per million); the histdata spread is a retail feed's, see README
mkdir -p results/figures
for d in 14 15 16 17 18; do $B ticks --file data/ticks/EURUSD_2025-04-$d.csv --out results/ticks_2025-04-$d.json > /dev/null; done
$B quotes --sigma 0.3 --gamma 0.02 --psi $PSI --out results/quotes.json > /dev/null
python tools/mbt_gym_check.py results/quotes.json 2>/dev/null | tail -2 | tee results/mbt_gym_check.txt
$B run --quoter HJB --ticks data/ticks/EURUSD_2025-04-14.csv --psi $PSI --seed 3 --out results/run_HJB.json > /dev/null
$B run --quoter HJB_2023 --ticks data/ticks/EURUSD_2025-04-14.csv --psi $PSI --seed 3 --out results/run_HJB_2023.json > /dev/null
$B frontier --ticks $DAYS --psi $PSI --Q 60 --hedge-max 20 --seeds 4 --gammas 0.005,0.01,0.02,0.05,0.1 --out results/frontier.json > /dev/null
$B regime --ticks $DAYS --psi $PSI --Q 60 --hedge-max 20 --seeds 4 --out results/regime.json > /dev/null
$B sensitivity --ticks $DAYS --psi $PSI --seeds 3 --out results/sensitivity.json > /dev/null
$B lastlook --ticks $DAYS --psi $PSI --seeds 3 --out results/lastlook.json > /dev/null
./build/fxmm_tests | tee results/tests.txt
python scripts/plots.py results results/figures
python scripts/summarize.py results
python scripts/report.py results report.pdf
echo done
