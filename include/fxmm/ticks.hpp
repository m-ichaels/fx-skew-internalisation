#pragma once
// Dukascopy top-of-book ticks: the empirical layer (reference price, spread, volatility).
// Prices are held in pips (1 pip = 1e-4 for EURUSD); inventory in millions of base currency.
#include "types.hpp"
#include "stats.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

namespace fxmm {

struct Tick { Ts ts_ms; double bid, ask, bid_vol, ask_vol; double mid() const { return 0.5 * (bid + ask); } };

struct TickSeries {
    std::string symbol;
    double pip = 1e-4;
    std::vector<Tick> ticks;               // prices in pips
    Ts t0() const { return ticks.empty() ? 0 : ticks.front().ts_ms; }
    Ts t1() const { return ticks.empty() ? 0 : ticks.back().ts_ms; }
    double seconds() const { return (t1() - t0()) / 1e3; }
    // index of the last tick at or before t (ms)
    size_t index_at(Ts t) const {
        auto it = std::upper_bound(ticks.begin(), ticks.end(), t, [](Ts x, const Tick& k) { return x < k.ts_ms; });
        return it == ticks.begin() ? 0 : (size_t)(it - ticks.begin()) - 1;
    }
    double mid_at(Ts t) const { return ticks.empty() ? 0 : ticks[index_at(t)].mid(); }
};

inline TickSeries read_ticks(const std::string& path, const std::string& symbol = "EURUSD", double pip = 1e-4) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw Error("cannot open " + path);
    TickSeries s; s.symbol = symbol; s.pip = pip;
    char line[256]; bool first = true;
    while (std::fgets(line, sizeof line, f)) {
        if (first) { first = false; continue; }
        long long ts; double b, a, bv, av;
        if (std::sscanf(line, "%lld,%lf,%lf,%lf,%lf", &ts, &b, &a, &bv, &av) == 5) s.ticks.push_back({ts, b / pip, a / pip, bv, av});
    }
    std::fclose(f);
    return s;
}

// Summary used for calibration and the README: spread, tick rate, realised vol per second, top sizes.
struct TickStats {
    double seconds = 0, ticks_per_s = 0, spread_mean = 0, spread_p50 = 0, sigma_1s = 0, sigma_1min = 0, top_size_mean = 0;
    std::vector<double> sigma_by_hour;   // realised 1 s vol by UTC hour (pips/sqrt(s))
};
inline TickStats tick_stats(const TickSeries& s) {
    TickStats r; r.seconds = s.seconds(); if (s.ticks.size() < 2) return r;
    r.ticks_per_s = s.ticks.size() / std::max(r.seconds, 1.0);
    std::vector<double> sp, sz; for (auto& k : s.ticks) { sp.push_back(k.ask - k.bid); sz.push_back(0.5 * (k.bid_vol + k.ask_vol)); }
    r.spread_mean = mean(sp); r.spread_p50 = median(sp); r.top_size_mean = mean(sz);
    // 1 s and 60 s sampled mid returns
    std::vector<double> m1, m60; std::vector<int> hour1;
    for (Ts t = s.t0(); t <= s.t1(); t += 1000) { m1.push_back(s.mid_at(t)); hour1.push_back((int)((t / 1000) % 86400 / 3600)); }
    for (Ts t = s.t0(); t <= s.t1(); t += 60000) m60.push_back(s.mid_at(t));
    std::vector<double> r1, r60; for (size_t i = 1; i < m1.size(); ++i) r1.push_back(m1[i] - m1[i - 1]); for (size_t i = 1; i < m60.size(); ++i) r60.push_back(m60[i] - m60[i - 1]);
    r.sigma_1s = stdev(r1); r.sigma_1min = stdev(r60) / std::sqrt(60.0);
    r.sigma_by_hour.assign(24, 0.0);
    for (int h = 0; h < 24; ++h) { std::vector<double> v; for (size_t i = 1; i < m1.size(); ++i) if (hour1[i] == h) v.push_back(m1[i] - m1[i - 1]); r.sigma_by_hour[h] = stdev(v); }
    return r;
}

} // namespace fxmm
