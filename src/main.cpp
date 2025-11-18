// fxmm - FX dealer market-making simulator (skew, internalisation, adverse selection, price reading).
//
//   fxmm ticks      --file data/ticks/EURUSD_2025-04-14.csv                    tick statistics (sigma, spread)
//   fxmm quotes     --config configs/clients.json --sigma S [--gamma G]          ladders of every quoter over q
//   fxmm run        --quoter HJB --ticks F [--seed N] [...]                      one episode, P&L decomposition
//   fxmm frontier   --ticks F1,F2 --quoters A,B --gammas g1,g2 --seeds N         risk-return frontier with seed bands
//   fxmm regime     --ticks F --quoters A,B --seeds N                            client-flow regime change mid-episode
//   fxmm sensitivity --ticks F --quoters A,B --seeds N                           latency and toxicity-calibration sweeps
#include "fxmm/dealer.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <filesystem>

using namespace fxmm;

struct Args {
    std::map<std::string, std::string> kv; std::vector<std::string> pos;
    std::string get(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::vector<std::string> list(const std::string& k, const std::vector<std::string>& d) const {
        if (!has(k)) return d; std::vector<std::string> v; std::string s = get(k), cur;
        for (char c : s) { if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; } if (!cur.empty()) v.push_back(cur); return v;
    }
    std::vector<double> nums(const std::string& k, const std::vector<double>& d) const { if (!has(k)) return d; std::vector<double> v; for (auto& s : list(k, {})) v.push_back(std::atof(s.c_str())); return v; }
};
static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 2; i < argc; ++i) { std::string s = argv[i]; if (s.rfind("--", 0) == 0) { std::string k = s.substr(2); if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.kv[k] = argv[++i]; else a.kv[k] = "1"; } else a.pos.push_back(s); }
    return a;
}
static void logmsg(const std::string& s) { std::cerr << "[fxmm] " << s << std::endl; }

static MarketParams market_from(const Args& a, const TickSeries* ticks) {
    MarketParams m;
    if (ticks) { TickStats st = tick_stats(*ticks); m.sigma = st.sigma_1s; m.psi = std::max(0.05, 0.5 * st.spread_p50); }
    m.sigma = a.num("sigma", m.sigma); m.gamma = a.num("gamma", m.gamma); m.psi = a.num("psi", m.psi); m.eta = a.num("eta", m.eta); m.kappa = a.num("kappa", m.kappa);
    m.Q = a.num("Q", m.Q); m.dq = a.num("dq", m.dq); m.dt = a.num("dt", m.dt); m.hedge_max = a.num("hedge-max", m.hedge_max);
    return m;
}
static ClientConfig clients_from(const Args& a) { return a.has("config") ? ClientConfig::load(a.get("config")) : ClientConfig::defaults(); }

// Quoter factory. Names: AS, GLFT, QH, QH_adverse, HJB, HJB_noadv, HJB_noread, HJB_nohedge, HJB_tox (HJB + toxicity routing),
// HJB_nwz (HJB + anticipatory hedging), HJB_ll (HJB + last look).  The suffix variants share the HJB policy and switch SimConfig flags.
static std::unique_ptr<Quoter> make_quoter(const std::string& name, const ClientConfig& c, MarketParams mp) {
    if (name == "AS") return std::make_unique<ASQuoter>(c, mp);
    if (name == "GLFT") return std::make_unique<GLFTQuoter>(c, mp);
    if (name == "QH") return std::make_unique<QHQuoter>(c, mp, false);
    if (name == "QH_adverse") return std::make_unique<QHQuoter>(c, mp, true);
    HJBOptions o;
    if (name == "HJB_noadv") o.adverse = false;
    if (name == "HJB_noread") o.readers = false;
    if (name == "HJB_nohedge") o.hedging = false;
    if (name == "HJB_2023") { o.adverse = false; o.readers = false; }     // the base model without the 2025 extension
    return std::make_unique<HJBQuoter>(c, mp, o);
}
static void apply_variant(const std::string& name, SimConfig& sc) {
    if (name == "HJB_tox") sc.tox_route = true;
    if (name == "HJB_nwz") sc.anticipate = true;
    if (name == "HJB_ll") { sc.last_look = true; }
}
static std::string base_name(const std::string& n) { return (n == "HJB_tox" || n == "HJB_nwz" || n == "HJB_ll") ? "HJB" : n; }

static Json result_json(const SimResult& r, bool with_paths) {
    Json j = Json::object(); j["quoter"] = r.quoter;
    Json p = Json::object(); p["spread"] = r.pnl.spread; p["inventory"] = r.pnl.inventory; p["adverse_selection"] = r.pnl.adverse; p["hedge_execution"] = r.pnl.hedge_exec; p["hedge_impact"] = r.pnl.hedge_impact; p["total"] = r.pnl.total; p["decomposition_residual"] = r.pnl.check(); j["pnl_pips_M"] = p;
    j["pnl_usd"] = r.pnl_usd; j["pnl_1min_std"] = r.pnl_std_proxy; j["mean_abs_q"] = r.mean_abs_q; j["max_abs_q"] = r.max_abs_q; j["q_std"] = r.q_std;
    j["n_trades"] = (long long)r.n_trades; j["n_hedges"] = (long long)r.n_hedges; j["n_rejected"] = (long long)r.n_rejected; j["n_externalised"] = (long long)r.n_externalised;
    j["client_volume_M"] = r.client_volume; j["hedged_volume_M"] = r.hedged_volume; j["internalisation_ratio"] = r.internalisation_ratio;
    j["holding_time_s_p50"] = median(r.holding_times_s); j["holding_time_s_mean"] = mean(r.holding_times_s); j["n_holding_periods"] = (long long)r.holding_times_s.size();
    j["tier_markouts"] = r.tier_markouts; j["classifier"] = r.classifier;
    if (with_paths) { j["pnl_1min"] = Json(r.pnl_1min); j["q_1min"] = Json(r.q_path_1min); j["holding_times_s"] = Json(r.holding_times_s);
        Json tr = Json::array(); for (auto& t : r.trades) if (t.markouts.size() == 5) { Json x = Json::object(); x["tier"] = t.tier; x["z"] = t.z; x["offset"] = t.offset; x["tox"] = t.tox_score; x["mo"] = Json(t.markouts); tr.push(x); } j["trades"] = tr; }
    return j;
}

static int cmd_ticks(const Args& a) {
    TickSeries s = read_ticks(a.get("file"));
    TickStats st = tick_stats(s);
    Json j = Json::object(); j["file"] = a.get("file"); j["ticks"] = (long long)s.ticks.size(); j["seconds"] = st.seconds; j["ticks_per_s"] = st.ticks_per_s;
    j["spread_mean_pips"] = st.spread_mean; j["spread_p50_pips"] = st.spread_p50; j["sigma_1s_pips"] = st.sigma_1s; j["sigma_1min_per_sqrt_s"] = st.sigma_1min; j["top_size_mean_M"] = st.top_size_mean; j["sigma_by_hour"] = Json(st.sigma_by_hour);
    if (a.has("out")) write_file(a.get("out"), j.dump(1));
    std::cout << j.dump(1) << std::endl; return 0;
}

static int cmd_quotes(const Args& a) {
    ClientConfig c = clients_from(a); MarketParams mp = market_from(a, nullptr);
    std::vector<std::string> names = a.list("quoters", {"AS", "GLFT", "QH", "QH_adverse", "HJB_2023", "HJB_noread", "HJB"});
    Json out = Json::object(); out["market"] = Json::object(); out["market"]["sigma"] = mp.sigma; out["market"]["gamma"] = mp.gamma; out["market"]["psi"] = mp.psi;
    Json qs = Json::array();
    for (auto& n : names) {
        logmsg("solving " + n);
        auto qt = make_quoter(n, c, mp);
        Json e = Json::object(); e["quoter"] = qt->name(); Json rows = Json::array();
        Ladder L; L.resize(c);
        for (double q = -mp.Q; q <= mp.Q + 1e-9; q += 1.0) {
            qt->quote(q, L); Json r = Json::object(); r["q"] = q; r["hedge"] = qt->hedge(q);
            Json b = Json::array(), k = Json::array(); for (size_t i = 0; i < c.tiers.size(); ++i) { b.push(L.bid[i][0]); k.push(L.ask[i][0]); }
            r["bid"] = b; r["ask"] = k; rows.push(r);
        }
        e["rows"] = rows;
        if (auto* h = dynamic_cast<HJBQuoter*>(qt.get())) { e["band_M"] = h->band(); e["rho_per_s"] = h->rho(); e["iterations"] = h->iterations(); }
        if (auto* qh = dynamic_cast<QHQuoter*>(qt.get())) { e["a"] = qh->a(); e["band_M"] = mp.psi / std::max(qh->a(), 1e-9); Json sl = Json::array(); for (size_t i = 0; i < c.tiers.size(); ++i) sl.push(qh->skew_slope(i)); e["skew_slope"] = sl; }
        qs.push(e);
    }
    out["quoters"] = qs; out["tiers"] = c.to_json();
    write_file(a.get("out", "results/quotes.json"), out.dump(1));
    std::cout << "written " << a.get("out", "results/quotes.json") << std::endl; return 0;
}

static SimConfig simcfg_from(const Args& a, const std::string& variant, uint64_t seed) {
    SimConfig sc; sc.seed = seed; sc.horizon_s = a.num("horizon", 86400); sc.latency_ms = a.num("latency-ms", 0);
    sc.tox_threshold = a.num("tox-threshold", 0.5); sc.tox_label_pips = a.num("tox-label-pips", 0.5);
    sc.ll_hold_ms = a.num("ll-hold-ms", 0); sc.ll_tolerance_pips = a.num("ll-tolerance", 0.2);
    sc.diffusion = a.has("diffusion"); sc.diffusion_sigma = a.num("sigma", 0.35);
    apply_variant(variant, sc);
    return sc;
}

static int cmd_run(const Args& a) {
    std::unique_ptr<TickSeries> ticks; if (a.has("ticks")) ticks = std::make_unique<TickSeries>(read_ticks(a.get("ticks")));
    ClientConfig c = clients_from(a); MarketParams mp = market_from(a, ticks.get());
    std::string name = a.get("quoter", "HJB");
    auto qt = make_quoter(base_name(name), c, mp);
    SimConfig sc = simcfg_from(a, name, (uint64_t)a.num("seed", 1));
    DealerSim sim(ticks.get(), c, *qt, sc);
    if (auto* qh = dynamic_cast<QHQuoter*>(qt.get())) sim.set_reader_reference_slope(qh->a()); else { QHQuoter ref(c, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); }
    SimResult r = sim.run(); r.quoter = name;
    Json j = result_json(r, true);
    if (a.has("out")) write_file(a.get("out"), j.dump(1));
    std::cout << j.dump(1) << std::endl; return 0;
}

// Run (quoter, gamma) over days x seeds; report mean / std of daily P&L with bootstrap bands.
static int cmd_frontier(const Args& a) {
    std::vector<std::string> files = a.list("ticks", {}); std::vector<TickSeries> days; for (auto& f : files) days.push_back(read_ticks(f));
    std::vector<std::string> names = a.list("quoters", {"AS", "GLFT", "QH_adverse", "HJB_2023", "HJB", "HJB_tox", "HJB_nwz"});
    std::vector<double> gammas = a.nums("gammas", {0.005, 0.01, 0.02, 0.05, 0.1});
    int seeds = (int)a.num("seeds", 8);
    ClientConfig c = clients_from(a);
    Json out = Json::object(); Json cells = Json::array();
    for (auto& n : names) for (double g : gammas) {
        std::vector<double> pnl, sd, ir, ht, maxq;
        Json decomp = Json::object(); double sp = 0, inv = 0, adv = 0, hx = 0, hi = 0; size_t k = 0;
        for (size_t d = 0; d < std::max<size_t>(1, days.size()); ++d) {
            const TickSeries* ts = days.empty() ? nullptr : &days[d];
            MarketParams mp = market_from(a, ts); mp.gamma = g;
            auto qt = make_quoter(base_name(n), c, mp);
            for (int s = 0; s < seeds; ++s) {
                SimConfig sc = simcfg_from(a, n, (uint64_t)(1000 * d + s + 1));
                DealerSim sim(ts, c, *qt, sc); { QHQuoter ref(c, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); }
                SimResult r = sim.run();
                pnl.push_back(r.pnl_usd); sd.push_back(r.pnl_std_proxy * 100); ir.push_back(r.internalisation_ratio); ht.push_back(median(r.holding_times_s)); maxq.push_back(r.max_abs_q);
                sp += r.pnl.spread; inv += r.pnl.inventory; adv += r.pnl.adverse; hx += r.pnl.hedge_exec; hi += r.pnl.hedge_impact; ++k;
            }
        }
        CI ci = bootstrap_mean_ci(pnl);
        Json e = Json::object(); e["quoter"] = n; e["gamma"] = g; e["n"] = (long long)pnl.size();
        e["pnl_usd_mean"] = ci.mean; e["pnl_usd_lo"] = ci.lo; e["pnl_usd_hi"] = ci.hi; e["pnl_usd_std"] = stdev(pnl);
        e["pnl_1min_std_usd"] = mean(sd); e["pnl_per_unit_var"] = mean(sd) > 0 ? ci.mean / (mean(sd) * mean(sd)) : 0; e["sharpe_daily"] = stdev(pnl) > 0 ? ci.mean / stdev(pnl) : 0;
        e["internalisation_ratio"] = mean(ir); e["holding_time_s_p50"] = mean(ht); e["max_abs_q"] = mean(maxq);
        decomp["spread"] = sp / k * 100; decomp["inventory"] = inv / k * 100; decomp["adverse_selection"] = adv / k * 100; decomp["hedge_execution"] = hx / k * 100; decomp["hedge_impact"] = hi / k * 100; e["pnl_decomposition_usd"] = decomp;
        cells.push(e);
        logmsg(n + " gamma=" + std::to_string(g) + ": P&L $" + std::to_string((int)ci.mean) + " +/- " + std::to_string((int)stdev(pnl)) + ", 1-min std $" + std::to_string((int)mean(sd)));
    }
    out["cells"] = cells; out["days"] = Json(files); out["seeds"] = seeds;
    write_file(a.get("out", "results/frontier.json"), out.dump(1));
    std::cout << "written " << a.get("out", "results/frontier.json") << std::endl; return 0;
}

// Regime change: at half the horizon, informed/reader intensity and drifts jump (x mult); compare P&L before/after per quoter.
static int cmd_regime(const Args& a) {
    std::vector<std::string> files = a.list("ticks", {}); std::vector<TickSeries> days; for (auto& f : files) days.push_back(read_ticks(f));
    std::vector<std::string> names = a.list("quoters", {"AS", "GLFT", "QH_adverse", "HJB_2023", "HJB", "HJB_tox", "HJB_nwz"});
    int seeds = (int)a.num("seeds", 6); double lm = a.num("lambda-mult", 1.0), mm = a.num("mu-mult", 4.0);
    ClientConfig c = clients_from(a);
    Json out = Json::object(); Json cells = Json::array();
    for (auto& n : names) {
        std::vector<double> before, after;
        for (size_t d = 0; d < std::max<size_t>(1, days.size()); ++d) {
            const TickSeries* ts = days.empty() ? nullptr : &days[d];
            MarketParams mp = market_from(a, ts);
            auto qt = make_quoter(base_name(n), c, mp);
            for (int s = 0; s < seeds; ++s) {
                SimConfig sc = simcfg_from(a, n, (uint64_t)(1000 * d + s + 1)); sc.regime_t = sc.horizon_s / 2; sc.regime_lambda_mult = lm; sc.regime_mu_mult = mm;
                DealerSim sim(ts, c, *qt, sc); { QHQuoter ref(c, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); }
                SimResult r = sim.run();
                size_t half = r.pnl_1min.size() / 2; double b = 0, af = 0; for (size_t i = 0; i < r.pnl_1min.size(); ++i) (i < half ? b : af) += r.pnl_1min[i];
                before.push_back(b * 100); after.push_back(af * 100);
            }
        }
        CI cb = bootstrap_mean_ci(before), ca = bootstrap_mean_ci(after);
        Json e = Json::object(); e["quoter"] = n; e["pnl_before_usd"] = cb.mean; e["pnl_before_lo"] = cb.lo; e["pnl_before_hi"] = cb.hi; e["pnl_after_usd"] = ca.mean; e["pnl_after_lo"] = ca.lo; e["pnl_after_hi"] = ca.hi; e["survives"] = ca.lo > 0;
        cells.push(e); logmsg(n + ": before $" + std::to_string((int)cb.mean) + " after $" + std::to_string((int)ca.mean));
    }
    out["cells"] = cells; out["lambda_mult"] = lm; out["mu_mult"] = mm;
    write_file(a.get("out", "results/regime.json"), out.dump(1));
    std::cout << "written " << a.get("out", "results/regime.json") << std::endl; return 0;
}

// Sensitivity: latency of quote updates, and toxicity calibration (drift multiplier) for the HJB and the toxicity-routed variant.
static int cmd_sensitivity(const Args& a) {
    std::vector<std::string> files = a.list("ticks", {}); std::vector<TickSeries> days; for (auto& f : files) days.push_back(read_ticks(f));
    std::vector<std::string> names = a.list("quoters", {"HJB_2023", "HJB", "HJB_tox"});
    int seeds = (int)a.num("seeds", 6);
    ClientConfig c0 = clients_from(a);
    Json out = Json::object(); Json lat = Json::array(), tox = Json::array();
    for (double L : a.nums("latencies-ms", {0, 50, 200, 1000})) for (auto& n : names) {
        std::vector<double> pnl;
        for (size_t d = 0; d < std::max<size_t>(1, days.size()); ++d) {
            const TickSeries* ts = days.empty() ? nullptr : &days[d]; MarketParams mp = market_from(a, ts); auto qt = make_quoter(base_name(n), c0, mp);
            for (int s = 0; s < seeds; ++s) { SimConfig sc = simcfg_from(a, n, (uint64_t)(1000 * d + s + 1)); sc.latency_ms = L; DealerSim sim(ts, c0, *qt, sc); { QHQuoter ref(c0, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); } pnl.push_back(sim.run().pnl_usd); }
        }
        CI ci = bootstrap_mean_ci(pnl); Json e = Json::object(); e["quoter"] = n; e["latency_ms"] = L; e["pnl_usd_mean"] = ci.mean; e["pnl_usd_lo"] = ci.lo; e["pnl_usd_hi"] = ci.hi; lat.push(e);
        logmsg(n + " latency " + std::to_string((int)L) + " ms: $" + std::to_string((int)ci.mean));
    }
    // toxicity calibration: the quoter is solved with the baseline drifts, the world has drifts x mult
    for (double mult : a.nums("mu-mults", {0.0, 0.5, 1.0, 2.0, 4.0})) for (auto& n : names) {
        std::vector<double> pnl;
        ClientConfig cw = c0; for (auto& t : cw.tiers) t.mu *= mult;
        for (size_t d = 0; d < std::max<size_t>(1, days.size()); ++d) {
            const TickSeries* ts = days.empty() ? nullptr : &days[d]; MarketParams mp = market_from(a, ts); auto qt = make_quoter(base_name(n), c0, mp);
            for (int s = 0; s < seeds; ++s) { SimConfig sc = simcfg_from(a, n, (uint64_t)(1000 * d + s + 1)); DealerSim sim(ts, cw, *qt, sc); { QHQuoter ref(c0, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); } pnl.push_back(sim.run().pnl_usd); }
        }
        CI ci = bootstrap_mean_ci(pnl); Json e = Json::object(); e["quoter"] = n; e["mu_mult"] = mult; e["pnl_usd_mean"] = ci.mean; e["pnl_usd_lo"] = ci.lo; e["pnl_usd_hi"] = ci.hi; tox.push(e);
        logmsg(n + " mu x" + std::to_string(mult) + ": $" + std::to_string((int)ci.mean));
    }
    out["latency"] = lat; out["toxicity_calibration"] = tox;
    write_file(a.get("out", "results/sensitivity.json"), out.dump(1));
    std::cout << "written " << a.get("out", "results/sensitivity.json") << std::endl; return 0;
}

// Last look as a control (Oomen 2017; Cartea, Jaimungal & Walton 2019): hold time and slippage tolerance,
// framed by the GFXC Global Code (2025 update: no additional hold time). Reports P&L, rejection rate and
// the mark-out of accepted trades per setting.
static int cmd_lastlook(const Args& a) {
    std::vector<std::string> files = a.list("ticks", {}); std::vector<TickSeries> days; for (auto& f : files) days.push_back(read_ticks(f));
    int seeds = (int)a.num("seeds", 4); ClientConfig c = clients_from(a);
    struct Setting { double hold_ms, tol; }; std::vector<Setting> settings = {{0, 1e9}, {0, 0.5}, {0, 0.2}, {0, 0.1}, {100, 0.2}, {200, 0.2}};
    Json out = Json::array();
    for (auto& st : settings) {
        std::vector<double> pnl, rej, mo;
        for (size_t d = 0; d < std::max<size_t>(1, days.size()); ++d) {
            const TickSeries* ts = days.empty() ? nullptr : &days[d]; MarketParams mp = market_from(a, ts); HJBQuoter qt(c, mp);
            for (int s = 0; s < seeds; ++s) {
                SimConfig sc = simcfg_from(a, "HJB", (uint64_t)(1000 * d + s + 1)); sc.last_look = st.tol < 1e8; sc.ll_hold_ms = st.hold_ms; sc.ll_tolerance_pips = st.tol;
                DealerSim sim(ts, c, qt, sc); { QHQuoter ref(c, mp, false); sim.set_reader_reference_slope(std::max(ref.a(), 1e-3)); }
                SimResult r = sim.run(); pnl.push_back(r.pnl_usd); rej.push_back(r.n_trades + r.n_rejected > 0 ? (double)r.n_rejected / (r.n_trades + r.n_rejected) : 0);
                double m = 0; size_t n = 0; for (auto& t : r.trades) if (t.markouts.size() == 5) { m += t.markouts[2]; ++n; } mo.push_back(n ? m / n : 0);
            }
        }
        CI ci = bootstrap_mean_ci(pnl); Json e = Json::object(); e["hold_ms"] = st.hold_ms; e["tolerance_pips"] = st.tol < 1e8 ? st.tol : -1; e["pnl_usd_mean"] = ci.mean; e["pnl_usd_lo"] = ci.lo; e["pnl_usd_hi"] = ci.hi; e["rejection_rate"] = mean(rej); e["accepted_markout_10s_client_favour"] = mean(mo); out.push(e);
        logmsg("last look hold " + std::to_string((int)st.hold_ms) + " ms tol " + std::to_string(st.tol) + ": $" + std::to_string((int)ci.mean) + " reject " + std::to_string(mean(rej)));
    }
    write_file(a.get("out", "results/lastlook.json"), out.dump(1));
    std::cout << "written " << a.get("out", "results/lastlook.json") << std::endl; return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: fxmm <ticks|quotes|run|frontier|regime|sensitivity> [--key value ...]\n"; return 1; }
    std::string cmd = argv[1]; Args a = parse_args(argc, argv);
    try {
        if (cmd == "ticks") return cmd_ticks(a);
        if (cmd == "quotes") return cmd_quotes(a);
        if (cmd == "run") return cmd_run(a);
        if (cmd == "frontier") return cmd_frontier(a);
        if (cmd == "regime") return cmd_regime(a);
        if (cmd == "sensitivity") return cmd_sensitivity(a);
        if (cmd == "lastlook") return cmd_lastlook(a);
        std::cerr << "unknown command " << cmd << "\n"; return 1;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
