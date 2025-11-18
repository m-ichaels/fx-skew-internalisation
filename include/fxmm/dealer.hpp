#pragma once
// Event-driven dealer simulation: reference price from Dukascopy ticks (or a diffusion), client
// requests as marked point processes on the dealer's ladder, adverse-selection drift, skew readers,
// hedging at an interbank venue with temporary and permanent impact, an online toxicity classifier
// that can route a trade straight to the hedge (Cartea & Sanchez-Betancourt 2025), anticipatory
// hedging of autocorrelated flow (Nutz, Webster & Zhao 2025), last look, and a P&L decomposition
// that sums to the mark-to-market P&L by construction.
//
// Units: prices in pips, quantities in millions of base currency, P&L in pips x millions
// (1 unit = $100 for EURUSD).  Reference price S_t = tick mid + D_t, where D_t is the overlay of
// client-induced drifts and the dealer's own permanent impact.
#include "ticks.hpp"
#include "clients.hpp"
#include "quoters.hpp"
#include "toxicity.hpp"
#include <deque>
#include <functional>

namespace fxmm {

struct Trade {
    double t;            // seconds from episode start
    int tier, size_idx;
    double z;            // millions
    int side;            // +1 client buys (dealer sells), -1 client sells
    double mid, price;   // pips at trade time
    double offset;       // pips earned vs mid
    double skew_shown;   // (bid - ask)/2 of the ladder shown to this tier
    double q_before;
    std::vector<double> markouts;   // signed post-trade mid moves in the CLIENT's favour, per horizon
    double tox_score = 0;           // classifier probability at trade time
    bool externalised = false;
};

struct Hedge { double t, h, mid, price, cost; };

struct SimConfig {
    double horizon_s = 86400;
    double hedge_review_s = 2.0;
    double latency_ms = 0;          // quote staleness: clients trade on quotes computed latency_ms ago
    double latency_arb = 5.0;       // fast tiers' extra intensity per pip of stale-quote mispricing in their favour
    std::vector<double> markout_h = {0.1, 1, 10, 30, 60};
    // toxicity routing (Cartea & Sanchez-Betancourt 2025): externalise a trade whose toxicity score exceeds the threshold
    bool tox_route = false; double tox_threshold = 0.5; double tox_label_horizon = 30; double tox_label_pips = 0.5;
    // Nutz-Webster-Zhao anticipatory hedging: hedge the forecast of future client flow as if it were inventory
    bool anticipate = false; double flow_ewma_s = 300; double flow_horizon_mult = 0.5;
    // last look
    bool last_look = false; double ll_hold_ms = 0; double ll_tolerance_pips = 0.2;
    // regime change at t_switch: scale tier intensities and drifts
    double regime_t = -1; double regime_lambda_mult = 1.0; double regime_mu_mult = 1.0;
    // price source
    bool diffusion = false; double diffusion_sigma = 0.35;
    uint64_t seed = 1;
};

struct PnLDecomp {
    double spread = 0, inventory = 0, adverse = 0, hedge_exec = 0, hedge_impact = 0, total = 0;
    double check() const { return total - (spread + inventory + adverse + hedge_exec + hedge_impact); }
};

struct SimResult {
    std::string quoter;
    PnLDecomp pnl;
    double pnl_usd = 0, pnl_std_proxy = 0;   // total P&L in USD; std of 1-minute P&L increments (for frontiers)
    std::vector<double> pnl_1min;             // 1-minute P&L path (pips x M)
    double mean_abs_q = 0, max_abs_q = 0, q_std = 0;
    size_t n_trades = 0, n_hedges = 0, n_rejected = 0, n_externalised = 0;
    double hedged_volume = 0, client_volume = 0, internalisation_ratio = 0;
    std::vector<double> holding_times_s;      // Butz-Oomen: time from a risk position opening to being flat/hedged
    std::vector<Trade> trades;
    std::vector<Hedge> hedges;
    std::vector<double> q_path_1min;
    Json tier_markouts;                       // Oomen price signatures per tier
    Json classifier;                          // toxicity classifier diagnostics
};

class DealerSim {
public:
    DealerSim(const TickSeries* ticks, const ClientConfig& clients, const Quoter& quoter, SimConfig cfg)
        : ticks_(ticks), c_(clients), quoter_(quoter), cfg_(cfg), rng_(cfg.seed), clf_((int)clients.tiers.size()) {}

    SimResult run() {
        SimResult R; R.quoter = quoter_.name();
        const double T = cfg_.horizon_s;
        double t = 0, S_tick = ticks_ ? ticks_->ticks[0].mid() : 10000.0, D = 0;   // D: overlay (drift + own impact)
        double S = S_tick + D;
        double q = 0, cash = 0;
        Ladder L, L_stale; L.resize(c_); L_stale.resize(c_);
        double t_ladder = -1e9;                       // when the stale ladder was computed
        std::vector<ClientCfgScaled> tiers = scaled_tiers(1.0, 1.0);
        struct Drift { double t_end, rate; };
        std::vector<Drift> drifts;                    // active drift impulses (pips/s until t_end)
        double next_review = cfg_.hedge_review_s, next_minute = 60;
        double flow_ewma = 0, last_flow_t = 0;        // signed client flow forecast (millions)
        double flow_dir = rng_.uniform() < 0.5 ? -1.0 : 1.0, next_switch = rng_.exponential(std::max(c_.flow_switch, 1e-9));   // hidden flow direction
        size_t tick_i = 0;
        double last_pnl = 0, risk_open_t = -1;
        size_t n_markout_pending = 0;
        auto mtm = [&](double Sx) { return cash + q * Sx; };
        // advance the reference price to time t2: tick mid at t2 plus the overlay integrated over [t, t2]
        auto advance = [&](double t2) {
            double dt = t2 - t; if (dt <= 0) { return; }
            double S_prev = S;
            double dD = 0;
            for (auto& d : drifts) { double a = std::max(0.0, std::min(dt, d.t_end - t)); dD += d.rate * a; }
            drifts.erase(std::remove_if(drifts.begin(), drifts.end(), [&](const Drift& d) { return d.t_end <= t2; }), drifts.end());
            double S_tick2;
            if (cfg_.diffusion || !ticks_) S_tick2 = S_tick + cfg_.diffusion_sigma * std::sqrt(dt) * rng_.normal();
            else { Ts ms = ticks_->t0() + (Ts)(t2 * 1000); while (tick_i + 1 < ticks_->ticks.size() && ticks_->ticks[tick_i + 1].ts_ms <= ms) ++tick_i; S_tick2 = ticks_->ticks[tick_i].mid(); }
            double dS_exo = S_tick2 - S_tick;
            R.pnl.inventory += q * dS_exo; R.pnl.adverse += q * dD;
            S_tick = S_tick2; D += dD; S = S_tick + D; t = t2;
            (void)S_prev;
        };
        auto refresh_quotes = [&]() {
            double q_eff = q;
            if (cfg_.anticipate) { double decay = std::exp(-(t - last_flow_t) / cfg_.flow_ewma_s); q_eff = q - cfg_.flow_horizon_mult * flow_ewma * decay; }
            quoter_.quote(q_eff, L);
            if (t - t_ladder >= cfg_.latency_ms / 1000.0) { L_stale = L; t_ladder = t; S_at_ladder_ = S; }
        };
        auto do_hedge = [&](double h, const char* why) {
            if (std::fabs(h) < 1e-9) return;
            double half = quoter_.mp.psi, temp = quoter_.mp.eta * std::fabs(h);
            double px = S + (h > 0 ? 1 : -1) * (half + temp);
            cash -= h * px; q += h;
            double cost = std::fabs(h) * (half + temp);
            R.pnl.hedge_exec -= cost;
            // permanent impact: our own hedge moves the reference price against the remaining inventory
            double imp = quoter_.mp.kappa * h;
            R.pnl.hedge_impact += q * imp; D += imp; S += imp;
            R.hedges.push_back({t, h, S - imp, px, cost}); R.hedged_volume += std::fabs(h);
            (void)why;
        };
        // total client intensity for the current ladder (stale ladder if latency)
        auto intensities = [&](std::vector<double>& out) {
            out.clear(); double tot = 0;
            for (size_t i = 0; i < tiers.size(); ++i) {
                const Tier& tr = tiers[i].t;
                double mag = tr.reader ? std::fabs(0.5 * (L_stale.bid[i][0] - L_stale.ask[i][0])) / s_ref_ : 0;   // |q| read from the shown skew
                for (size_t j = 0; j < tr.sizes.size(); ++j) {
                    double lam = tr.lambda * tr.size_w[j];
                    double rr = tr.rho * mag / c_.Q;   // readers trade against the dealer's position and avoid helping it
                    double wsell = q > 0 ? 1 + rr : std::max(0.0, 1 - rr), wbuy = q < 0 ? 1 + rr : std::max(0.0, 1 - rr);
                    if (std::fabs(q) < 1e-9) { wsell = 1; wbuy = 1; }
                    // latency arbitrage: fast tiers (informed, readers) hit a stale quote in proportion to its mispricing in their favour
                    double edge_sell = 0, edge_buy = 0;
                    if (cfg_.latency_ms > 0 && (tr.mu > 0 || tr.reader)) { double stale = S - S_at_ladder_; edge_sell = std::max(0.0, -stale) * cfg_.latency_arb; edge_buy = std::max(0.0, stale) * cfg_.latency_arb; }
                    double lb = lam * tr.f(L_stale.bid[i][j] - std::min(L_stale.bid[i][j], edge_sell / std::max(cfg_.latency_arb, 1e-9))) * wsell * (1 - c_.flow_phi * flow_dir) * (1 + edge_sell);   // client sells to us at the (stale) bid
                    double la = lam * tr.f(L_stale.ask[i][j] - std::min(L_stale.ask[i][j], edge_buy / std::max(cfg_.latency_arb, 1e-9))) * wbuy * (1 + c_.flow_phi * flow_dir) * (1 + edge_buy);    // client buys from us at the (stale) ask
                    out.push_back(lb); out.push_back(la); tot += lb + la;
                }
            }
            return tot;
        };
        std::vector<double> lam;
        std::vector<std::pair<size_t, size_t>> slots; for (size_t i = 0; i < tiers.size(); ++i) for (size_t j = 0; j < tiers[i].t.sizes.size(); ++j) slots.emplace_back(i, j);
        refresh_quotes();
        while (t < T) {
            if (cfg_.regime_t > 0 && t >= cfg_.regime_t && !regime_applied_) { tiers = scaled_tiers(cfg_.regime_lambda_mult, cfg_.regime_mu_mult); regime_applied_ = true; }
            double tot = intensities(lam);
            double tau = tot > 0 ? rng_.exponential(tot) : 1e9;
            double t_next = std::min({t + tau, next_review, next_minute, next_switch, T});
            advance(t_next);
            if (t_next == next_switch) { flow_dir = -flow_dir; next_switch = t + rng_.exponential(std::max(c_.flow_switch, 1e-9)); continue; }
            process_markouts(R, t, S);
            if (t_next == T) break;
            if (t_next == next_minute) {
                double p = mtm(S); R.pnl_1min.push_back(p - last_pnl); last_pnl = p; R.q_path_1min.push_back(q); next_minute += 60; refresh_quotes(); continue;
            }
            if (t_next == next_review) {
                next_review += cfg_.hedge_review_s;
                double q_eff = q;
                if (cfg_.anticipate) { double decay = std::exp(-(t - last_flow_t) / cfg_.flow_ewma_s); q_eff = q - cfg_.flow_horizon_mult * flow_ewma * decay; }
                double h = quoter_.hedge(q_eff);
                if (h != 0) { do_hedge(h, "review"); R.n_hedges++; }
                if (std::fabs(q) < 1e-9 && risk_open_t >= 0) { R.holding_times_s.push_back(t - risk_open_t); risk_open_t = -1; }
                refresh_quotes(); continue;
            }
            // ---- client request ----------------------------------------------------------------------
            size_t k = rng_.weighted(lam);
            size_t i = slots[k / 2].first, j = slots[k / 2].second;
            const Tier& tr = tiers[i].t;
            int side = (k % 2 == 0) ? -1 : +1;   // even: client sells to us (bid); odd: client buys (ask)
            double z = tr.sizes[j];
            double off = side < 0 ? L_stale.bid[i][j] : L_stale.ask[i][j];
            double S_quote = S;                   // quotes are stale by latency: the price was fixed at t_ladder
            if (cfg_.latency_ms > 0) S_quote = S_at_ladder_;
            double price = side < 0 ? S_quote - off : S_quote + off;
            // last look: hold, then reject if the mid moved against us beyond the tolerance
            if (cfg_.last_look) {
                double S_before = S; if (cfg_.ll_hold_ms > 0) { advance(t + cfg_.ll_hold_ms / 1000.0); }
                double adverse_move = side < 0 ? (S_before - S) : (S - S_before);   // client sells and price fell => we are worse off
                if (adverse_move > cfg_.ll_tolerance_pips) { R.n_rejected++; refresh_quotes(); continue; }
            }
            Trade tdd; tdd.t = t; tdd.tier = (int)i; tdd.size_idx = (int)j; tdd.z = z; tdd.side = side; tdd.mid = S; tdd.price = price; tdd.q_before = q;
            tdd.offset = side < 0 ? (S - price) : (price - S);
            tdd.skew_shown = 0.5 * (L_stale.bid[i][0] - L_stale.ask[i][0]);
            // toxicity score before the trade (features: tier, size, shown skew, EWMA of this tier's past mark-outs)
            std::vector<double> x = clf_.features((int)i, (int)tiers.size(), z, tdd.skew_shown * side, side);
            tdd.tox_score = clf_.score(x);
            // execute
            cash += side * z * price; q -= side * z;                          // client buys => we sell
            R.pnl.spread += z * tdd.offset;
            R.client_volume += z; R.n_trades++;
            if (risk_open_t < 0 && std::fabs(q) > 1e-9) risk_open_t = t;
            // adverse-selection drift in the client's direction
            if (tr.mu > 0) drifts.push_back({t + tr.tau, side * tr.mu * z / tr.tau});
            // flow forecast
            { double decay = std::exp(-(t - last_flow_t) / cfg_.flow_ewma_s); flow_ewma = flow_ewma * decay + side * z; last_flow_t = t; }
            // routing: externalise toxic flow immediately
            if (cfg_.tox_route && tdd.tox_score > cfg_.tox_threshold) { do_hedge(side * z, "externalise"); tdd.externalised = true; R.n_externalised++; }
            pending_.push_back({R.trades.size(), x});
            R.trades.push_back(tdd); ++n_markout_pending;
            refresh_quotes();
        }
        // finish
        R.pnl.total = mtm(S) - 0.0;
        R.pnl_usd = R.pnl.total * 100.0;
        R.pnl_std_proxy = stdev(R.pnl_1min);
        { double s = 0, s2 = 0; for (double x : R.q_path_1min) { s += std::fabs(x); s2 += x * x; R.max_abs_q = std::max(R.max_abs_q, std::fabs(x)); } size_t n = std::max<size_t>(1, R.q_path_1min.size()); R.mean_abs_q = s / n; R.q_std = std::sqrt(s2 / n); }
        R.internalisation_ratio = R.client_volume > 0 ? 1.0 - R.hedged_volume / R.client_volume : 0;
        R.tier_markouts = markout_table(R);
        R.classifier = clf_.diagnostics();
        return R;
    }

private:
    struct ClientCfgScaled { Tier t; };
    std::vector<ClientCfgScaled> scaled_tiers(double lm, double mm) {
        std::vector<ClientCfgScaled> v; for (auto& t : c_.tiers) { ClientCfgScaled s{t}; s.t.lambda *= lm; s.t.mu *= mm; v.push_back(s); } return v;
    }
    // mark-outs are filled in once the horizon has passed; the classifier is trained on the labelled trade
    void process_markouts(SimResult& R, double t, double S) {
        for (size_t k = markout_cursor_; k < R.trades.size(); ++k) {
            Trade& tr = R.trades[k];
            size_t h = tr.markouts.size();
            while (h < cfg_.markout_h.size() && t >= tr.t + cfg_.markout_h[h]) { tr.markouts.push_back(tr.side * (S - tr.mid)); ++h; }
            if (h == cfg_.markout_h.size()) {
                // label: toxic if at the label horizon the client is ahead net of the spread paid by more than tox_label_pips
                // (the trade lost the dealer money: the Cartea & Sanchez-Betancourt notion of toxic flow)
                size_t hi = 0; for (size_t m = 0; m < cfg_.markout_h.size(); ++m) if (cfg_.markout_h[m] <= cfg_.tox_label_horizon) hi = m;
                double y = tr.markouts[hi] - tr.offset > cfg_.tox_label_pips ? 1.0 : 0.0;
                for (auto& p : pending_) if (p.first == k) { clf_.update(p.second, y); }
                pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const std::pair<size_t, std::vector<double>>& p) { return p.first == k; }), pending_.end());
                clf_.observe_markout(tr.tier, tr.markouts[hi]);
                if (k == markout_cursor_) ++markout_cursor_;
            } else break;
        }
    }
    Json markout_table(const SimResult& R) const {
        Json a = Json::array();
        for (size_t i = 0; i < c_.tiers.size(); ++i) {
            Json e = Json::object(); e["tier"] = c_.tiers[i].name; size_t n = 0;
            std::vector<double> m(cfg_.markout_h.size(), 0.0), off;
            for (auto& tr : R.trades) if (tr.tier == (int)i && tr.markouts.size() == cfg_.markout_h.size()) { ++n; for (size_t h = 0; h < m.size(); ++h) m[h] += tr.markouts[h]; off.push_back(tr.offset); }
            Json mk = Json::array(); for (size_t h = 0; h < m.size(); ++h) { Json x = Json::object(); x["horizon_s"] = cfg_.markout_h[h]; x["markout_pips_client_favour"] = n ? m[h] / n : 0; mk.push(x); }
            e["n"] = (long long)n; e["markouts"] = mk; e["mean_offset_pips"] = mean(off); a.push(e);
        }
        return a;
    }
    const TickSeries* ticks_; const ClientConfig& c_; const Quoter& quoter_; SimConfig cfg_; Rng rng_;
    ToxicityClassifier clf_;
    std::vector<std::pair<size_t, std::vector<double>>> pending_;
    size_t markout_cursor_ = 0; bool regime_applied_ = false;
    double s_ref_ = 0.05, S_at_ladder_ = 0;
public:
    void set_reader_reference_slope(double s) { s_ref_ = s; }
};

} // namespace fxmm
