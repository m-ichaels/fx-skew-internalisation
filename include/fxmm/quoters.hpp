#pragma once
// Quoting policies.  All map the dealer's inventory q (millions) to a ladder of bid/ask offsets
// (pips) per tier and size bucket, and to a hedge quantity at review times.
//
//  AS     Avellaneda & Stoikov (2008): reservation price S - q gamma sigma^2 H, spread
//         gamma sigma^2 H + (2/gamma) ln(1 + gamma/k); one quote for every client.
//  GLFT   Gueant, Lehalle & Fernandez-Tapia (2013) closed form (CARA, A e^{-k delta}, unit size z):
//         delta_b(q) = (1/gamma) ln(1+gamma/k) + (2q+1)/2 * sqrt( sigma^2 gamma/(2 k A) (1+gamma/k)^{1+k/gamma} )
//         delta_a(q) = (1/gamma) ln(1+gamma/k) - (2q-1)/2 * sqrt( ... ),  q counted in units of z.
//  QH     Quadratic-Hamiltonian closed form of the ergodic dealer problem with tiers, sizes and
//         adverse selection (Barzykin-Bergault-Gueant-Lemmel 2025 approach; derivation in README):
//         with theta(q) = -a q^2 / 2 and H_i(p) ~ (A_i/k_i) e^{-1} (1 - k_i p + k_i^2 p^2/2),
//         sum_i z_i A_i k_i e^{-1} (a + mu_i)^2 = gamma sigma^2 / 2   (solve for a)
//         delta^{b/a}_{i,z}(q) = 1/k_i + (a + 2 mu_i) z/2  +/-  (a + mu_i) q
//         i.e. adverse selection widens every quote by mu_i z and adds mu_i to the skew slope.
//         Hedging band: hedge when |a q| > psi (linear execution cost psi per million).
//  HJB    Numerical ergodic control (relative value iteration on an inventory grid) with the
//         logistic ladder, running penalty gamma sigma^2 q^2 / 2, hedging cost psi|h| + eta h^2,
//         permanent impact kappa, tier drifts, and readers whose intensity depends on the skew
//         shown to them.  This is the Barzykin-Bergault-Gueant (2023) dealer with the 2025
//         adverse-selection / price-reading extension; the skew withdrawal for readers emerges here.
#include "clients.hpp"
#include <memory>
#include <algorithm>

namespace fxmm {

struct MarketParams {
    double sigma = 0.35;      // pips / sqrt(s)
    double gamma = 0.02;      // risk aversion (running penalty gamma sigma^2 q^2 / 2, pips per million^2 per s)
    double psi = 0.25;        // interbank half spread + fee, pips per million hedged
    double eta = 0.02;        // temporary impact, pips per million^2
    double kappa = 0.01;      // permanent impact, pips per million
    double Q = 30;            // inventory grid half-width, millions
    double dq = 0.5;          // grid step, millions (must divide every size bucket)
    double dt = 2.0;          // HJB time step, s (also the hedge review interval)
    double hedge_max = 10;    // max hedge per review, millions
    double delta_max = 3.0;   // quote grid, pips
    int delta_steps = 30;
    double AS_horizon = 60;   // AS effective horizon, s (time to flatten; AS has no hedging)
};

class Quoter {
public:
    virtual ~Quoter() = default;
    virtual std::string name() const = 0;
    virtual void quote(double q, Ladder& L) const = 0;          // fills L.bid/L.ask (pips >= 0)
    virtual double hedge(double q) const = 0;                    // millions to buy (>0) / sell (<0) at a review
    const ClientConfig* clients = nullptr; MarketParams mp;
protected:
    static double clamp0(double x) { return x < 0 ? 0 : x; }
};

// ---- Avellaneda-Stoikov -----------------------------------------------------------------------------
class ASQuoter : public Quoter {
public:
    ASQuoter(const ClientConfig& c, MarketParams p) { clients = &c; mp = p; double ksum = 0, lsum = 0; for (auto& t : c.tiers) { double A, k; t.exp_fit(A, k); ksum += k * t.lambda; lsum += t.lambda; } k_ = ksum / std::max(1e-12, lsum); }
    std::string name() const override { return "AS"; }
    void quote(double q, Ladder& L) const override {
        double g = mp.gamma, s2 = mp.sigma * mp.sigma, H = mp.AS_horizon;
        double half = 0.5 * (g * s2 * H + (2.0 / g) * std::log(1 + g / k_));
        double r = -q * g * s2 * H;                      // reservation price offset from mid
        for (auto& row : L.bid) for (auto& x : row) x = clamp0(half - r);   // bid = mid + r - half  =>  offset below mid
        for (auto& row : L.ask) for (auto& x : row) x = clamp0(half + r);
    }
    double hedge(double) const override { return 0; }   // AS has no hedging
private:
    double k_ = 5;
};

// ---- GLFT closed form --------------------------------------------------------------------------------
class GLFTQuoter : public Quoter {
public:
    GLFTQuoter(const ClientConfig& c, MarketParams p) { clients = &c; mp = p; A_.resize(c.tiers.size()); k_.resize(c.tiers.size()); for (size_t i = 0; i < c.tiers.size(); ++i) { c.tiers[i].exp_fit(A_[i], k_[i]); A_[i] *= c.tiers[i].lambda; } }
    std::string name() const override { return "GLFT"; }
    void quote(double q, Ladder& L) const override {
        double g = mp.gamma, s2 = mp.sigma * mp.sigma;
        for (size_t i = 0; i < clients->tiers.size(); ++i) for (size_t j = 0; j < clients->tiers[i].sizes.size(); ++j) {
            double z = clients->tiers[i].sizes[j], A = A_[i] / z, k = k_[i];
            double qz = q / z;   // inventory in units of the trade size
            double base = (1.0 / g) * std::log(1 + g / k);
            double sq = std::sqrt(s2 * g / (2 * k * A) * std::pow(1 + g / k, 1 + k / g));
            L.bid[i][j] = clamp0(base + (2 * qz + 1) / 2 * sq);
            L.ask[i][j] = clamp0(base - (2 * qz - 1) / 2 * sq);
        }
    }
    double hedge(double) const override { return 0; }
private:
    std::vector<double> A_, k_;
};

// ---- quadratic-Hamiltonian closed form (with adverse selection) ------------------------------------
class QHQuoter : public Quoter {
public:
    QHQuoter(const ClientConfig& c, MarketParams p, bool with_adverse = true) : adverse_(with_adverse) {
        clients = &c; mp = p;
        size_t n = c.tiers.size(); A_.resize(n); k_.resize(n); zbar_.resize(n); mu_.resize(n);
        for (size_t i = 0; i < n; ++i) { c.tiers[i].exp_fit(A_[i], k_[i]); A_[i] *= c.tiers[i].lambda; double zb = 0; for (size_t j = 0; j < c.tiers[i].sizes.size(); ++j) zb += c.tiers[i].sizes[j] * c.tiers[i].size_w[j]; zbar_[i] = zb; mu_[i] = with_adverse ? c.tiers[i].mu : 0.0; }
        // solve sum_i z_i A_i k_i e^{-1} (a + mu_i)^2 = gamma sigma^2 / 2 for a >= 0 (bisection)
        double target = p.gamma * p.sigma * p.sigma / 2;
        auto lhs = [&](double a) { double s = 0; for (size_t i = 0; i < n; ++i) s += zbar_[i] * A_[i] * k_[i] * std::exp(-1.0) * (a + mu_[i]) * (a + mu_[i]); return s; };
        double lo = 0, hi = 1; while (lhs(hi) < target) hi *= 2;
        for (int it = 0; it < 200; ++it) { double m = 0.5 * (lo + hi); (lhs(m) < target ? lo : hi) = m; }
        a_ = 0.5 * (lo + hi);
    }
    std::string name() const override { return adverse_ ? "QH_adverse" : "QH"; }
    double a() const { return a_; }
    double skew_slope(size_t i) const { return a_ + mu_[i]; }
    double spread0(size_t i, double z) const { return 1.0 / k_[i] + (a_ + 2 * mu_[i]) * z / 2; }
    void quote(double q, Ladder& L) const override {
        for (size_t i = 0; i < clients->tiers.size(); ++i) for (size_t j = 0; j < clients->tiers[i].sizes.size(); ++j) {
            double z = clients->tiers[i].sizes[j];
            L.bid[i][j] = clamp0(spread0(i, z) + skew_slope(i) * q);
            L.ask[i][j] = clamp0(spread0(i, z) - skew_slope(i) * q);
        }
    }
    // hedge toward the band edge: |theta'(q)| = a|q| > psi
    double hedge(double q) const override {
        double band = mp.psi / std::max(a_, 1e-9);
        if (std::fabs(q) <= band) return 0;
        double target = q > 0 ? band : -band;
        return std::max(-mp.hedge_max, std::min(mp.hedge_max, target - q));
    }
private:
    bool adverse_;
    std::vector<double> A_, k_, zbar_, mu_;
    double a_ = 0;
};

// ---- numerical HJB (relative value iteration) --------------------------------------------------------
struct HJBOptions {
    bool adverse = true;        // tier drifts in the value function
    bool readers = true;        // readers' intensity depends on the skew shown to them
    bool hedging = true;
    int max_iter = 4000;
    double tol = 1e-7;
};

class HJBQuoter : public Quoter {
public:
    HJBQuoter(const ClientConfig& c, MarketParams p, HJBOptions o = {}) : opt_(o) { clients = &c; mp = p; solve(); }
    std::string name() const override { return std::string("HJB") + (opt_.adverse ? "" : "_noadv") + (opt_.readers ? "" : "_noread") + (opt_.hedging ? "" : "_nohedge"); }
    void quote(double q, Ladder& L) const override { int n = idx(q); for (size_t i = 0; i < clients->tiers.size(); ++i) for (size_t j = 0; j < clients->tiers[i].sizes.size(); ++j) { L.bid[i][j] = pol_bid_[n][i][j]; L.ask[i][j] = pol_ask_[n][i][j]; } }
    double hedge(double q) const override { return opt_.hedging ? pol_h_[idx(q)] : 0; }
    double value(double q) const { return V_[idx(q)]; }
    double rho() const { return rho_; }
    int iterations() const { return iters_; }
    // band = largest |q| with zero hedge
    double band() const { double b = 0; for (int n = 0; n < N_; ++n) if (pol_h_[n] == 0) b = std::max(b, std::fabs(qgrid_[n])); return b; }

private:
    int idx(double q) const { int n = (int)std::llround((q + mp.Q) / mp.dq); return std::max(0, std::min(N_ - 1, n)); }
    void solve() {
        const ClientConfig& c = *clients;
        N_ = (int)std::llround(2 * mp.Q / mp.dq) + 1; qgrid_.resize(N_); for (int n = 0; n < N_; ++n) qgrid_[n] = -mp.Q + n * mp.dq;
        size_t T = c.tiers.size();
        V_.assign(N_, 0.0);
        pol_bid_.assign(N_, std::vector<std::vector<double>>(T)); pol_ask_ = pol_bid_; pol_h_.assign(N_, 0.0);
        std::vector<std::vector<std::vector<double>>> cand_bid = pol_bid_, cand_ask = pol_ask_;
        for (int n = 0; n < N_; ++n) for (size_t i = 0; i < T; ++i) { pol_bid_[n][i].assign(c.tiers[i].sizes.size(), 0.5); pol_ask_[n][i].assign(c.tiers[i].sizes.size(), 0.5); cand_bid[n][i] = pol_bid_[n][i]; cand_ask[n][i] = pol_ask_[n][i]; }
        std::vector<double> dgrid; for (int st = 0; st <= mp.delta_steps; ++st) dgrid.push_back(mp.delta_max * st / mp.delta_steps);
        int hsteps = (int)std::llround(mp.hedge_max / mp.dq);
        double dt = mp.dt, s2 = mp.sigma * mp.sigma;
        // reference skew slope readers use to infer q (the closed-form slope)
        double s_ref = 0.05; { QHQuoter qh(c, mp, false); s_ref = std::max(qh.a(), 1e-3); }   // readers' reference skew slope: the base closed form
        // cap so that no per-step transition probability can exceed 1/2 (discrete-time validity)
        double w_cap = 0; for (auto& t : c.tiers) w_cap = std::max(w_cap, t.lambda * dt); w_cap = std::max(1.0, 0.5 / std::max(w_cap, 1e-9));
        std::vector<double> trade_term(N_), Vn(N_);
        for (iters_ = 0; iters_ < opt_.max_iter; ++iters_) {
            // stage 1: optimal quotes and expected trade gain at every post-hedge inventory q
            for (int n = 0; n < N_; ++n) {
                double q = qgrid_[n]; double acc = 0;
                for (size_t i = 0; i < T; ++i) {
                    const Tier& t = c.tiers[i];
                    for (size_t j = 0; j < t.sizes.size(); ++j) {
                        double z = t.sizes[j], lam = t.lambda * t.size_w[j], mu = opt_.adverse ? t.mu : 0.0;
                        bool okb = q + z <= mp.Q + 1e-9, oka = q - z >= -mp.Q - 1e-9;
                        // continuation of "dealer buys z at the bid" / "dealer sells z at the ask", relative to V(q), incl. expected drift P&L
                        double cb = okb ? V_[idx(q + z)] - V_[n] - (q + z) * mu * z : 0;
                        double ca = oka ? V_[idx(q - z)] - V_[n] + (q - z) * mu * z : 0;
                        double bb = mp.delta_max, ba = mp.delta_max, bv = -1e300;   // at the grid boundary a side is refused (widest offset)
                        if (t.reader && opt_.readers) {
                            // Readers know which way the dealer is positioned and read the magnitude from the skew shown
                            // to them; they then trade more on the side that worsens the dealer's inventory and less on
                            // the side that would help it.  The dealer cannot fool them - only show them less skew.
                            for (double db : dgrid) for (double da : dgrid) {
                                double mag = std::fabs(0.5 * (db - da)) / s_ref;   // inferred |q|
                                // harmful side (adds to the position) scaled up, helpful side scaled down, both by the read magnitude
                                double r = t.rho * mag / c.Q;
                                double wsell = q > 0 ? std::min(w_cap, 1 + r) : std::max(0.0, 1 - r), wbuy = q < 0 ? std::min(w_cap, 1 + r) : std::max(0.0, 1 - r);
                                if (q == 0) { wsell = 1; wbuy = 1; }
                                double v = (okb ? lam * t.f(db) * wsell * dt * (z * db + cb) : 0) + (oka ? lam * t.f(da) * wbuy * dt * (z * da + ca) : 0);
                                if (v > bv) { bv = v; bb = db; ba = da; }
                            }
                            if (!okb) bb = mp.delta_max; if (!oka) ba = mp.delta_max;   // refused side at the grid boundary
                        } else {
                            double vb = -1e300, va = -1e300;
                            for (double d : dgrid) { double x = lam * t.f(d) * dt; if (okb) { double v = x * (z * d + cb); if (v > vb) { vb = v; bb = d; } } if (oka) { double v = x * (z * d + ca); if (v > va) { va = v; ba = d; } } }
                            bv = (okb ? vb : 0) + (oka ? va : 0);
                        }
                        cand_bid[n][i][j] = bb; cand_ask[n][i][j] = ba; acc += bv;
                    }
                }
                trade_term[n] = acc;
            }
            // stage 2: hedge choice at the review
            for (int n = 0; n < N_; ++n) {
                double q0 = qgrid_[n], best = -1e300, best_h = 0;
                for (int hs = (opt_.hedging ? -hsteps : 0); hs <= (opt_.hedging ? hsteps : 0); ++hs) {
                    double h = hs * mp.dq, q = q0 + h;
                    if (std::fabs(q) > mp.Q + 1e-9) continue;
                    int m = idx(q);
                    double val = -(mp.psi * std::fabs(h) + mp.eta * h * h) + q * mp.kappa * h - 0.5 * mp.gamma * s2 * q * q * dt + trade_term[m] + V_[m];
                    if (val > best) { best = val; best_h = h; }
                }
                Vn[n] = best; pol_h_[n] = best_h;
            }
            double ref = Vn[idx(0)], diff = 0;
            for (int n = 0; n < N_; ++n) { double v = Vn[n] - ref; diff = std::max(diff, std::fabs(v - V_[n])); V_[n] = v; }
            rho_ = ref / dt;
            if (diff < opt_.tol) break;
        }
        // quotes are those of the post-hedge inventory
        for (int n = 0; n < N_; ++n) { int m = idx(qgrid_[n] + pol_h_[n]); pol_bid_[n] = cand_bid[m]; pol_ask_[n] = cand_ask[m]; }
    }
    HJBOptions opt_;
    int N_ = 0, iters_ = 0; double rho_ = 0;
    std::vector<double> qgrid_, V_, pol_h_;
    std::vector<std::vector<std::vector<double>>> pol_bid_, pol_ask_;
};

} // namespace fxmm
