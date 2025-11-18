#pragma once
// Client layer: tiers, size buckets (the pricing ladder), spread-responsive arrival intensities,
// post-trade drift (adverse selection) and skew readers.  Everything here is a calibrated
// simulation - real single-dealer client flow does not exist publicly (see README, Data).
//
// Arrival of a (tier i, size z, side) request when the dealer shows offset delta (pips from the
// reference mid) is a Poisson process with intensity
//     Lambda_{i,z}(delta) = lambda_{i,z} * f_i(delta),  f_i(delta) = 1 / (1 + exp(alpha_i + beta_i delta))
// (the logistic ladder of Barzykin, Bergault & Gueant 2021/2023), or A e^{-k delta} for the
// closed-form quoters.  A trade of size z by tier i moves the reference price by mu_i * z pips
// in the client's direction over tau_i seconds (Oomen-style price signature: informed flow).
// Readers (tier 4) additionally observe the skew the dealer shows them, infer the inventory and
// trade against it with intensity scaled by (1 + rho * |q_hat| / Q).
#include "types.hpp"
#include "json.hpp"
#include "stats.hpp"
#include <vector>
#include <string>
#include <cmath>

namespace fxmm {

struct Tier {
    std::string name;
    double lambda = 0.01;           // requests per second per side, summed over sizes
    std::vector<double> sizes;      // ladder buckets, millions
    std::vector<double> size_w;     // probability of each bucket
    double alpha = -2.0, beta = 6.0;// logistic response: f(delta) = 1/(1+exp(alpha + beta delta)), delta in pips
    double mu = 0.0;                // adverse-selection drift, pips per million, in the client's direction
    double tau = 30.0;              // seconds over which the drift is realised
    bool reader = false;            // reads the skew and trades against the inferred inventory
    double rho = 0.0;               // reader sensitivity
    double f(double delta) const { return 1.0 / (1.0 + std::exp(alpha + beta * delta)); }
    // exponential fit A e^{-k delta} of the logistic on [0, 2] pips (least squares in log space), for the closed-form quoters
    void exp_fit(double& A, double& k) const {
        std::vector<double> x, y; for (int i = 0; i <= 40; ++i) { double d = 0.05 * i; x.push_back(d); y.push_back(std::log(std::max(f(d), 1e-12))); }
        Ols o = ols(x, y); k = -o.b; A = std::exp(o.a);
    }
};

struct ClientConfig {
    std::vector<Tier> tiers;
    double Q = 20;                  // inventory scale used by readers, millions
    // autocorrelated aggregate flow (Nutz, Webster & Zhao 2025 setting): a hidden direction m in {-1,+1}
    // switching at rate flow_switch (per s) tilts every client's buy probability to 1/2 + phi m / 2
    double flow_phi = 0.2, flow_switch = 1.0 / 600;
    static ClientConfig defaults() {
        ClientConfig c;
        // Assumptions (README, Data): shares of turnover by client type after the BIS Triennial Survey;
        // spread sensitivities by tier after the FX dealer papers; drifts after Oomen (2019) / LMAX
        // toxic-flow mark-outs; all stated as calibration targets, not measurements.
        c.tiers = {
            {"informed",  0.020, {1, 5, 10}, {0.4, 0.4, 0.2}, -1.0, 6.0, 0.12, 30.0, false, 0.0},
            {"corporate", 0.030, {1, 3, 10}, {0.5, 0.4, 0.1}, -1.5, 4.0, 0.02, 60.0, false, 0.0},
            {"retail",    0.060, {0.5, 1, 2}, {0.5, 0.35, 0.15}, -2.0, 3.0, 0.00, 30.0, false, 0.0},
            {"reader",    0.012, {1, 5}, {0.6, 0.4}, -1.0, 6.0, 0.06, 30.0, true, 3.0}};
        return c;
    }
    Json to_json() const {
        Json j = Json::object(); j["Q"] = Q; j["flow_phi"] = flow_phi; j["flow_switch"] = flow_switch; Json a = Json::array();
        for (auto& t : tiers) { Json x = Json::object(); x["name"] = t.name; x["lambda"] = t.lambda; x["sizes"] = Json(t.sizes); x["size_w"] = Json(t.size_w); x["alpha"] = t.alpha; x["beta"] = t.beta; x["mu"] = t.mu; x["tau"] = t.tau; x["reader"] = t.reader; x["rho"] = t.rho; a.push(x); }
        j["tiers"] = a; return j;
    }
    static ClientConfig from_json(const Json& j) {
        ClientConfig c; c.Q = j.get("Q", 20.0); c.flow_phi = j.get("flow_phi", 0.2); c.flow_switch = j.get("flow_switch", 1.0 / 600);
        for (auto& x : j["tiers"].arr()) { Tier t; t.name = x.get("name", std::string("tier")); t.lambda = x.get("lambda", 0.01); for (auto& v : x["sizes"].arr()) t.sizes.push_back(v.num()); for (auto& v : x["size_w"].arr()) t.size_w.push_back(v.num()); t.alpha = x.get("alpha", -2.0); t.beta = x.get("beta", 6.0); t.mu = x.get("mu", 0.0); t.tau = x.get("tau", 30.0); t.reader = x.get("reader", false); t.rho = x.get("rho", 0.0); c.tiers.push_back(t); }
        return c;
    }
    static ClientConfig load(const std::string& p) { return from_json(Json::parse(read_file(p))); }
};

// A quote ladder: bid/ask offsets (pips, >= 0 means inside our favour) per tier and size bucket.
struct Ladder {
    std::vector<std::vector<double>> bid, ask;   // [tier][size]
    void resize(const ClientConfig& c) { bid.assign(c.tiers.size(), {}); ask = bid; for (size_t i = 0; i < c.tiers.size(); ++i) { bid[i].assign(c.tiers[i].sizes.size(), 0.0); ask[i].assign(c.tiers[i].sizes.size(), 0.0); } }
    double skew(size_t tier) const { return bid[tier].empty() ? 0 : 0.5 * (ask[tier][0] - bid[tier][0]); }   // > 0: dealer wants to buy (long ask offset, short bid)
};

} // namespace fxmm
