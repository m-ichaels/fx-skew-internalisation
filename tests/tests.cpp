// Unit and known-answer tests for the dealer simulator and quoters.
#include "fxmm/dealer.hpp"
#include <iostream>
#include <cmath>

using namespace fxmm;
static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; std::cerr << "FAIL: " << msg << "  [" #cond "]\n"; } } while (0)
#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((double)(a) - (double)(b)) <= (tol), msg << " (" << (a) << " vs " << (b) << ")")

static MarketParams mp_default() { MarketParams m; m.sigma = 0.3; m.gamma = 0.02; m.psi = 0.15; m.Q = 20; return m; }

static void test_pnl_decomposition_sums() {
    ClientConfig c = ClientConfig::defaults(); MarketParams mp = mp_default();
    for (std::string n : {"HJB", "GLFT", "AS"}) {
        std::unique_ptr<Quoter> q; if (n == "HJB") q = std::make_unique<HJBQuoter>(c, mp); else if (n == "GLFT") q = std::make_unique<GLFTQuoter>(c, mp); else q = std::make_unique<ASQuoter>(c, mp);
        for (int seed = 1; seed <= 3; ++seed) {
            SimConfig sc; sc.diffusion = true; sc.diffusion_sigma = 0.3; sc.horizon_s = 3600; sc.seed = seed; sc.tox_route = (n == "HJB"); sc.last_look = (seed == 2); sc.anticipate = (seed == 3);
            DealerSim sim(nullptr, c, *q, sc); sim.set_reader_reference_slope(0.05);
            SimResult r = sim.run();
            CHECK_NEAR(r.pnl.check(), 0.0, 1e-6 * std::max(1.0, std::fabs(r.pnl.total)), n << " seed " << seed << ": spread + inventory + adverse + hedge exec + hedge impact == total P&L");
            CHECK(r.n_trades > 0, n << ": trades happened");
        }
    }
}

static void test_no_skew_is_symmetric() {
    // gamma = 0, no adverse selection, no readers, no hedging: the optimal ladder is symmetric and inventory-independent
    ClientConfig c = ClientConfig::defaults(); for (auto& t : c.tiers) { t.mu = 0; t.reader = false; }
    MarketParams mp = mp_default(); mp.gamma = 0; mp.Q = 30;
    HJBOptions o; o.adverse = false; o.readers = false; o.hedging = false;
    HJBQuoter h(c, mp, o); Ladder L; L.resize(c);
    double maxasym = 0, maxdep = 0; Ladder L0; L0.resize(c); h.quote(0, L0);
    for (double q = -8; q <= 8; q += 2) {   // interior of the grid: the boundary truncates trades
        h.quote(q, L); for (size_t i = 0; i < L.bid.size(); ++i) for (size_t j = 0; j < L.bid[i].size(); ++j) { maxasym = std::max(maxasym, std::fabs(L.bid[i][j] - L.ask[i][j])); maxdep = std::max(maxdep, std::fabs(L.bid[i][j] - L0.bid[i][j])); } }
    CHECK_NEAR(maxasym, 0.0, 0.1 + 1e-9, "no risk aversion: symmetric quotes (within one 0.1-pip grid step; the finite grid still reflects at +-Q)");
    CHECK_NEAR(maxdep, 0.0, 0.1 + 1e-9, "no risk aversion: quotes independent of inventory (within one grid step)");
    // with risk aversion the skew has the right sign: long => lower ask, higher bid offset
    MarketParams mp2 = mp_default(); HJBQuoter h2(c, mp2, o);
    Ladder Lp, Lm; Lp.resize(c); Lm.resize(c); h2.quote(5, Lp); h2.quote(-5, Lm);
    CHECK(Lp.ask[0][0] < Lp.bid[0][0] && Lm.ask[0][0] > Lm.bid[0][0], "long inventory skews to sell, short skews to buy");
}

static void test_no_readers_reduces_to_2023_model() {
    ClientConfig c = ClientConfig::defaults(); for (auto& t : c.tiers) if (t.reader) t.rho = 0;   // readers present but not reading
    MarketParams mp = mp_default();
    HJBOptions a; HJBOptions b; b.readers = false;
    HJBQuoter h1(c, mp, a), h2(c, mp, b);
    double maxdiff = 0; Ladder L1, L2; L1.resize(c); L2.resize(c);
    for (double q = -20; q <= 20; q += 1) { h1.quote(q, L1); h2.quote(q, L2); for (size_t i = 0; i < L1.bid.size(); ++i) for (size_t j = 0; j < L1.bid[i].size(); ++j) maxdiff = std::max({maxdiff, std::fabs(L1.bid[i][j] - L2.bid[i][j]), std::fabs(L1.ask[i][j] - L2.ask[i][j])}); maxdiff = std::max(maxdiff, std::fabs(h1.hedge(q) - h2.hedge(q))); }
    CHECK_NEAR(maxdiff, 0.0, 1e-9, "rho = 0: the price-reading model reduces to the 2023 model exactly");
    // and with rho > 0 the reader tier gets less skew than without readers
    ClientConfig c2 = ClientConfig::defaults(); HJBQuoter r1(c2, mp, a), r2(c2, mp, b);
    Ladder A, B; A.resize(c2); B.resize(c2); r1.quote(15, A); r2.quote(15, B);
    size_t rt = 3;
    CHECK(std::fabs(A.bid[rt][0] - A.ask[rt][0]) < std::fabs(B.bid[rt][0] - B.ask[rt][0]), "readers are shown less skew than the same tier without price reading");
    CHECK(std::fabs(A.bid[0][0] - A.ask[0][0]) > 0.5, "non-reader tiers keep their skew");
}

static void test_adverse_selection_widens() {
    ClientConfig c = ClientConfig::defaults(); MarketParams mp = mp_default();
    HJBOptions a; a.readers = false; HJBOptions b; b.readers = false; b.adverse = false;
    HJBQuoter with(c, mp, a), without(c, mp, b);
    Ladder A, B; A.resize(c); B.resize(c); with.quote(0, A); without.quote(0, B);
    CHECK(A.bid[0][0] + A.ask[0][0] > B.bid[0][0] + B.ask[0][0] - 1e-9, "adverse selection does not narrow the informed tier's spread at q = 0");
    QHQuoter qa(c, mp, true), qb(c, mp, false);
    CHECK(qa.spread0(0, 5.0) > qb.spread0(0, 5.0), "closed form: adverse selection widens the informed tier by mu z / 2");
    CHECK_NEAR(qa.spread0(0, 5.0) - qb.spread0(0, 5.0), c.tiers[0].mu * 5.0 + (qa.a() - qb.a()) * 5.0 / 2, 1e-9, "closed-form widening term mu z");
}

static void test_closed_form_vs_numerical() {
    // one tier, exponential-like response, no adverse selection: the QH skew slope should be close to the HJB's
    ClientConfig c; c.Q = 20; c.tiers = {{"flow", 0.05, {1}, {1.0}, -1.0, 6.0, 0.0, 30, false, 0}};
    MarketParams mp = mp_default(); mp.psi = 1e6;   // no hedging in either
    QHQuoter qh(c, mp, false);
    HJBOptions o; o.adverse = false; o.readers = false; o.hedging = false; HJBQuoter h(c, mp, o);
    Ladder L5, Lm5; L5.resize(c); Lm5.resize(c); h.quote(5, L5); h.quote(-5, Lm5);
    double slope_num = (L5.bid[0][0] - Lm5.bid[0][0]) / 10.0;
    CHECK(slope_num > 0.5 * qh.a() && slope_num < 2.0 * qh.a(), "quadratic-Hamiltonian skew slope within a factor 2 of the numerical HJB (" << qh.a() << " vs " << slope_num << ")");
    Ladder L0; L0.resize(c); h.quote(0, L0);
    CHECK_NEAR(L0.bid[0][0], qh.spread0(0, 1.0), 0.25, "quadratic-Hamiltonian base offset vs numerical HJB");
    // GLFT: symmetric at q=0 and skew sign correct
    GLFTQuoter g(c, mp); Ladder G; G.resize(c); g.quote(0, G); CHECK_NEAR(G.bid[0][0], G.ask[0][0], 1e-12, "GLFT symmetric at q = 0");
    g.quote(3, G); CHECK(G.bid[0][0] > G.ask[0][0], "GLFT long inventory skews to sell");
}

static void test_classifier_learns_planted_signal() {
    ToxicityClassifier clf(2); Rng rng(5);
    for (int n = 0; n < 4000; ++n) {
        int tier = rng.uniform() < 0.5 ? 0 : 1; double z = 1 + 4 * rng.uniform();
        std::vector<double> x = clf.features(tier, 2, z, 0.0, 1);
        double ptrue = tier == 0 ? 0.8 : 0.1;
        clf.update(x, rng.uniform() < ptrue ? 1.0 : 0.0);
    }
    double s0 = clf.score(clf.features(0, 2, 2.0, 0.0, 1)), s1 = clf.score(clf.features(1, 2, 2.0, 0.0, 1));
    CHECK(s0 > 0.6 && s1 < 0.3, "online Bayesian logistic classifier separates a toxic tier (" << s0 << ") from a benign one (" << s1 << ")");
}

static void test_last_look_and_latency() {
    ClientConfig c = ClientConfig::defaults(); MarketParams mp = mp_default(); HJBQuoter h(c, mp);
    SimConfig sc; sc.diffusion = true; sc.diffusion_sigma = 0.3; sc.horizon_s = 3600; sc.seed = 7;
    DealerSim base(nullptr, c, h, sc); base.set_reader_reference_slope(0.05); SimResult r0 = base.run();
    SimConfig ll = sc; ll.last_look = true; ll.ll_hold_ms = 200; ll.ll_tolerance_pips = 0.05;
    DealerSim s1(nullptr, c, h, ll); s1.set_reader_reference_slope(0.05); SimResult r1 = s1.run();
    CHECK(r1.n_rejected > 0, "last look with a tight tolerance rejects some requests");
    CHECK(r0.n_rejected == 0, "no last look: no rejections");
    SimConfig lat = sc; lat.latency_ms = 2000;
    DealerSim s2(nullptr, c, h, lat); s2.set_reader_reference_slope(0.05); SimResult r2 = s2.run();
    CHECK_NEAR(r2.pnl.check(), 0.0, 1e-6, "P&L identity holds with stale quotes");
    CHECK(r2.n_trades > 0, "trades with latency");
}

static void test_tick_loader() {
    FILE* f = std::fopen("kat_ticks.csv", "w"); std::fprintf(f, "ts_ms,bid,ask,bid_vol,ask_vol\n1000,1.08000,1.08010,1,1\n2000,1.08010,1.08020,1,1\n3000,1.08000,1.08010,1,1\n"); std::fclose(f);
    TickSeries s = read_ticks("kat_ticks.csv"); std::remove("kat_ticks.csv");
    CHECK(s.ticks.size() == 3, "tick file read"); CHECK_NEAR(s.ticks[0].ask - s.ticks[0].bid, 1.0, 1e-9, "spread in pips"); CHECK_NEAR(s.mid_at(2500), 10801.5, 1e-9, "mid lookup at time");
}

int main() {
    test_pnl_decomposition_sums(); test_no_skew_is_symmetric(); test_no_readers_reduces_to_2023_model(); test_adverse_selection_widens();
    test_closed_form_vs_numerical(); test_classifier_learns_planted_signal(); test_last_look_and_latency(); test_tick_loader();
    std::cout << (failures ? "TESTS FAILED: " : "TESTS PASSED: ") << (checks - failures) << "/" << checks << " checks\n";
    return failures ? 1 : 0;
}
