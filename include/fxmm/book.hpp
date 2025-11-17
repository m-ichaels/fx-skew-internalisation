#pragma once
// Plain L2 book: absolute quantity per price level. Used by the rebuilder and by the replay
// simulator; the generative simulators keep a relative (reference-price) book, see relbook.hpp.
#include "types.hpp"
#include <map>
#include <functional>
#include <algorithm>

namespace fxmm {

class Book {
public:
    using Bids = std::map<Price, Qty, std::greater<Price>>;
    using Asks = std::map<Price, Qty, std::less<Price>>;

    void clear() { bids_.clear(); asks_.clear(); }

    // Returns previous quantity at that level.
    Qty set(Side s, Price p, Qty q) {
        Qty prev = 0;
        if (s == Side::Bid) prev = set_in(bids_, p, q); else prev = set_in(asks_, p, q);
        return prev;
    }
    Qty qty_at(Side s, Price p) const {
        if (s == Side::Bid) { auto it = bids_.find(p); return it == bids_.end() ? 0 : it->second; }
        auto it = asks_.find(p); return it == asks_.end() ? 0 : it->second;
    }
    bool has_bid() const { return !bids_.empty(); }
    bool has_ask() const { return !asks_.empty(); }
    bool valid() const { return has_bid() && has_ask(); }
    Price best_bid() const { return bids_.empty() ? 0 : bids_.begin()->first; }
    Price best_ask() const { return asks_.empty() ? 0 : asks_.begin()->first; }
    Qty best_bid_qty() const { return bids_.empty() ? 0 : bids_.begin()->second; }
    Qty best_ask_qty() const { return asks_.empty() ? 0 : asks_.begin()->second; }
    Price best(Side s) const { return s == Side::Bid ? best_bid() : best_ask(); }
    Qty best_qty(Side s) const { return s == Side::Bid ? best_bid_qty() : best_ask_qty(); }
    double mid() const { return 0.5 * (best_bid() + best_ask()); }
    Price spread() const { return best_ask() - best_bid(); }
    // Sum of quantity over the first n levels of a side.
    Qty depth(Side s, int n) const {
        Qty d = 0; int i = 0;
        if (s == Side::Bid) for (auto& kv : bids_) { if (i++ >= n) break; d += kv.second; }
        else for (auto& kv : asks_) { if (i++ >= n) break; d += kv.second; }
        return d;
    }
    // Qty available at prices at least as good as `limit` (for a market/marketable order on side `taker`)
    // taker=Bid consumes asks with price <= limit; taker=Ask consumes bids with price >= limit.
    template <class F> void walk(Side taker, F&& f) const {
        if (taker == Side::Bid) { for (auto& kv : asks_) if (!f(kv.first, kv.second)) break; }
        else { for (auto& kv : bids_) if (!f(kv.first, kv.second)) break; }
    }
    const Bids& bids() const { return bids_; }
    const Asks& asks() const { return asks_; }
    size_t levels(Side s) const { return s == Side::Bid ? bids_.size() : asks_.size(); }

private:
    template <class M> static Qty set_in(M& m, Price p, Qty q) {
        auto it = m.find(p);
        Qty prev = it == m.end() ? 0 : it->second;
        if (q <= 0) { if (it != m.end()) m.erase(it); }
        else if (it == m.end()) m.emplace(p, q);
        else it->second = q;
        return prev;
    }
    Bids bids_;
    Asks asks_;
};

} // namespace fxmm
