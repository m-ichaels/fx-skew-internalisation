#pragma once
// Core integer types. Prices are integer ticks, quantities integer lots, times ns since epoch.
// Working in integers is deliberate: the replay must be bit-for-bit deterministic and the
// known-answer tests compare exact fills.
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace fxmm {

using Price = int64_t;   // ticks
using Qty   = int64_t;   // lots
using Ts    = int64_t;   // ns since epoch (exchange event time unless stated otherwise)

constexpr Ts NS_PER_MS = 1'000'000LL;
constexpr Ts NS_PER_S  = 1'000'000'000LL;

enum class Side : uint8_t { Bid = 0, Ask = 1 };
inline Side other(Side s) { return s == Side::Bid ? Side::Ask : Side::Bid; }
inline int sign(Side s) { return s == Side::Bid ? +1 : -1; }   // +1 buy, -1 sell
inline const char* side_str(Side s) { return s == Side::Bid ? "bid" : "ask"; }

// Tape event kinds.
//  Level    : absolute quantity at (side, price) is now `qty` (0 = level removed)
//  Trade    : `qty` traded at `price`; `side` is the PASSIVE side that was hit
//             (Bid => a seller crossed the spread, Ask => a buyer did)
//  Snapshot : book reset marker (a synchronisation snapshot follows as Level events)
enum class EvKind : uint8_t { Level = 0, Trade = 1, Snapshot = 2 };

struct L2Event {
    Ts     ts_ex;   // exchange event time
    Ts     ts_rx;   // local receive time (0 for synthetic tapes)
    EvKind kind;
    Side   side;
    Price  price;
    Qty    qty;
};

struct SymbolSpec {
    std::string name = "BTCUSDT";
    double tick = 0.01;      // price increment in quote currency
    double lot  = 0.00001;   // quantity increment in base currency
    Price to_ticks(double px) const { return (Price)std::llround(px / tick); }
    Qty   to_lots(double q)   const { return (Qty)std::llround(q / lot); }
    double px(Price p) const { return p * tick; }
    double q(Qty l)    const { return l * lot; }
};

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

} // namespace fxmm
