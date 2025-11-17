#!/usr/bin/env python3
"""Download Dukascopy top-of-book ticks and write one CSV per day.

    python tools/dukascopy.py EURUSD 2025-04-14 2025-04-18 data/ticks

Format: hourly `.bi5` files, LZMA-compressed, 20-byte big-endian records
(ms since hour, ask, bid, ask volume, bid volume); prices are integers scaled by 1e5 for EURUSD.
Months in the URL are zero-based. Free, no key.
"""
import datetime as dt
import lzma
import os
import struct
import sys
import time
import urllib.request

SCALE = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}


def fetch(url, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; fx-skew-internalisation)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            time.sleep(1.2)   # be polite: the feed rate-limits bursts (HTTP 429)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(3.0 * (k + 1))
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise RuntimeError("giving up on " + url)


def day_ticks(sym, day):
    scale = SCALE.get(sym, 1e5)
    rows = []
    for h in range(24):
        url = f"https://datafeed.dukascopy.com/datafeed/{sym}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{h:02d}h_ticks.bi5"
        raw = fetch(url)
        if not raw:
            continue
        data = lzma.decompress(raw)
        base = dt.datetime(day.year, day.month, day.day, h, tzinfo=dt.timezone.utc)
        for off in range(0, len(data) - len(data) % 20, 20):
            ms, ask, bid, av, bv = struct.unpack(">iiiff", data[off:off + 20])
            t = base + dt.timedelta(milliseconds=ms)
            rows.append((int(t.timestamp() * 1000), bid / scale, ask / scale, bv, av))
    return rows


def main():
    sym, d0, d1, out = sys.argv[1], dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    d = d0
    while d <= d1:
        path = os.path.join(out, f"{sym}_{d.isoformat()}.csv")
        rows = day_ticks(sym, d)
        with open(path, "w") as f:
            f.write("ts_ms,bid,ask,bid_vol,ask_vol\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.5f},{r[2]:.5f},{r[3]:.2f},{r[4]:.2f}\n")
        print(path, len(rows), "ticks", flush=True)
        d += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
