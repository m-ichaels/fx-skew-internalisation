#!/usr/bin/env python3
"""Download a month of EURUSD tick quotes from histdata.com (free) and split it into UTC days.

    python tools/histdata.py EURUSD 2025 4 data/ticks [--days 2025-04-14,2025-04-18]

histdata timestamps are fixed EST (UTC-5, no DST): the converter adds 5 hours.  Output columns
match tools/dukascopy.py (ts_ms,bid,ask,bid_vol,ask_vol; volumes are 0 here).
"""
import datetime as dt
import io
import os
import re
import sys
import zipfile

import requests


def download(sym, year, month):
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0"})
    url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{sym.lower()}/{year}/{month}"
    tk = re.search(r'id="tk" value="([^"]+)"', s.get(url, timeout=30).text).group(1)
    r = s.post("https://www.histdata.com/get.php", data={"tk": tk, "date": str(year), "datemonth": f"{year}{month:02d}", "platform": "ASCII", "timeframe": "T", "fxpair": sym.upper()},
               headers={"Referer": url}, timeout=600)
    assert r.content[:2] == b"PK", "download failed"
    return r.content


def main():
    sym, year, month, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    days = None
    if "--days" in sys.argv:
        a, b = sys.argv[sys.argv.index("--days") + 1].split(",")
        d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        days = {(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)}
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, f"histdata_{sym}_{year}{month:02d}.zip")
    if not os.path.exists(cache):
        open(cache, "wb").write(download(sym, year, month))
    z = zipfile.ZipFile(cache)
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    files = {}
    with z.open(name) as f:
        for line in io.TextIOWrapper(f, encoding="ascii"):
            stamp, bid, ask, _ = line.strip().split(",")
            t = dt.datetime.strptime(stamp, "%Y%m%d %H%M%S%f").replace(tzinfo=dt.timezone(dt.timedelta(hours=-5))).astimezone(dt.timezone.utc)
            d = t.date().isoformat()
            if days and d not in days:
                continue
            if d not in files:
                files[d] = open(os.path.join(out, f"{sym}_{d}.csv"), "w")
                files[d].write("ts_ms,bid,ask,bid_vol,ask_vol\n")
            files[d].write(f"{int(t.timestamp() * 1000)},{bid},{ask},0,0\n")
    for d, fh in sorted(files.items()):
        fh.close(); print(d, os.path.getsize(os.path.join(out, f"{sym}_{d}.csv")) // 1024, "KB")


if __name__ == "__main__":
    main()
