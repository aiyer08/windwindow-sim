"""
Download real hourly weather once and cache it to data/weather.

Run this before the rest of the pipeline. Everything afterwards reads the
cached CSVs, so results are reproducible offline and do not depend on the
API staying available.
"""

import json
import pathlib
import shutil
import ssl
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from windwindow.realweather import LOCATIONS, WEATHER_DIR, archive_url

def fetch_json(url: str) -> dict:
    """GET a JSON document, coping with a missing CA bundle.

    Python builds installed from python.org often ship without access to the
    system trust store, so a plain urlopen fails TLS verification even though
    the network is fine. certifi supplies the bundle; curl is the fallback,
    since it uses the system store directly.
    """
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=60, context=ctx) as fh:
            return json.load(fh)
    except Exception as exc:
        if not shutil.which("curl"):
            raise
        print(f"    (urllib failed: {type(exc).__name__}; falling back to curl)")
        out = subprocess.run(["curl", "-sS", "--max-time", "60", url],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)


def main():
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, loc in LOCATIONS.items():
        path = WEATHER_DIR / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["time"])
            print(f"  {name:12s} cached      {len(df):4d} hours  {path}")
        else:
            payload = fetch_json(archive_url(loc))
            h = payload["hourly"]
            df = pd.DataFrame({
                "time": pd.to_datetime(h["time"]),
                "temperature_2m": h["temperature_2m"],
                "relative_humidity_2m": h.get("relative_humidity_2m"),
                "cloud_cover": h.get("cloud_cover"),
            }).dropna(subset=["temperature_2m"])
            df.to_csv(path, index=False)
            print(f"  {name:12s} downloaded  {len(df):4d} hours  {path}")
        rows.append({
            "location": name, "label": loc["label"], "note": loc["note"],
            "hours": len(df),
            "t_min": df.temperature_2m.min(), "t_max": df.temperature_2m.max(),
            "t_mean": round(df.temperature_2m.mean(), 2),
        })

    s = pd.DataFrame(rows)
    print()
    print(s.to_string(index=False))
    pathlib.Path("results/tables").mkdir(parents=True, exist_ok=True)
    s.to_csv("results/tables/weather_sources.csv", index=False)
    print("\n  wrote results/tables/weather_sources.csv")

    ok = len(s) == len(LOCATIONS) and bool((s.hours >= 48).all())
    print(f"\n  [{'PASS' if ok else 'FAIL'}] all {len(LOCATIONS)} locations "
          f"cached with at least 48 hours")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
