"""
Print a ventilation schedule for a real place and date.

    .venv/bin/python scripts/advise.py --location palo_alto
    .venv/bin/python scripts/advise.py --lat 40.71 --lon -74.01 --date 2024-07-15
    .venv/bin/python scripts/advise.py --location seattle --hours 24 --no-fan

This is the bit you'd actually use. Give it a place and a date, and it pulls
the weather, runs the fitted room model and the planning controller over it,
and tells you when to open the window and what to expect if you do. It also
prints what happens if you do nothing, because advice without that comparison
isn't worth acting on.

The room is whatever is in windwindow/config.py. Edit those numbers to match
your own room; that's the intended way to use this somewhere else.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from windwindow import config as cfg
from windwindow.forecast import Forecast
from windwindow.gains import daily_summary
from windwindow.mpc import MPCPolicy
from windwindow.policies import FixedActionPolicy, SimpleRulePolicy
from windwindow.realweather import LOCATIONS, RealWeather, build
from windwindow.simulator import run_session, summarise
from windwindow.statespace import StateSpaceModel

MODEL_PATH = pathlib.Path("results/statespace.json")
PENALTY_PATH = pathlib.Path("results/fan_penalty.json")


def _hhmm(hours: float) -> str:
    h = hours % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def describe(u: float, f: float) -> str:
    if u < 0.05:
        return "keep shut"
    if u < 0.4:
        part = "open a little"
    elif u < 0.75:
        part = "open halfway"
    else:
        part = "open fully"
    return part + (", fan on" if f > 0.05 else "")


def schedule(df: pd.DataFrame, start_hour: float, min_block_min=45.0) -> list:
    """Collapse the controller's trace into a readable list of instructions.

    Anything shorter than min_block_min is folded into the preceding
    instruction. The controller adjusts the opening more finely than this, but
    a schedule a person is meant to follow should not ask them to change the
    window every ten minutes, and the fine adjustments are worth very little.
    """
    step_min = cfg.DT_CONTROL / 60.0
    text = [describe(u, f) for u, f in zip(df["u"], df["f"])]
    blocks, start = [], 0
    for i in range(1, len(text) + 1):
        if i == len(text) or text[i] != text[start]:
            blocks.append({"from_min": start * step_min,
                           "to_min": i * step_min,
                           "action": text[start],
                           "t_out": float(df["t_out_true"].iloc[start:i].mean()),
                           "t_in": float(df["t_in_true"].iloc[start:i].mean())})
            start = i
    # Merge anything too short to be worth acting on into its predecessor.
    merged = []
    for b in blocks:
        if merged and (b["to_min"] - b["from_min"]) < min_block_min:
            merged[-1]["to_min"] = b["to_min"]
        elif merged and merged[-1]["action"] == b["action"]:
            merged[-1]["to_min"] = b["to_min"]
        else:
            merged.append(dict(b))
    # Split anything that runs past midnight, so each printed line belongs to
    # one day and the clock times read forwards.
    out = []
    for b in merged:
        h0 = start_hour + b["from_min"] / 60.0
        h1 = start_hour + b["to_min"] / 60.0
        while True:
            day_end = (int(h0 // 24) + 1) * 24.0
            stop = min(h1, day_end)
            # Recompute the outdoor mean over this segment rather than
            # inheriting the parent block's, which would be wrong for the half
            # of a block that falls on the other side of midnight.
            i0 = int(round((h0 - start_hour) * 60.0 / step_min))
            i1 = max(i0 + 1, int(round((stop - start_hour) * 60.0 / step_min)))
            seg = df["t_out_true"].iloc[i0:i1]
            out.append({**b,
                        "t_out": float(seg.mean()) if len(seg) else b["t_out"],
                        "day": int(h0 // 24) + 1,
                        "clock_from": _hhmm(h0),
                        "clock_to": "24:00" if stop == day_end else _hhmm(stop)})
            if stop >= h1 - 1e-9:
                break
            h0 = stop
    return out


def fetch_adhoc(lat, lon, date, hours):
    """Download weather for an arbitrary point, via the step 15 fetcher."""
    from fetch_weather import fetch_json

    end = (pd.Timestamp(date) + pd.Timedelta(hours=hours + 24)).strftime("%Y-%m-%d")
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={end}"
           "&hourly=temperature_2m&timezone=auto")
    payload = fetch_json(url)
    h = payload["hourly"]
    temps = np.array([v for v in h["temperature_2m"] if v is not None], dtype=float)
    times = pd.to_datetime(h["time"][:len(temps)])
    n = int(hours * 3600 / cfg.DT_PHYSICS) + 1
    t = np.arange(n) * cfg.DT_PHYSICS
    grid = np.interp(t, np.arange(len(temps)) * 3600.0, temps)
    return RealWeather(
        name=f"{lat},{lon}", label=f"{lat:.2f}, {lon:.2f}",
        note=f"fetched for {date}", t_out_grid=grid, start_time=times[0],
        duration_s=hours * 3600.0, t_in0=float(grid[0]) + 2.0,
        t_wall0=float(grid[0]) + 1.0, heater_w=cfg.GAIN_BASE_W)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--location", choices=sorted(LOCATIONS),
                   help="one of the cached climates")
    g.add_argument("--lat", type=float, help="latitude, with --lon and --date")
    ap.add_argument("--lon", type=float)
    ap.add_argument("--date", default="2024-07-08")
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--start-hour", type=int, default=0,
                    help="hour of day to begin planning from")
    ap.add_argument("--no-fan", action="store_true",
                    help="advise on the window alone")
    ap.add_argument("--comfort", type=float, default=cfg.COMFORT_C,
                    help="temperature to stay below, degC")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH} is missing. Run scripts/run_all.py once first.")
        return 1
    model = StateSpaceModel(theta=np.array(json.load(open(MODEL_PATH))["theta"]))
    penalty = (1e9 if args.no_fan
               else json.load(open(PENALTY_PATH))["fan_penalty"]
               if PENALTY_PATH.exists() else 0.35)

    if args.location:
        w = build(args.location, hours=args.hours, start_hour=args.start_hour)
    else:
        if args.lon is None:
            ap.error("--lat requires --lon")
        w = fetch_adhoc(args.lat, args.lon, args.date, args.hours)

    gs = daily_summary()
    print(f"\nVentilation advice for {w.label}")
    print(f"  {w.start_time:%Y-%m-%d %H:%M} onward, {args.hours:.0f} hours")
    print(f"  outdoor {w.t_out_grid.min():.1f} to {w.t_out_grid.max():.1f} degC, "
          f"overnight mean {w.summary()['night_mean']:.1f} degC")
    print(f"  room {cfg.ROOM_L:.0f} x {cfg.ROOM_W:.0f} x {cfg.ROOM_H:.1f} m, "
          f"heat gains {gs['mean_w']:.0f} W mean and {gs['peak_w']:.0f} W peak")
    print(f"  staying below {args.comfort:.0f} degC")

    fc = Forecast.build(w.t_out_grid, seed=0)
    start_hour = w.start_time.hour + w.start_time.minute / 60.0

    runs = {}
    for label, pol in (
            ("controller", MPCPolicy(model, fc, fan_penalty=penalty,
                                     comfort=args.comfort)),
            ("do nothing (shut)", FixedActionPolicy(0.0, 0.0)),
            ("window open throughout", FixedActionPolicy(1.0, 0.0)),
            ("open when cooler outside", SimpleRulePolicy())):
        df_run = run_session(w, pol, sensor_seed=7)
        runs[label] = summarise(df_run) | {"_df": df_run}

    df = runs["controller"]["_df"]
    print(f"\nRecommended schedule")
    day = None
    for b in schedule(df, start_hour):
        if b["day"] != day:
            day = b["day"]
            print(f"  Day {day}")
        print(f"    {b['clock_from']} - {b['clock_to']}  {b['action']:22s} "
              f"outdoor {b['t_out']:5.1f} degC")

    print(f"\nExpected outcome over {args.hours:.0f} hours")
    print(f"  {'':28s} {'peak':>7s} {'hours above':>12s} {'fan':>9s}")
    print(f"  {'':28s} {'degC':>7s} {args.comfort:>11.0f}C {'kWh':>9s}")
    for label in ("controller", "open when cooler outside",
                  "window open throughout", "do nothing (shut)"):
        s = runs[label]
        print(f"  {label:28s} {s['max_t_in']:7.1f} "
              f"{s['minutes_over_26']/60:12.1f} {s['fan_wh']/1000:9.2f}")

    ctl = runs["controller"]
    shut = runs["do nothing (shut)"]
    rule = runs["open when cooler outside"]
    saved = shut["degree_minutes_over_26"] - ctl["degree_minutes_over_26"]
    print(f"\nVerdict")
    if shut["degree_minutes_over_26"] < 60:
        print(f"  This room stays comfortable here without doing anything. "
              f"Ventilation control is not worth installing for this weather.")
    else:
        pct = saved / shut["degree_minutes_over_26"] * 100
        print(f"  Following the schedule cuts time-above-{args.comfort:.0f} by "
              f"{pct:.0f}% against leaving the window shut,")
        print(f"  and the peak falls from {shut['max_t_in']:.1f} to "
              f"{ctl['max_t_in']:.1f} degC, for {ctl['fan_wh']/1000:.2f} kWh of "
              f"fan electricity.")
        better = (1 - ctl["degree_minutes_over_26"]
                  / max(rule["degree_minutes_over_26"], 1e-9)) * 100
        print(f"  Against the simpler habit of opening up whenever it is cooler "
              f"outside, it is {better:+.0f}%.")
        if ctl["max_t_in"] > args.comfort + 3:
            print(f"  It still peaks well above {args.comfort:.0f} degC, so "
                  f"ventilation alone is not sufficient in this climate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
