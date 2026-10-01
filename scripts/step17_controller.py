"""
Step 17: settle the controller's one free parameter, and measure what each
part of it contributes.

The fan penalty is the only quantity in the objective that is not fixed by
physics or by the comfort threshold. It is chosen here on a single climate and
a single weather seed; step 18 then evaluates that choice on four climates and
three seeds it has never seen.

The ablations answer a question the first version of this project could not:
how much of the result comes from planning ahead, how much from the forecast,
and how much from being allowed to open the window part way.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.forecast import Forecast
from windwindow.mpc import BLOCKS_MIN, HORIZON_S, MPCPolicy, REPLAN_S
from windwindow.policies import FixedActionPolicy, SimpleRulePolicy, TimerPolicy
from windwindow.realweather import build
from windwindow.simulator import run_session, summarise
from windwindow.statespace import StateSpaceModel

TUNE_LOC, TUNE_SEED = "palo_alto", 1
HOURS = 48.0
FAN_PENALTIES = (0.0, 0.15, 0.35, 0.7, 1.5, 4.0)
CHOICE_PATH = pathlib.Path("results/fan_penalty.json")

def load_model():
    blob = json.load(open("results/statespace.json"))
    return StateSpaceModel(theta=np.array(blob["theta"]))

def main():
    model = load_model()
    w = build(TUNE_LOC, seed=TUNE_SEED, hours=HOURS, start_hour=0)
    fc = Forecast.build(w.t_out_grid, seed=TUNE_SEED)
    ss = 20_000 + TUNE_SEED

    print(f"planning horizon {HORIZON_S/3600:.1f} h in {len(BLOCKS_MIN)} blocks, "
          f"re-planned every {REPLAN_S/60:.0f} min")
    print(f"tuning on {w.label}, seed {TUNE_SEED}, {HOURS:.0f} h, outdoor "
          f"{w.t_out_grid.min():.1f} to {w.t_out_grid.max():.1f} degC\n")

    rows = []
    def run(label, pol, group):
        t0 = time.time()
        s = summarise(run_session(w, pol, sensor_seed=ss))
        s.update(label=label, group=group, seconds=round(time.time() - t0, 1))
        rows.append(s)
        return s

    print("references:")
    for label, pol in (("sealed", FixedActionPolicy(0.0, 0.0)),
                       ("window open throughout", FixedActionPolicy(1.0, 0.0)),
                       ("open with fan throughout", FixedActionPolicy(1.0, 1.0)),
                       ("timer, 50% duty", TimerPolicy()),
                       ("threshold rule", SimpleRulePolicy())):
        s = run(label, pol, "reference")
        print(f"  {label:26s} {s['degree_minutes_over_26']:8.0f} degree-minutes  "
              f"fan {s['fan_wh']:6.0f} Wh")

    print(f"\nfan penalty sweep (degree-minutes charged per watt-hour):")
    print(f"  {'penalty':>8s} {'deg-min':>9s} {'fan Wh':>8s} {'peak':>6s} "
          f"{'open h':>7s}")
    for fp in FAN_PENALTIES:
        s = run(f"mpc, fan penalty {fp:g}", MPCPolicy(model, fc, fan_penalty=fp),
                "sweep")
        s["fan_penalty"] = fp
        print(f"  {fp:8.2f} {s['degree_minutes_over_26']:9.0f} {s['fan_wh']:8.0f} "
              f"{s['max_t_in']:6.1f} {s['window_open_minutes']/60:7.1f}")

    sw = pd.DataFrame([r for r in rows if r["group"] == "sweep"])
    # The fan buys comfort with electricity, and the sweep is monotonic: more
    # fan is always more comfortable. So the choice is where the returns stop
    # justifying the electricity. Walk up the fan-energy axis and keep going
    # while the next increment still yields at least MIN_RETURN degree-minutes
    # per watt-hour.
    #
    # Walking is necessary rather than picking the point whose local gradient
    # clears the threshold. The gradient at a point describes the value of
    # moving away from it, so selecting on it directly picks the stingiest
    # setting, which is the worst one. An earlier version did exactly that and
    # chose the setting that switched the fan off entirely.
    MIN_RETURN = 1.0
    sw = sw.sort_values("fan_wh").reset_index(drop=True)
    returns = [float("nan")]
    for i in range(1, len(sw)):
        d_comfort = sw.degree_minutes_over_26[i - 1] - sw.degree_minutes_over_26[i]
        d_energy = sw.fan_wh[i] - sw.fan_wh[i - 1]
        returns.append(d_comfort / d_energy if d_energy > 1e-9 else float("nan"))
    sw["return_deg_min_per_wh"] = returns
    chosen_i = 0
    for i in range(1, len(sw)):
        if returns[i] >= MIN_RETURN:
            chosen_i = i
        else:
            break
    chosen = float(sw.fan_penalty[chosen_i])
    print(f"\n  marginal return of each extra step of fan energy:")
    for i, r in sw.iterrows():
        extra = "" if i == 0 else f"{returns[i]:6.2f} degree-minutes per Wh"
        mark = "  <- chosen" if i == chosen_i else ""
        print(f"    penalty {r.fan_penalty:4.2f}  {r.fan_wh:6.0f} Wh  "
              f"{r.degree_minutes_over_26:6.0f} deg-min  {extra}{mark}")
    print(f"  chosen fan penalty: {chosen:g}  "
          f"(returns fall below {MIN_RETURN:.1f} degree-minutes per Wh beyond it)")
    CHOICE_PATH.write_text(json.dumps(
        {"fan_penalty": chosen, "tuned_on": TUNE_LOC, "seed": TUNE_SEED}, indent=2))
    print(f"  wrote {CHOICE_PATH}")

    print(f"\nablations at the chosen penalty, each removing one capability:")
    abl = [
        ("full controller", dict()),
        ("no forecast (persistence)", dict(use_forecast=False)),
        ("perfect forecast", dict(_perfect=True)),
        ("short horizon (7.5 h)", dict(blocks_min=(15, 15, 30, 30, 60, 60, 120, 120))),
        ("very short horizon (1 h)", dict(blocks_min=(15, 15, 15, 15))),
        ("open or shut only", dict(_binary=True)),
    ]
    for label, kw in abl:
        kw = dict(kw)
        perfect = kw.pop("_perfect", False)
        binary = kw.pop("_binary", False)
        f_use = Forecast.build(w.t_out_grid, seed=TUNE_SEED, perfect=True) if perfect else fc
        pol = MPCPolicy(model, f_use, fan_penalty=chosen, **kw)
        if binary:
            pol = _binarise(pol)
        s = run(label, pol, "ablation")
        rows[-1]["ablation"] = label
        print(f"  {label:28s} {s['degree_minutes_over_26']:8.0f} degree-minutes  "
              f"fan {s['fan_wh']:6.0f} Wh")

    d = pd.DataFrame(rows)
    d.to_csv("results/tables/step17_controller.csv", index=False)
    print("\n  wrote results/tables/step17_controller.csv")

    fullrow = d[d.label == "full controller"].iloc[0]
    rulerow = d[d.label == "threshold rule"].iloc[0]
    full = fullrow.degree_minutes_over_26
    rule = rulerow.degree_minutes_over_26
    sealed = d[d.label == "sealed"].iloc[0].degree_minutes_over_26
    print(f"\n  full controller {full:.0f} degree-minutes on "
          f"{fullrow.fan_wh:.0f} Wh of fan electricity")
    print(f"  threshold rule  {rule:.0f} degree-minutes on "
          f"{rulerow.fan_wh:.0f} Wh")
    print(f"  sealed room     {sealed:.0f} degree-minutes on 0 Wh")
    d_comfort = (full / rule - 1) * 100
    d_fan = (1 - fullrow.fan_wh / max(rulerow.fan_wh, 1e-9)) * 100
    print(f"\n  So against the threshold rule the controller uses {d_fan:.0f}% "
          f"LESS fan electricity for a")
    print(f"  comfort difference of {abs(d_comfort):.1f}% "
          f"({'better' if d_comfort < 0 else 'worse'}). The gain here is energy, "
          f"not temperature:")
    print(f"  both keep the room in much the same place, and one does it on a "
          f"fraction of the power.")

    checks = [
        ("the sweep ran at every penalty", len(sw) == len(FAN_PENALTIES)),
        ("a fan penalty was chosen", CHOICE_PATH.exists()),
        ("all ablations ran", int((d.group == "ablation").sum()) == len(abl)),
        ("the controller matches the threshold rule on comfort",
         full <= rule * 1.03),
        ("and does so on much less fan electricity",
         fullrow.fan_wh <= rulerow.fan_wh * 0.6),
        ("the controller beats a sealed room by a wide margin", full < 0.5 * sealed),
        ("a longer horizon helps",
         full <= d[d.label == "very short horizon (1 h)"].iloc[0].degree_minutes_over_26),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(c[1] for c in checks)
    print(f"\nSTEP 17 DONE-WHEN: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


def _binarise(pol):
    """Wrap a policy so the window is only ever fully open or fully shut.

    The rounded position is written back onto the policy as well as returned.
    Without that the controller's own record of where the window is would
    disagree with where the simulator actually put it, which would corrupt the
    movement penalty and make the ablation measure something other than the
    loss of partial opening.
    """
    inner_act = pol.act

    def act(row, history):
        u, f = inner_act(row, history)
        u = 1.0 if u >= 0.5 else 0.0
        f = f if u > 0 else 0.0
        pol.u, pol.f = u, f
        return u, f

    pol.act = act
    pol.name = "mpc_binary"
    return pol


if __name__ == "__main__":
    raise SystemExit(main())
