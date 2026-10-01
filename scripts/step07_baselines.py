"""Step 7: do the timer and the baseline rule both run and produce results?"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from windwindow import config as cfg, plotting as P
from windwindow.simulator import run_session, summarise
from windwindow.policies import TimerPolicy, SimpleRulePolicy
from windwindow.weather import SCENARIOS, make_scenario

def main():
    rows, sessions = [], {}
    for name in SCENARIOS:
        sc = make_scenario(name, seed=1, duration_s=3 * 3600.0)
        for policy in (TimerPolicy(), SimpleRulePolicy()):
            df = run_session(sc, policy)
            rows.append(summarise(df))
            sessions[(name, policy.name)] = df

    s = pd.DataFrame(rows)
    show = ["scenario", "policy", "mean_t_in", "max_t_in", "degree_minutes_over_26",
            "minutes_over_26", "minutes_under_24", "fan_minutes", "fan_wh", "moves"]
    print(s[show].to_string(index=False,
          formatters={c: "{:.2f}".format for c in
                      ("mean_t_in", "max_t_in", "degree_minutes_over_26", "fan_wh")}))

    pathlib.Path("results/tables").mkdir(parents=True, exist_ok=True)
    s.to_csv("results/tables/step07_baselines.csv", index=False)
    print("\n  wrote results/tables/step07_baselines.csv")

    # Two figures: one run per policy in the trap scenario, where the simple
    # rule's weakness should be visible.
    trap = "hot_walls_cold_outside"
    for pol in ("timer", "simple_rule"):
        df = sessions[(trap, pol)]
        smry = s[(s.scenario == trap) & (s.policy == pol)].iloc[0]
        P.plot_session(
            df, f"Step 7  ·  {P.POLICY_LABEL[pol]} in the "
                f"'{P.SCENARIO_LABEL[trap].lower()}' scenario",
            note=f"{smry.degree_minutes_over_26:.0f} degree-minutes over 26 °C  ·  "
                 f"fan {smry.fan_minutes:.0f} min ({smry.fan_wh:.2f} Wh)  ·  "
                 f"{smry.moves:.0f} window moves",
            path=f"results/figures/step07_{pol}_{trap}.png", floor=cfg.T_IN_FLOOR_C)

    checks = [
        ("timer ran in all 4 scenarios", (s.policy == "timer").sum() == 4),
        ("simple rule ran in all 4 scenarios", (s.policy == "simple_rule").sum() == 4),
        ("every session produced 180 minutes", bool((s.minutes == 180).all())),
        ("the two policies behave differently",
         float((s[s.policy == "timer"].fan_minutes.to_numpy()
                - s[s.policy == "simple_rule"].fan_minutes.to_numpy()).__abs__().max()) > 5),
        ("the baseline rule reacts to the weather (fan time varies by scenario)",
         float(s[s.policy == "simple_rule"].fan_minutes.std()) > 5),
        ("the timer ignores the weather (same fan time everywhere)",
         float(s[s.policy == "timer"].fan_minutes.std()) < 1e-6),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 7 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
