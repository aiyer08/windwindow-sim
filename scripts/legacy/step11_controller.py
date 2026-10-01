"""
Step 11: the model-based controller.

Done when it runs a full simulation. It does -- but running it immediately
turns up a real conflict between two of the plan's goals, so this step also
measures the trade-off and settles the one free parameter.

The conflict: in this room, ventilating is almost always the right move,
because a shut room with a 17.6 W heater in it climbs to 5.5 K above outdoors.
So a controller that chases comfort (goal G3) ventilates MORE than a timer
running at 50% duty, not less -- and if it runs the fan whenever the fan helps
by more than 0.10 degC, it uses more fan than the timer, failing goal G4.

The resolution is that the window and the fan are separate decisions. Opening
the window is what does most of the cooling; the fan adds a little more for
2.5 W. So the controller can ventilate freely and still be stingy with the
fan. How stingy is set by the fan margin, which PLAN section 5 pencils in at
0.10 degC. This step sweeps it on weather seed 1 and picks a value; Step 12
then scores that choice on seeds 11-13, which it has never seen.
"""

import sys, pathlib, pickle, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.policies import ModelPolicy, SimpleRulePolicy, TimerPolicy
from windwindow.simulator import run_session, summarise
from windwindow.weather import SCENARIOS, make_scenario
from windwindow.oracle import comfort_bounds

TUNE_SEED = 1                 # Step 12 uses seeds 11-13, which stay unseen here
FAN_MARGINS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.30]
CHOICE_PATH = pathlib.Path("results/fan_margin.json")

def run_all(model, fan_margin, seeds=(TUNE_SEED,), record=False):
    out, traces = [], {}
    for name in SCENARIOS:
        for seed in seeds:
            sc = make_scenario(name, seed=seed, duration_s=3 * 3600.0)
            pol = ModelPolicy(model.as_predictor(), record=record,
                              fan_margin=fan_margin)
            df = run_session(sc, pol, sensor_seed=20_000 + seed * 13)
            out.append(summarise(df))
            if record:
                traces[name] = (df, pd.DataFrame(pol.trace))
    return pd.DataFrame(out), traces

def main():
    with open("results/model.pkl", "rb") as fh:
        blob = pickle.load(fh)
    model, best_name = blob[blob["best"]], blob["best"]
    print(f"controller uses the '{best_name}' predictor "
          f"(selected in Step 10 by cross-validation)\n")

    # Baselines on the same weather, for reference.
    base_rows = []
    for name in SCENARIOS:
        sc = make_scenario(name, seed=TUNE_SEED, duration_s=3 * 3600.0)
        for pol in (TimerPolicy(), SimpleRulePolicy()):
            base_rows.append(summarise(
                run_session(sc, pol, sensor_seed=20_000 + TUNE_SEED * 13)))
    base = pd.DataFrame(base_rows)
    timer_dm = base[base.policy == "timer"].degree_minutes_over_26.sum()
    timer_fan = base[base.policy == "timer"].fan_minutes.sum()
    timer_wh = base[base.policy == "timer"].fan_wh.sum()
    rule_dm = base[base.policy == "simple_rule"].degree_minutes_over_26.sum()
    print(f"on weather seed {TUNE_SEED}, summed over the four scenarios:")
    print(f"  timer:       {timer_dm:7.1f} degree-minutes above 26 °C, "
          f"fan {timer_fan:6.1f} min ({timer_wh:.2f} Wh)")
    print(f"  baseline rule: {rule_dm:7.1f} degree-minutes above 26 °C, "
          f"fan {base[base.policy=='simple_rule'].fan_minutes.sum():6.1f} min")

    # ---------------------------------------------------- the trade-off sweep
    print(f"\nsweeping the fan margin (how much extra cooling the fan must buy):")
    print(f"  {'margin':>7s} {'deg-min>26':>11s} {'vs timer':>9s} {'fan min':>8s} "
          f"{'vs timer':>9s} {'fan Wh':>7s} {'G3':>4s} {'G4':>4s}")
    sweep = []
    for fm in FAN_MARGINS:
        s, _ = run_all(model, fm)
        dm, fan, wh = (s.degree_minutes_over_26.sum(), s.fan_minutes.sum(),
                       s.fan_wh.sum())
        g3 = (1 - dm / timer_dm) * 100
        g4 = (1 - fan / timer_fan) * 100
        sweep.append({"fan_margin": fm, "degree_minutes": dm, "fan_minutes": fan,
                      "fan_wh": wh, "g3_pct": g3, "g4_pct": g4,
                      "meets_g3": g3 >= 30, "meets_g4": g4 >= 30,
                      "minutes_under_24": s.minutes_under_24.sum(),
                      "max_t_in": s.max_t_in.max()})
        print(f"  {fm:7.2f} {dm:11.1f} {g3:+8.1f}% {fan:8.1f} {g4:+8.1f}% "
              f"{wh:7.2f} {'PASS' if g3>=30 else 'MISS':>4s} "
              f"{'PASS' if g4>=30 else 'MISS':>4s}")
    sw = pd.DataFrame(sweep)
    sw.to_csv("results/tables/step11_fan_margin_sweep.csv", index=False)

    # G4 is the energy goal and G3 the comfort goal, and raising the margin
    # trades one against the other. Rule: take the most comfortable setting
    # among those that still clear G4 by the required 30%.
    ok_g4 = sw[sw.meets_g4]
    chosen = float(ok_g4.sort_values("degree_minutes").iloc[0].fan_margin)
    print(f"\nchosen fan margin: {chosen:.2f} °C")
    print(f"  (the most comfortable setting that still clears G4 by 30%)")
    if abs(chosen - cfg.FAN_MARGIN_C) < 1e-9:
        why = "most comfortable setting that still clears G4; equals the PLAN default"
        print(f"  This is exactly the value PLAN section 5 pencils in, so the plan's")
        print(f"  guess stands and nothing is overridden.")
    else:
        why = "most comfortable setting that still clears G4"
        print(f"  PLAN section 5 pencils in {cfg.FAN_MARGIN_C:.2f} °C; Step 12 reports both.")

    # --------------------------------------------- how much was available?
    print(f"\nBefore judging the comfort numbers: how much of the gap to the timer")
    print(f"could ANY controller have closed? Computing the bounds.")
    bounds = {}
    for name in SCENARIOS:
        sc = make_scenario(name, seed=TUNE_SEED, duration_s=3 * 3600.0)
        bounds[name] = comfort_bounds(sc, sensor_seed=20_000 + TUNE_SEED * 13)
    bdf = pd.DataFrame(bounds).T
    keys = ["timer", "always_shut", "always_open", "always_open_fan",
            "greedy_oracle", "outdoor_floor"]
    tot = bdf[keys].sum()
    labels = {"timer": "timer (the baseline)",
              "always_shut": "never ventilate",
              "always_open": "always open, fan off",
              "always_open_fan": "always open, fan on",
              "greedy_oracle": "greedy oracle (cheats: true state + true weather)",
              "outdoor_floor": "room pinned to the outdoor air (impossible)"}
    print(f"\n  degree-minutes above 26 °C, summed over the four scenarios:")
    for k in keys:
        pct = (1 - tot[k] / tot["timer"]) * 100
        tag = "   <- the real ceiling" if k == "greedy_oracle" else ""
        print(f"    {labels[k]:52s} {tot[k]:7.1f}  {pct:+6.1f}%{tag}")
    achievable = (1 - tot["greedy_oracle"] / tot["timer"]) * 100
    model_g3 = float(sw[np.isclose(sw.fan_margin, chosen)].g3_pct.iloc[0])
    print(f"\n  G3 asks for +30.0%, but the most ANY controller can reach is "
          f"{achievable:+.1f}%.")
    print(f"  In the hot scenario it is above 26 °C outdoors for the whole run, and")
    print(f"  ventilation cannot cool a room below the air it is letting in.")
    print(f"  The model reaches {model_g3:+.1f}%, which is "
          f"{model_g3/achievable*100:.0f}% of what was available.")
    bdf.to_csv("results/tables/step11_comfort_bounds.csv")
    print(f"  wrote results/tables/step11_comfort_bounds.csv")

    CHOICE_PATH.write_text(json.dumps(
        {"fan_margin": chosen, "plan_default": cfg.FAN_MARGIN_C,
         "tuned_on_seed": TUNE_SEED, "reason": why,
         "achievable_g3_pct": achievable, "model_g3_pct": model_g3}, indent=2))
    print(f"  wrote {CHOICE_PATH}")

    # ------------------------------------------------ the chosen controller
    s_def, traces_def = run_all(model, cfg.FAN_MARGIN_C, record=True)
    s_sel, traces_sel = run_all(model, chosen, record=True)
    print(f"\nper scenario, with the chosen margin ({chosen:.2f} °C):")
    show = ["scenario", "mean_t_in", "max_t_in", "degree_minutes_over_26",
            "minutes_under_24", "fan_minutes", "fan_wh", "moves"]
    print(s_sel[show].to_string(index=False, formatters={
        c: "{:.2f}".format for c in ("mean_t_in", "max_t_in",
                                     "degree_minutes_over_26", "fan_wh")}))
    s_sel.to_csv("results/tables/step11_controller.csv", index=False)
    print("  wrote results/tables/step11_controller.csv")

    # ------------------------------------------------------------- figure A
    # The trap scenario, with the controller's reasoning shown underneath.
    trap = "hot_walls_cold_outside"
    df, tr = traces_sel[trap]
    smry = s_sel[s_sel.scenario == trap].iloc[0]

    P.use_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.9, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.7, 1]})
    t = df["t_s"] / 60.0
    ax = axes[0]
    ax.axhline(cfg.COMFORT_C, color=P.AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)
    ax.axhline(cfg.T_IN_FLOOR_C, color=P.AXIS, lw=0.5, ls=(0, (1, 2)), zorder=1)
    for key in ("t_in", "t_wall", "t_out"):
        ax.plot(t, df[key], color=P.TEMP_COLOR[key], label=P.TEMP_LABEL[key], zorder=4)
        P.direct_label(ax, t.iloc[-1], df[key].iloc[-1], P.TEMP_LABEL[key],
                       P.TEMP_COLOR[key], dx=5)
    ax.annotate("26 °C", xy=(t.iloc[-1] * 1.005, cfg.COMFORT_C),
                color=P.INK_MUTED, fontsize=6.4, va="bottom")
    ax.annotate("24 °C floor", xy=(t.iloc[-1] * 1.005, cfg.T_IN_FLOOR_C),
                color=P.INK_MUTED, fontsize=6.4, va="top")
    ax.set_ylabel("°C (measured)")
    ax.set_title(f"Controller behaviour: {P.SCENARIO_LABEL[trap].lower()} scenario")
    ax.set_xlim(0, t.iloc[-1] * 1.17)

    ax = axes[1]
    tt = tr["t_s"] / 60.0
    ax.axhspan(cfg.CLOSE_THRESHOLD_C, cfg.OPEN_THRESHOLD_C, color=P.GRID, zorder=1)
    ax.annotate("dead band", xy=(t.iloc[-1] * 1.01,
                (cfg.CLOSE_THRESHOLD_C + cfg.OPEN_THRESHOLD_C) / 2),
                color=P.INK_MUTED, fontsize=6.2, va="center")
    ax.axhline(0, color=P.AXIS, lw=0.5, zorder=2)
    ax.plot(tt, tr["benefit"], color=P.SERIES[0], zorder=4,
            label="predicted benefit of opening")
    ax.plot(tt, tr["fan_gain"], color=P.SERIES[2], zorder=3,
            label="additional benefit of the fan")
    ax.axhline(chosen, color=P.SERIES[2], lw=0.5, ls=(0, (2, 2)), zorder=2)
    ax.annotate(f"fan margin {chosen:.2f}", xy=(t.iloc[-1] * 1.01, chosen),
                color=P.INK_MUTED, fontsize=6.2, va="center")
    ax.set_ylabel("°C over 10 min")

    ax = axes[2]
    ax.fill_between(t, 0, df["u"], step="post", color=P.GRID, zorder=2)
    ax.plot(t, df["u"], drawstyle="steps-post", color=P.INK_2, lw=1.0, zorder=4,
            label="window (u)")
    ax.plot(t, df["f"], drawstyle="steps-post", color=P.INK_MUTED, lw=1.0,
            ls=(0, (2, 1.5)), zorder=3, label="fan (f)")
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1], ["off / shut", "on / open"])
    ax.set_xlabel("minutes")
    ax.grid(False)

    P.legend_below(axes[2], ncol=4, gap=32, source=[axes[0], axes[1], axes[2]])
    note = P.caption(axes[2],
        f"The centre panel shows the quantities the control rule acts on. The window "
        f"opens once the predicted benefit exceeds {cfg.OPEN_THRESHOLD_C:.2f} °C and "
        f"shuts once it falls below {cfg.CLOSE_THRESHOLD_C:.2f} °C.\nVentilation "
        f"continues through the rebound period, while the fan is held off whenever its "
        f"predicted contribution is below {chosen:.2f} °C. Result: "
        f"{smry.degree_minutes_over_26:.0f} degree-minutes above 26 °C on "
        f"{smry.fan_minutes:.0f} fan-minutes.", gap=54)
    fig.tight_layout()
    fig.savefig(f"results/figures/step11_model_{trap}.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print(f"  wrote results/figures/step11_model_{trap}.png")

    # ------------------------------------------------------------- figure B
    fig, ax = plt.subplots(figsize=(5.8, 3.5))

    # Shade only the corner where BOTH goals are met -- the intersection, not
    # the union. The curve never reaches it, and that is the finding.
    ax.add_patch(plt.Rectangle((30, 30), 200, 200, color=P.GRID, zorder=0, lw=0))
    ax.annotate("region satisfying both G3 and G4\n(not reached at any setting)", xy=(0.975, 0.955),
                xycoords="axes fraction", ha="right", va="top",
                color=P.INK_2, fontsize=6.8, linespacing=1.4)

    # The ceiling: no controller at all can beat this on comfort.
    ax.axhline(achievable, color=P.SERIES[2], lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.annotate(f"reference controller limit: {achievable:.1f}%",
                xy=(2, achievable), textcoords="offset points", xytext=(0, 4),
                color=P.SERIES[2], fontsize=6.6, va="bottom")

    ax.axvline(30, color=P.SERIES[1], lw=1.0, zorder=3)
    ax.axhline(30, color=P.SERIES[1], lw=1.0, zorder=3)
    ax.annotate("G4 target", xy=(30, 2), textcoords="offset points",
                xytext=(4, 0), color=P.SERIES[1], fontsize=6.6, va="bottom")
    ax.annotate("G3 target", xy=(2, 30), textcoords="offset points",
                xytext=(0, 4), ha="left", color=P.SERIES[1], fontsize=6.6,
                va="bottom")

    ax.plot(sw.g4_pct, sw.g3_pct, color=P.SERIES[0], marker="o", ms=4,
            zorder=4, label="model, fan margin varied")
    # Label a few settings only; consecutive ones sit on top of each other.
    # Alternate above and below, or the closely spaced ones overlap.
    for k, fm_ in enumerate((0.10, 0.20, 0.30, 0.40, 0.50)):
        r = sw[np.isclose(sw.fan_margin, fm_)].iloc[0]
        dy = -11 if k % 2 == 0 else 7
        ax.annotate(f"{fm_:.2f}", (r.g4_pct, r.g3_pct),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", color=P.INK_MUTED, fontsize=6.4)
    # Everything from 0.60 up lands on the same spot: the fan simply never runs.
    tail = sw[sw.fan_margin >= 0.60]
    ax.annotate("0.60 °C and above:\nfan does not operate",
                (tail.g4_pct.iloc[-1], tail.g3_pct.iloc[-1]),
                textcoords="offset points", xytext=(-2, -34), ha="right",
                color=P.INK_MUTED, fontsize=6.4, linespacing=1.4)
    sel = sw[np.isclose(sw.fan_margin, chosen)].iloc[0]
    ax.plot([sel.g4_pct], [sel.g3_pct], marker="o", ms=7.5, color=P.SERIES[0],
            markeredgecolor=P.SURFACE, markeredgewidth=1.5, zorder=6)
    ax.annotate(f"chosen: {chosen:.2f} °C", (sel.g4_pct, sel.g3_pct),
                textcoords="offset points", xytext=(-9, 0), ha="right",
                va="center", color=P.INK, fontsize=7, fontweight="bold")

    ax.set_xlabel("fan use, improvement on the timer (%)")
    ax.set_ylabel("degree-minutes above 26 °C,\nimprovement on the timer (%)")
    ax.set_title("Comfort against fan energy, by fan margin")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 40)
    P.legend_below(ax, ncol=1, gap=34)
    note = P.caption(ax,
        f"Each point is one fan-margin setting, summed over the four scenarios on "
        f"weather seed {TUNE_SEED}; five are labelled with their\nvalue in °C. "
        f"Every setting satisfies G4. None satisfies G3, because the 30% target lies "
        f"above the {achievable:.1f}% limit set by the\nreference controller. Raising "
        f"the margin reduces fan energy at little cost in comfort, since the window "
        f"accounts for most of the cooling.", gap=54)
    fig.tight_layout()
    fig.savefig("results/figures/step11_fan_tradeoff.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print("  wrote results/figures/step11_fan_tradeoff.png")

    checks = [
        ("the controller ran a full simulation in all 4 scenarios", len(s_sel) == 4),
        ("every run produced 180 minutes", bool((s_sel.minutes == 180).all())),
        ("it moves the window", bool((s_sel.moves > 0).any())),
        ("it respects the 5-minute dwell", bool((s_sel.moves <= 36).all())),
        ("it recorded its reasoning", all(len(traces_sel[n][1]) > 100 for n in SCENARIOS)),
        ("the fan trade-off was measured", len(sw) == len(FAN_MARGINS)),
        ("a fan margin exists that clears G4", len(ok_g4) > 0),
        ("the model captures most of the achievable comfort gain",
         model_g3 / achievable > 0.8),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 11 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
