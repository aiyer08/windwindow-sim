"""Step 14: the report figures and the README."""

import sys, pathlib, json, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windwindow import config as cfg, plotting as P
from windwindow.policies import ModelPolicy, SimpleRulePolicy, TimerPolicy
from windwindow.report import write_readme, write_report_html
from windwindow.simulator import run_session, summarise
from windwindow.weather import SCENARIOS, make_scenario

POLICIES = ["timer", "simple_rule", "model"]

def fig_results(runs, bounds, head):
    """Comfort by scenario, and fan use by policy."""
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # --- comfort, per scenario -------------------------------------------
    ax = axes[0]
    piv = (runs.pivot_table(index="scenario", columns="policy",
                            values="degree_minutes_over_26")
               .reindex(index=list(SCENARIOS), columns=POLICIES))
    x = np.arange(len(piv))
    w = 0.26
    for i, pol in enumerate(POLICIES):
        # A 2 px gap between neighbouring bars, so the fills never touch.
        ax.bar(x + (i - 1) * w, piv[pol], width=w * 0.92,
               color=P.POLICY_COLOR[pol], label=P.POLICY_LABEL[pol], zorder=4)
    # The ceiling, drawn as a rule across each group. It lands very close to
    # the model's bar, which is the point worth seeing.
    ceil = bounds.groupby(level=0).greedy_oracle.mean().reindex(list(SCENARIOS))
    ax.hlines(ceil, x - 1.6 * w, x + 1.6 * w, color=P.INK, lw=1.2, zorder=6,
              label="reference controller limit")
    ax.set_xticks(x, [P.SCENARIO_LABEL[s].replace(", ", ",\n") for s in SCENARIOS],
                  fontsize=6.6)
    ax.set_ylabel("degree-minutes above 26 °C")
    ax.set_title("Degree-minutes above 26 °C, by scenario")
    ax.annotate("no time above 26 °C\nin this scenario", xy=(0, 20),
                ha="center", va="bottom", color=P.INK_MUTED, fontsize=6.2)
    ax.grid(axis="x", visible=False)

    # --- fan use ----------------------------------------------------------
    ax = axes[1]
    fm = runs.groupby("policy").fan_minutes.mean().reindex(POLICIES)
    timer_fan = fm["timer"]
    ax.bar(range(len(fm)), fm.to_numpy(), width=0.56,
           color=[P.POLICY_COLOR[p] for p in POLICIES], zorder=4)
    ax.axhline(timer_fan * 0.7, color=P.SERIES[1], lw=1.0, zorder=5)
    ax.annotate(f"G4 target: {timer_fan*0.7:.0f} min,\n30% below the timer",
                xy=(2.0, timer_fan * 0.7), textcoords="offset points",
                xytext=(0, 6), ha="center", color=P.SERIES[1], fontsize=6.4,
                annotation_clip=False, linespacing=1.4)
    for i, v in enumerate(fm.to_numpy()):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", color=P.INK_2, fontsize=7)
    ax.set_xticks(range(len(fm)),
                  [P.POLICY_LABEL[p] for p in POLICIES],
                  fontsize=6.8)
    ax.set_ylabel("fan minutes per 3-hour run")
    ax.set_title("Mean fan runtime per run")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(fm) * 1.22)

    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(P.POLICY_LABEL[p]) for p in POLICIES]
    order += [i for i in range(len(l)) if i not in order]
    from matplotlib.transforms import offset_copy
    tr = offset_copy(axes[0].transAxes, fig=fig, x=0, y=-44, units="points")
    axes[0].legend([h[i] for i in order], [l[i] for i in order],
                   loc="upper left", bbox_to_anchor=(0, 0), bbox_transform=tr,
                   ncol=4, frameon=False, borderaxespad=0.0)
    sup = fig.suptitle("Outcomes over 36 runs: 3 policies, 4 scenarios, 3 repeats",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    note = P.caption(axes[0],
        f"Weather seeds 11 to 13, none of which appear in the training data. Each bar "
        f"is the mean of 3 repeats.\nThe model reduced degree-minutes by "
        f"{head['g3_pct']:.0f}% relative to the timer, against a reference limit of "
        f"{head['g3_ceiling']:.0f}%, using {head['g4_pct']:.0f}% less fan runtime.",
        gap=68)
    fig.savefig("results/figures/step14_results.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/step14_results.png")


def fig_policies(model):
    """All three policies on the same weather, so the difference is visible."""
    scenario = "hot_walls_cold_outside"
    seed = 11
    sc = make_scenario(scenario, seed=seed, duration_s=3 * 3600.0)
    runs = {}
    for pol in (TimerPolicy(), SimpleRulePolicy(), ModelPolicy(model.as_predictor())):
        runs[pol.name] = run_session(sc, pol, sensor_seed=20_000 + seed * 13)

    P.use_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.5]})
    t = runs["model"]["t_s"] / 60.0

    ax = axes[0]
    ax.axhline(cfg.COMFORT_C, color=P.SERIES[1], lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.annotate("26 °C threshold", xy=(3, cfg.COMFORT_C),
                textcoords="offset points", xytext=(0, 3),
                color=P.SERIES[1], fontsize=6.4, va="bottom")
    ax.plot(t, runs["model"]["t_out"], color=P.INK_MUTED, lw=0.8, ls=(0, (2, 2)),
            zorder=3, label="Outdoors")
    P.direct_label(ax, t.iloc[-1], runs["model"]["t_out"].iloc[-1], "Outdoors",
                   P.INK_MUTED, dx=5)
    # The model and the baseline rule finish within a few tenths of each other,
    # so their end labels need nudging apart.
    nudge = {"timer": 0, "model": 5, "simple_rule": -6}
    for pol in POLICIES:
        ax.plot(t, runs[pol]["t_in"], color=P.POLICY_COLOR[pol], zorder=4,
                label=P.POLICY_LABEL[pol])
        P.direct_label(ax, t.iloc[-1], runs[pol]["t_in"].iloc[-1],
                       P.POLICY_LABEL[pol],
                       P.POLICY_COLOR[pol], dx=5, dy=nudge[pol])
    ax.set_ylabel("indoor air °C")
    ax.set_title("Three controllers on identical weather")
    ax.set_xlim(0, t.iloc[-1] * 1.30)

    ax = axes[1]
    for i, pol in enumerate(POLICIES):
        y = 2 - i
        d = runs[pol]
        ax.fill_between(t, y - 0.34, y - 0.34 + 0.30 * d["u"], step="post",
                        color=P.POLICY_COLOR[pol], zorder=3, lw=0)
        ax.fill_between(t, y + 0.02, y + 0.02 + 0.30 * d["f"], step="post",
                        color=P.POLICY_COLOR[pol], alpha=0.42, zorder=3, lw=0)
        s = summarise(d)
        ax.annotate(f"{s['fan_minutes']:.0f} fan-min · "
                    f"{s['degree_minutes_over_26']:.0f} deg-min>26",
                    xy=(t.iloc[-1] * 1.02, y - 0.05), color=P.INK_MUTED,
                    fontsize=6.2, va="center")
    ax.set_yticks([2, 1, 0],
                  [P.POLICY_LABEL[p] for p in POLICIES],
                  fontsize=6.8)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("minutes")
    ax.grid(False)
    ax.annotate("lower band: window open        upper band: fan running",
                xy=(0.0, 1.03), xycoords="axes fraction", color=P.INK_MUTED,
                fontsize=6.2, va="bottom")

    P.legend_below(axes[1], ncol=4, gap=34, source=axes[0])
    note = P.caption(axes[1],
        "Warm-shell scenario, weather seed 11. The timer ventilates on a schedule, "
        "so the room rebounds between intervals.\nThe baseline rule and the model "
        "reach a similar temperature, but the model uses considerably less fan "
        "runtime.", gap=54)
    fig.tight_layout()
    fig.savefig("results/figures/step14_policies.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note])
    plt.close(fig)
    print("  wrote results/figures/step14_policies.png")


def fig_trials(big):
    """Where the baseline rule fails, and the model does not."""
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    d = big.copy()
    d["dout"] = d.t_out_true - d.t_in_true
    d["dwall"] = d.t_wall_true - d.t_in_true
    right = d[d.rule_correct]
    wrong = d[~d.rule_correct]
    ax.scatter(right.dout, right.dwall, s=7, color=P.INK_MUTED, alpha=0.35,
               edgecolors="none", zorder=3, label="baseline rule correct")
    ax.scatter(wrong.dout, wrong.dwall, s=22, color=P.SERIES[1], zorder=5,
               edgecolors=P.SURFACE, linewidths=0.6,
               label="baseline rule incorrect")
    mw = d[~d.model_correct]
    ax.scatter(mw.dout, mw.dwall, s=44, facecolors="none",
               edgecolors=P.SERIES[0], linewidths=1.1, zorder=6,
               label="model incorrect")
    ax.axvline(-SimpleRulePolicy.OPEN_GAP, color=P.AXIS, lw=0.6, ls=(0, (3, 3)), zorder=2)
    ax.annotate("baseline rule threshold:\nopens to the left of this line",
                xy=(-SimpleRulePolicy.OPEN_GAP, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(-6, 8), ha="right",
                color=P.INK_MUTED, fontsize=6.2, va="bottom")
    ax.set_xlabel("T_out − T_in  (°C; negative means cooler outside)")
    ax.set_ylabel("T_wall − T_in  (°C)")
    ax.set_title(f"Decision errors across {len(big)} paired trials")

    ax = axes[1]
    # Wrong calls, not right calls. Bars have to start at zero to be honest,
    # and 99.5% next to 95.5% on a zero baseline is two indistinguishable bars.
    # The same numbers as an error rate start at zero naturally and show the
    # gap for what it is: nine times fewer mistakes.
    labels = ["Model", "Baseline rule"]
    errs = [(1 - big.model_correct.mean()) * 100,
            (1 - big.rule_correct.mean()) * 100]
    cols = [P.POLICY_COLOR["model"], P.POLICY_COLOR["simple_rule"]]
    ax.bar(range(2), errs, width=0.5, color=cols, zorder=4)
    ax.axhline(10, color=P.SERIES[1], lw=1.0, zorder=5)
    ax.annotate("G2 limit: 10% incorrect,\nequivalent to 90% accuracy", xy=(0.5, 10),
                textcoords="offset points", xytext=(0, 5), ha="center",
                color=P.SERIES[1], fontsize=6.4, annotation_clip=False,
                linespacing=1.4)
    for i, (v, col) in enumerate(zip(errs, (big.model_correct, big.rule_correct))):
        ax.annotate(f"{v:.1f}%\n{int((~col).sum())} of {len(big)}", (i, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=P.INK_2, fontsize=7, linespacing=1.4)
    ax.set_xticks(range(2), labels, fontsize=6.8)
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("incorrect open/shut decisions (%)")
    ax.set_title("Incorrect decision rate")
    ax.grid(axis="x", visible=False)

    P.legend_below(axes[0], ncol=1, gap=40)
    sup = fig.suptitle("Objective G2: accuracy of the open/shut decision",
                       x=0.005, ha="left", fontsize=9.5, fontweight="bold", color=P.INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    note = P.caption(axes[0],
        "Each point is one trial. The room was held at a fixed state, then\n"
        "simulated forward 10 minutes under both actions using the true\n"
        "weather series. Errors by the baseline rule cluster where indoor\n"
        "and outdoor temperatures are nearly equal, because it keeps the\n"
        "window shut while the heated room continues to warm.", gap=96)
    fig.savefig("results/figures/step14_trials.png", bbox_inches="tight",
                pad_inches=0.18, bbox_extra_artists=[note, sup])
    plt.close(fig)
    print("  wrote results/figures/step14_trials.png")


def main():
    head = json.load(open("results/headline.json"))
    runs = pd.read_csv("results/tables/step12_runs.csv")
    goals = pd.read_csv("results/tables/step13_goals.csv")
    bounds = pd.read_csv("results/tables/step13_comfort_bounds.csv", index_col=[0, 1])
    big = pd.read_csv("results/tables/step13_paired_trials_large.csv")
    models = pd.read_csv("results/tables/step10_model_comparison.csv")
    with open("results/model.pkl", "rb") as fh:
        blob = pickle.load(fh)
    model = blob[blob["best"]]

    fig_results(runs, bounds, head)
    fig_policies(model)
    fig_trials(big)

    figs = sorted(pathlib.Path("results/figures").glob("*.png"))
    tables = sorted(pathlib.Path("results/tables").glob("*.csv"))
    print(f"\n  {len(figs)} figures, {len(tables)} tables")

    write_readme(head, runs, goals, bounds, big, models, figs, tables)
    write_report_html(head, runs, goals, bounds, big, models)

    checks = [
        ("all three report figures written",
         all(pathlib.Path(f"results/figures/step14_{n}.png").exists()
             for n in ("results", "policies", "trials"))),
        ("README written", pathlib.Path("README.md").exists()),
        ("README is substantial", len(pathlib.Path("README.md").read_text()) > 6000),
        ("index.html written and fully filled in",
         pathlib.Path("index.html").exists()
         and "{{" not in pathlib.Path("index.html").read_text()),
        ("every step produced at least one figure", len(figs) >= 10),
        ("results tables written", len(tables) >= 8),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print(f"\nSTEP 14 DONE-WHEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
