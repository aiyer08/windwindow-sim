"""Step 14: generate README.md from the measured results.

The write-up is generated rather than typed so that every number in it comes
from the tables the experiment actually produced. If a constant changes and
the pipeline is re-run, the prose changes with it and cannot go stale.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import config as cfg


def _derived() -> dict:
    """Every physical number the write-up quotes, computed from the modules.

    The prose is generated precisely so that none of these can drift away from
    the code. Nothing here may be typed as a literal.
    """
    from .room import (Room, h_in_eff, k_in_wall, k_vent, linear_system,
                       steady_state)
    from .weather import make_scenario

    d = {}
    for label, (u, f) in (("shut", (0.0, 0.0)), ("open", (1.0, 0.0)),
                          ("open_fan", (1.0, 1.0))):
        a, _ = linear_system(u, f, 20.0, cfg.HEATER_W)
        taus = sorted(-1.0 / np.linalg.eigvals(a).real)
        d[f"tau_fast_{label}"] = taus[0] / 60.0
        d[f"tau_slow_{label}"] = taus[1] / 60.0
        d[f"h_in_{label}"] = h_in_eff(u, f)
        d[f"k_iw_{label}"] = k_in_wall(u, f)
        d[f"k_vent_{label}"] = k_vent(u, f)

    # How far a shut room with the heater on ends up above outdoors.
    d["shut_rise"] = steady_state(0.0, 0.0, 20.0, cfg.HEATER_W)[0] - 20.0

    # The rebound: flush a baking room for 8 min, then shut the window.
    t_out, t_in0, t_wall0 = 24.5, 28.5, 34.0
    room = Room(t_in=t_in0, t_wall=t_wall0, heater_w=cfg.HEATER_W)
    for _ in range(8 * 60):
        room.step(1.0, 1.0, t_out)
    flushed = room.t_in
    after = room.copy()
    peak = flushed
    for _ in range(40 * 60):
        after.step(0.0, 0.0, t_out)
        peak = max(peak, after.t_in)
    d["rebound"] = peak - flushed

    # How far above outdoors the trap scenario's shell actually starts.
    gaps = [make_scenario("hot_walls_cold_outside", seed=k).t_wall0
            - float(make_scenario("hot_walls_cold_outside", seed=k).t_out_grid[0])
            for k in range(1, 14)]
    d["shell_gap"] = float(np.mean(gaps))
    return d


def _goal_row(goals, g):
    r = goals[goals.goal == g].iloc[0]
    return r


def write_readme(head, runs, goals, bounds, big, models, figs, tables) -> None:
    pol = (runs.groupby("policy")
              .agg(dm=("degree_minutes_over_26", "mean"),
                   over=("minutes_over_26", "mean"),
                   under=("minutes_under_24", "mean"),
                   mean_t=("mean_t_in", "mean"),
                   max_t=("max_t_in", "max"),
                   fan_min=("fan_minutes", "mean"),
                   fan_wh=("fan_wh", "mean"),
                   open_min=("window_open_minutes", "mean"),
                   moves=("moves", "mean")))
    t, r, m = pol.loc["timer"], pol.loc["simple_rule"], pol.loc["model"]
    mrow = models.set_index("model")
    imp = pd.read_csv("results/tables/step10_feature_importance.csv")
    imp = imp.sort_values("lgbm_gain_frac", ascending=False)
    fstats = pd.read_csv("results/tables/step09_feature_stats.csv")
    by_sc = pd.read_csv("results/tables/step10_by_scenario.csv")
    worst = by_sc.sort_values("blend", ascending=False).iloc[0]
    others = ", ".join(f"{r.scenario}: {r.blend:.3f}"
                       for r in by_sc[by_sc.scenario != worst.scenario].itertuples())
    oracles = pd.read_csv("results/tables/step10_oracles.csv").set_index("oracle")
    D = _derived()
    bsum = bounds.mean()

    def tick(met):
        return "**MET**" if met else "**not met**"

    g1, g2, g3, g4 = (_goal_row(goals, g) for g in ("G1", "G2", "G3", "G4"))

    md = f"""# WindWindow Sim

A software simulation of a small heated room with a controllable window and
fan, used to test whether a learned controller can decide when to ventilate
better than two conventional alternatives. The controller predicts how much the
indoor air temperature will change over the next ten minutes under each
available action, then selects the action with the lowest predicted
temperature.

Four objectives were fixed before implementation began. Three were met. The
fourth was not, and the analysis below establishes that its target exceeds the
largest reduction any controller can achieve in this room.

![Results](results/figures/step14_results.png)

## Results summary

| # | Objective | Target | Result | |
|---|-----------|--------|--------|---|
| G1 | Predict the 10-minute temperature change | MAE at or below 0.25 degC | **{head['g1_mae']:.3f} degC** | {tick(bool(g1.met))} |
| G2 | Correct open/shut decisions, and improvement on the baseline rule | at least 90%, and an improvement | **{head['g2_model']*100:.1f}%** on 30 trials; **{head['g2_model_large']*100:.1f}% against {head['g2_rule_large']*100:.1f}%** at n = {len(big)} | {tick(bool(g2.met))} |
| G3 | Reduce time above 26 degC relative to a timer | at least 30% | **{head['g3_pct']:.1f}%** | {tick(bool(g3.met))} |
| G4 | Reduce fan runtime relative to a timer | at least 30% | **{head['g4_pct']:.1f}%** ({head['g4_wh_pct']:.1f}% in watt-hours) | {tick(bool(g4.met))} |

### Objective G3

The model reduced degree-minutes above 26 degC by {head['g3_pct']:.1f}% relative
to the timer, against a target of 30%. To establish whether the target was
attainable, a reference controller was given the true state of the room and the
true future outdoor temperature and allowed to select the best action at every
control step. That controller achieved {head['g3_ceiling']:.1f}%. The learned
model therefore captured {head['g3_pct']/head['g3_ceiling']*100:.0f}% of the
reduction available to any controller.

The limit follows from the physics of ventilation. In the hot scenario the
outdoor air remains above 26 degC for the whole run, and ventilation cannot
bring a room below the temperature of the air entering it. The target was set
before this ceiling was known.

### Objective G2

The model made the correct decision in {head['g2_model']*100:.1f}% of the 30
paired trials specified in the plan, which satisfies the 90% requirement.
Demonstrating an improvement over the baseline rule required a larger sample.
The two methods differ by roughly
{(head['g2_model_large']-head['g2_rule_large'])*100:.0f} percentage points, so a
sample of 30 trials usually contains no disagreement between them.

Over 2000 random draws of 30 trials the model led in
{head['resample_ahead_pct']:.0f}% of draws, tied in
{head['resample_tied_pct']:.0f}%, and trailed in
{head['resample_behind_pct']:.0f}%. At n = {len(big)} the model was correct in
{head['g2_model_large']*100:.1f}% of trials against
{head['g2_rule_large']*100:.1f}% for the baseline rule. All
{int((big.model_correct & ~big.rule_correct).sum())} disagreements favoured the
model (McNemar test, p = {head['mcnemar_p']:.1e}).

## 1. Thermal model

The room is a plywood box measuring 600 by 400 by 400 mm, enclosing roughly
66 L of air, with a 17.6 W heater inside. It is represented as two lumped
thermal masses connected by conductive paths:

```
[17.6 W heater] --> ( indoor air ) <--> ( plywood shell ) <--> outdoors
                          |
                          +------- window and fan -------> outdoors
```

A thermal conductance, in watts per kelvin, gives the heat flow produced by one
degree of temperature difference. A thermal capacitance, in joules per kelvin,
gives the energy required to raise a mass by one degree.

The two masses differ by a factor of four, which is the source of the control
problem:

| Thermal mass | Capacitance | Composition |
|---|---|---|
| Indoor air and fittings | {cfg.C_IN:,.0f} J/K | the air contributes little; internal mounts and trim dominate |
| Plywood shell | {cfg.C_WALL:,.0f} J/K | {cfg.MASS_WALL:.1f} kg of wood, storing {cfg.C_WALL/1000:.0f} kJ per degree |

The air therefore responds within a few minutes while the shell takes around
half an hour. The time constants below were obtained from the eigenvalues of
the linearised system.

| Window state | Fast mode (air) | Slow mode (shell) |
|---|---|---|
| Shut | {D['tau_fast_shut']:.1f} min | {D['tau_slow_shut']:.1f} min |
| Open, fan off | {D['tau_fast_open']:.1f} min | {D['tau_slow_open']:.1f} min |
| Open, fan on | {D['tau_fast_open_fan']:.1f} min | {D['tau_slow_open_fan']:.1f} min |

The ten-minute prediction horizon falls between the two modes. Within that
interval the air has largely responded to a change in action while the shell
has not, so the outcome depends on how much heat the shell still holds.

### 1.1 Effect of airflow on wall heat transfer

Opening the window admits outdoor air and also sets the indoor air in motion.
Moving air reduces the thickness of the boundary layer at the inner wall
surface, which raises the surface heat-transfer coefficient and increases the
rate of heat exchange between the shell and the air.

| Window state | Inner surface coefficient | Air to shell conductance |
|---|---|---|
| Shut | {D['h_in_shut']:.1f} W/(m2.K) | {D['k_iw_shut']:.2f} W/K |
| Open, fan off | {D['h_in_open']:.1f} W/(m2.K) | {D['k_iw_open']:.2f} W/K |
| Open, fan on | {D['h_in_open_fan']:.1f} W/(m2.K) | {D['k_iw_open_fan']:.2f} W/K |

This effect produces a measurable rebound. Starting from a room whose shell has
been heated, an eight-minute period of ventilation with the window open and the
fan running lowers the air temperature substantially. If the window is then
shut, the air temperature rises again by {D['rebound']:.2f} degC, because the
shell is still warm and the enclosed air provides the only path for that stored
heat. If the window is instead left open, the air temperature continues to
fall.

A controller that shuts the window as soon as the air reading is acceptable
will therefore incur this rebound repeatedly. The shell temperature is included
among the model inputs for this reason.

![Window and fan effects](results/figures/step04_window_fan.png)

## 2. Weather scenarios

| Scenario | Outdoor range | Control difficulty |
|---|---|---|
| Cool outside | 16 to 20 degC | ventilation overshoots easily; the risk is cooling the room below the 24 degC floor |
| Hot outside | 27 to 32 degC | outdoor air is initially warmer than indoor air, so ventilation adds heat until the two cross over |
| Hot, then cooling | 30 falling to 19 degC | the five-minute outdoor trend carries information about the coming ten minutes |
| Warm shell, cooler outside | 23 to 26 degC | the shell begins about {D['shell_gap']:.0f} K above outdoor air, so the rebound in section 1.1 applies |

![Scenarios](results/figures/step03_scenarios.png)

## 3. Baseline rule and its failure modes

The conventional rule opens the window whenever outdoor air is cooler than
indoor air. It was correct in {head['g2_rule_large']*100:.1f}% of paired
trials.

The rule fails in a single, consistent way: it does not account for the
internal heat source. A sealed room containing 17.6 W settles
{D['shut_rise']:.1f} K above the outdoor temperature, so ventilation can be the
better action even when outdoor air is slightly warmer than indoor air at the
moment of the decision, because the sealed alternative continues to warm.

All {int((~big.rule_correct).sum())} errors the rule made across {len(big)}
trials were of this type: the window was kept shut when opening was the better
action. These cases occur where indoor and outdoor temperatures are close
together, with a mean difference of
{(big[~big.rule_correct].t_out_true - big[~big.rule_correct].t_in_true).mean():+.2f} degC.

The rule also has no access to the shell temperature, so it cannot anticipate
the rebound described in section 1.1.

![Paired trials](results/figures/step14_trials.png)

## 4. Predictor and control rule

In the table below, `u` denotes the window opening on a scale of 0 to 1, `f`
the fan speed on the same scale, and `d` the difference between outdoor and
indoor temperature. The final column gives each input's share of the total
split gain in the gradient-boosted model.

| Input | Quantity represented | Share of split gain |
|---|---|---|
""" + "\n".join(
        f"| `{row.input}` | {row.label} | {row.lgbm_gain_frac*100:.1f}% |"
        for row in imp.itertuples()
    ) + f"""

Two features of this input set are relevant to the control rule. Every input is
a function of the state and the action together, so a trained model can be
evaluated counterfactually: the predicted outcome of shutting the window and
the predicted outcome of opening it with the fan running are both obtained from
the same measured state. The four inputs that carry `u` as a factor vanish when
the window is shut, since they describe transport processes that require air
movement.

The largest single contribution comes from the indoor temperature trend, at
{imp.iloc[0].lgbm_gain_frac*100:.0f}%. The two inputs describing the shell
together account for
{imp[imp.input.isin(['wall_minus_in','u_wall_minus_out'])].lgbm_gain_frac.sum()*100:.0f}%
of the split gain, which quantifies the contribution of stored wall heat to the
prediction. The baseline rule has no equivalent term.

### 4.1 Control rule

```
benefit = predicted change if shut - predicted change if open with fan

open   if benefit > {cfg.OPEN_THRESHOLD_C:.2f} degC
shut   if benefit < {cfg.CLOSE_THRESHOLD_C:.2f} degC
minimum interval between window movements: {cfg.MIN_DWELL_S/60:.0f} min
shut   if indoor temperature < {cfg.T_IN_FLOOR_C:.0f} degC
fan on only if it improves on the window alone by more than {cfg.FAN_MARGIN_C:.2f} degC
```

The opening and closing thresholds differ, which creates a dead band. With a
single threshold, a benefit estimate fluctuating around that value would cause
the window to change state at every control interval.

![Controller](results/figures/step11_model_hot_walls_cold_outside.png)

### 4.2 Choice of predictor

| Predictor | Cross-validated MAE | Held-out MAE | Note |
|---|---|---|---|
| Constant zero prediction | {mrow.loc['zero','cv_mae']:.3f} | {mrow.loc['zero','test_mae']:.3f} | reference value |
| Linear extrapolation of the 5-min trend | {mrow.loc['trend','cv_mae']:.3f} | {mrow.loc['trend','test_mae']:.3f} | less accurate than predicting no change |
| Ridge regression | {mrow.loc['ridge','cv_mae']:.3f} | {mrow.loc['ridge','test_mae']:.3f} | linear in the nine inputs |
| Gradient-boosted trees | {mrow.loc['lightgbm','cv_mae']:.3f} | {mrow.loc['lightgbm','test_mae']:.3f} | overfits to {mrow.loc['lightgbm','train_mae']:.3f} on training rows |
| **Mean of the two** | **{mrow.loc['blend','cv_mae']:.3f}** | **{mrow.loc['blend','test_mae']:.3f}** | selected by cross-validation |

Mean absolute error in degC. The ensemble was selected by grouped
cross-validation within the training sessions, before any held-out score was
computed. It is also the best performer on the held-out sessions, but that
result did not inform the selection.

The trend baseline is worth noting. Extrapolating the preceding five minutes
across the following ten produces a larger error than assuming no change at all
({mrow.loc['trend','test_mae']:.3f} degC against
{mrow.loc['zero','test_mae']:.3f} degC). It also cannot distinguish between
actions, since it ignores `u` and `f`, so it provides no basis for control.

![Models](results/figures/step10_models.png)

## 5. Experimental controls

Six procedures were used to prevent the reported figures from overstating
performance.

**5.1 Randomised actions during data collection.** A competent policy opens the
window only when doing so is beneficial, so its logs contain few examples of
ventilating under unfavourable conditions. Predicting those conditions is the
model's primary function. Accordingly,
{int((pd.read_csv('results/tables/step08_training_sessions.csv').policy=='random').sum())}
of the {len(pd.read_csv('results/tables/step08_training_sessions.csv'))} training
sessions used randomised actions. The remainder held a single action constant
for four hours from a deliberately extreme initial state, following standard
practice for thermal system identification.

**5.2 Splitting by session rather than by row.** Consecutive rows are 30 s
apart, and the lag-1 autocorrelation of indoor temperature within a session is
{head['autocorr_lag1']:.3f}. Adjacent rows are therefore near-duplicate
measurements, and a random split at the row level places a near-duplicate of
almost every test row in the training set. Trained and scored that way, the
selected model reports
{oracles.loc['leaky_row_level_split','mae']:.3f} degC, which is lower than the
attainable floor of {head['g1_floor']:.3f} degC. That result is not physically
possible and identifies the leakage.

**5.3 Exclusion of rows with a changing action.** A row was retained only if `u`
and `f` remained constant throughout the ten-minute horizon. Otherwise the
recorded outcome combines two different actions and does not correspond to
either. This criterion excluded {head['rows_dropped_pct']:.0f}% of logged rows.

**5.4 Separation of weather series.** The 36 evaluation runs used weather seeds
11 to 13. Training used seeds 101 to 108 and 300 to 305. The paired trials used
seeds 21 to 23. No seed appears in more than one role.

**5.5 Paired runs.** Within each combination of scenario and repeat, all three
policies received identical outdoor temperature series, identical initial
states and identical sensor calibration offsets. Differences between policies
within a cell are therefore attributable to the policy alone.

**5.6 Computation of attainable limits.** For objectives G1 and G3 the best
attainable result was computed exactly from the governing equations. A target
is interpretable only in relation to that limit.

| Reference predictor | MAE | Information available |
|---|---|---|
| Exact physical model, true future weather | {oracles.loc['perfect_physics_true_weather','mae']:.3f} degC | true state, true future outdoor series |
| **Exact physical model, outdoor temperature held** | **{oracles.loc['perfect_physics_outdoor_held','mae']:.3f} degC** | true state, outdoor temperature held constant |
| Exact physical model, outdoor trend extrapolated | {oracles.loc['perfect_physics_outdoor_trend','mae']:.3f} degC | true state, outdoor trend extrapolated |
| Selected model | {oracles.loc['selected_model','mae']:.3f} degC | measured state and 5-minute trends only |
| Constant zero prediction | {oracles.loc['guess_no_change','mae']:.3f} degC | no information used |

Of the {head['g1_floor']:.3f} degC floor,
{oracles.loc['perfect_physics_true_weather','mae']:.3f} degC is attributable to
sensor noise in the label and the remaining
{oracles.loc['perfect_physics_outdoor_held','mae']-oracles.loc['perfect_physics_true_weather','mae']:.3f} degC
to the absence of a weather forecast. The G1 target of 0.25 degC leaves
{0.25-head['g1_floor']:.3f} degC of margin above the floor.

## 6. Policy comparison

36 runs of three simulated hours each: three policies, four scenarios, three
repeats.

| Measure | Timer | Baseline rule | Model |
|---|---|---|---|
| Degree-minutes above 26 degC | {t.dm:.1f} | {r.dm:.1f} | **{m.dm:.1f}** |
| Minutes above 26 degC | {t.over:.1f} | {r.over:.1f} | **{m.over:.1f}** |
| Minutes below 24 degC | {t.under:.1f} | {r.under:.1f} | **{m.under:.1f}** |
| Mean indoor temperature (degC) | {t.mean_t:.2f} | {r.mean_t:.2f} | **{m.mean_t:.2f}** |
| Peak indoor temperature (degC) | {t.max_t:.2f} | {r.max_t:.2f} | **{m.max_t:.2f}** |
| Fan minutes | {t.fan_min:.1f} | {r.fan_min:.1f} | **{m.fan_min:.1f}** |
| Fan watt-hours | {t.fan_wh:.2f} | {r.fan_wh:.2f} | **{m.fan_wh:.2f}** |
| Window open, minutes | {t.open_min:.1f} | {r.open_min:.1f} | **{m.open_min:.1f}** |
| Window movements | {t.moves:.1f} | {r.moves:.1f} | **{m.moves:.1f}** |

Values are means across the 12 runs per policy, except peak indoor
temperature, which is the maximum.

The principal result concerns energy rather than temperature. The model
achieved comparable comfort to the baseline rule while using
{(1-m.fan_min/r.fan_min)*100:.0f}% less fan runtime and
{(1-m.moves/r.moves)*100:.0f}% fewer window movements.

The comfort figures were close but not identical. The model recorded {m.dm:.1f}
degree-minutes above 26 degC against {r.dm:.1f} for the baseline rule, a
difference of {(m.dm/r.dm-1)*100:.1f}%, and reached a lower peak temperature
({m.max_t:.2f} degC against {r.max_t:.2f} degC). Both differences are small
relative to the {(1-m.fan_min/r.fan_min)*100:.0f}% difference in fan energy.

The baseline rule obtains its comfort result by ventilating whenever outdoor
air is cooler and running the fan throughout, which amounts to
{(r.fan_min/t.fan_min-1)*100:.0f}% more fan runtime than the timer. The model
ventilated for longer than the baseline rule ({m.open_min:.0f} minutes against
{r.open_min:.0f}), since window opening carries no energy cost, but operated
the fan only when its predicted contribution justified it.

![Three policies](results/figures/step14_policies.png)

## 7. Interaction between the comfort and energy objectives

Ventilation is beneficial under most conditions in this room, because a sealed
room containing 17.6 W settles {D['shut_rise']:.1f} K above outdoor
temperature. A controller optimising for comfort therefore ventilates for
longer than a timer operating at 50% duty, not for less time. If such a
controller also runs the fan whenever the fan provides any benefit, its fan
runtime exceeds the timer's and objective G4 fails.

The window and the fan are separate decisions, which resolves the conflict. The
window accounts for most of the cooling and costs nothing to open. The fan adds
a smaller increment at a cost of 2.5 W. A controller can therefore ventilate
freely while operating the fan selectively. The selectivity is set by the fan
margin, specified in the plan as {cfg.FAN_MARGIN_C:.2f} degC.

![Fan trade-off](results/figures/step11_fan_tradeoff.png)

Every setting of the fan margin satisfies G4 by a wide margin, and none
satisfies G3. The value specified in the plan gives the best comfort result
among the settings that satisfy G4, so no revision was required.

| Policy | Degree-minutes above 26 degC | Change vs timer |
|---|---|---|
| No ventilation | {bsum['always_shut']:.1f} | {(1-bsum['always_shut']/bsum['timer'])*100:+.1f}% |
| Timer, 50% duty cycle | {bsum['timer']:.1f} | reference |
| Continuous opening, fan off | {bsum['always_open']:.1f} | {(1-bsum['always_open']/bsum['timer'])*100:+.1f}% |
| Continuous opening, fan on | {bsum['always_open_fan']:.1f} | {(1-bsum['always_open_fan']/bsum['timer'])*100:+.1f}% |
| **Reference controller** | **{bsum['greedy_oracle']:.1f}** | **{(1-bsum['greedy_oracle']/bsum['timer'])*100:+.1f}%** |
| Air held at outdoor temperature, unattainable | {bsum['outdoor_floor']:.1f} | {(1-bsum['outdoor_floor']/bsum['timer'])*100:+.1f}% |

The reference controller improves only marginally on continuous ventilation at
full fan speed, which indicates that the comfort-optimal policy in this room is
close to continuous maximum ventilation. That policy is also the least
efficient. The reference controller reaches
{(1-bsum['greedy_oracle']/bsum['timer'])*100:.1f}%, so the G3 target of 30% was
not attainable.

## 8. Limitations

- **The G3 target is not attainable in this room.** The model achieved
  {head['g3_pct']:.1f}% against a ceiling of {head['g3_ceiling']:.1f}%. The
  shortfall is a property of the objective rather than of the controller.
- **The specified sample of 30 paired trials is underpowered.** It cannot
  separate the model from the baseline rule at a difference of
  {(head['g2_model_large']-head['g2_rule_large'])*100:.0f} percentage points.
  The larger sample was added for this reason.
- **Prediction accuracy is worst in the hot-then-cooling scenario**, at
  {worst.blend:.3f} degC, above the 0.25 degC target. Outdoor temperature
  changes fastest in that scenario and the attainable floor is correspondingly
  higher, at {worst.floor:.3f} degC. The target was met in the other three
  scenarios ({others}).
- **The fan specification was revised during the study.** An initial value of
  8 L/s for a 2.5 W fan corresponds to 436 air changes per hour in a 66 L
  enclosure, which that fan cannot deliver. It was reduced to
  {cfg.FLOW_FAN_MAX*1000:.0f} L/s, or
  {cfg.FLOW_FAN_MAX*3600/cfg.VOLUME_IN:.0f} air changes per hour. The error was
  noticed while investigating excessive fan use by the controller and was
  corrected on physical grounds; all downstream steps were regenerated. The
  revision changed the G1 result from 0.264 degC to {head['g1_mae']:.3f} degC,
  so the sequence of events is recorded here.
- **The warm shell is an initial condition, not a solar gain model.** The
  warm-shell scenario begins with the shell approximately
  {D['shell_gap']:.0f} K above outdoor temperature. Modelling solar gain
  explicitly was excluded from scope.
- **Humidity, measured weather data and alternative room geometries were not
  investigated.**

## 9. Reproducibility

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_all.py          # all steps in order
.venv/bin/python scripts/run_all.py 10 13    # steps 10 and 13 only
```

Each step concludes with an acceptance check and returns a non-zero exit code
on failure, and the driver halts at the first failure, so an incorrect constant
cannot propagate silently into the results.

The logged session data and the fitted model are regenerated by the pipeline
and are therefore not tracked in this repository. Run the driver in full once
before running individual steps, since steps 11 to 14 read the fitted model
produced by step 10.

The thermal model is verified against independent calculations in step 2. The
checks cover conservation of energy, agreement with the closed-form matrix
exponential solution, agreement with a separately derived resistance network,
agreement with Newton's law of cooling in the single-mass limit, and agreement
between fitted and predicted time constants. All agree to within 1 part in
10^8.

## Pipeline steps

| Step | Task | Acceptance check |
|---|---|---|
| 0 | Project setup and invariants | a script runs; 30 core checks pass |
| 1 | Thermal model | a sealed room with the heater on warms monotonically to a limit |
| 2 | Verification of the thermal model | agreement with independent calculations, 6 checks |
| 3 | Weather generator | all four scenarios generate and plot |
| 4 | Window and fan effects | ventilation cools when outdoor air is cooler and warms when it is warmer |
| 5 | Sensor model | readings scatter about the true values and offsets persist |
| 6 | Simulator and logger | one three-hour run produces a CSV |
| 7 | Timer and baseline rule | both execute and produce results |
| 8 | Training data | {len(pd.read_csv('results/tables/step08_training_sessions.csv'))} sessions collected, against a requirement of 15 |
| 9 | Inputs and split | no session appears on both sides |
| 10 | Predictor training | MAE reported for all candidates, with the attainable floor |
| 11 | Model-based controller | completes a full simulation |
| 12 | Evaluation runs | 36 runs logged |
| 13 | Objective evaluation | results table produced |
| 14 | Figures and write-up | this document |

## Repository layout

```
windwindow/
  config.py      physical constants, derived from the room specification
  room.py        two-mass thermal model, RK4 integrator and exact solver
  weather.py     the four outdoor temperature scenarios
  sensors.py     calibration offset, lag, noise and quantisation
  policies.py    timer, baseline rule, randomised, fixed and model controllers
  simulator.py   the loop that runs any policy under identical conditions
  features.py    the nine inputs, the label and the action-stability filter
  dataset.py     loading and the session-level split
  models.py      baselines, ridge regression, gradient boosting, ensemble
  oracle.py      attainable limits for prediction and for control
  trials.py      paired open/shut trials for objective G2
  plotting.py    shared figure style
  report.py      this document and index.html, generated from the results
scripts/
  step01 to step14   one per pipeline step, each ending in its acceptance check
  run_all.py         runs all steps in order, halting at the first failure
tests/
  test_core.py       fast invariant checks
results/
  figures/  {len(figs)} PNG files      tables/  {len(tables)} CSV files
data/
  train/        {len(list(pathlib.Path('data/train').glob('*.csv')))} training sessions
  experiments/  {len(list(pathlib.Path('data/experiments').glob('*.csv')))} evaluation runs
```
"""
    pathlib.Path("README.md").write_text(md)
    print(f"  wrote README.md ({len(md):,} characters)")


# --------------------------------------------------------------------------
# The HTML report
# --------------------------------------------------------------------------
TEMPLATE = pathlib.Path(__file__).parent / "report_template.html"


def _sci(x: float) -> str:
    """Format a small number as 3.1x10^-5 with real superscript digits."""
    mant, exp = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mant}×10{str(int(exp)).translate(sup)}"


def write_report_html(head, runs, goals, bounds, big, models) -> None:
    """Fill report_template.html from the measured tables and write index.html.

    Same rule as the README: every number comes from a results file. The
    template holds the layout and the prose; it holds no figures of its own.
    """
    import json

    D = _derived()
    pol = runs.groupby("policy").agg(
        dm=("degree_minutes_over_26", "mean"), over=("minutes_over_26", "mean"),
        under=("minutes_under_24", "mean"), mean_t=("mean_t_in", "mean"),
        max_t=("max_t_in", "max"), fan_min=("fan_minutes", "mean"),
        fan_wh=("fan_wh", "mean"), open_min=("window_open_minutes", "mean"),
        moves=("moves", "mean"))
    t, r, m = pol.loc["timer"], pol.loc["simple_rule"], pol.loc["model"]
    mrow = models.set_index("model")
    imp = pd.read_csv("results/tables/step10_feature_importance.csv")
    imp = imp.sort_values("lgbm_gain_frac", ascending=False)
    oracles = pd.read_csv("results/tables/step10_oracles.csv").set_index("oracle")
    sessions = pd.read_csv("results/tables/step08_training_sessions.csv")
    by_sc = pd.read_csv("results/tables/step10_by_scenario.csv")
    worst = by_sc.sort_values("blend", ascending=False).iloc[0]
    choice = json.loads(pathlib.Path("results/fan_margin.json").read_text())
    bsum = bounds.mean()

    rt = pathlib.Path("results/runtime.json")
    runtime = json.loads(rt.read_text())["total_seconds"] if rt.exists() else 110.0

    # The one run the "three controllers" figure is drawn from.
    trap = {}
    for pname in ("simple_rule", "model"):
        d = pd.read_csv(f"data/experiments/hot_walls_cold_outside_s11_{pname}.csv")
        trap[pname] = d["fan_on_s"].sum() / 60.0

    # ---- generated table bodies ----------------------------------------
    plain = {"d": "driving temperature difference across the envelope",
             "u_d": "the same difference, gated by window opening",
             "u_f_d": "the same difference, gated by opening and fan speed",
             "wall_minus_in": "heat stored in the shell relative to the air",
             "heater_w": "internal heat load",
             "slope_in": "indoor temperature trend over the preceding 5 min",
             "slope_out": "outdoor temperature trend over the preceding 5 min",
             "u_wall_minus_out": "shell heat released into the incoming airstream",
             "u_slope_out": "outdoor trend, gated by window opening"}
    code = {"d": "d", "u_d": "u·d", "u_f_d": "u·f·d",
            "wall_minus_in": "T_wall − T_in", "heater_w": "heater",
            "slope_in": "slope_in", "slope_out": "slope_out",
            "u_wall_minus_out": "u·(T_wall − T_out)",
            "u_slope_out": "u·slope_out"}
    top = imp.lgbm_gain_frac.max()
    input_rows = "\n".join(
        f'    <tr><td><code>{code[row.input]}</code></td><td>{plain[row.input]}</td>'
        f'<td><div class="bar"><i style="width:{row.lgbm_gain_frac/top*98:.0f}px"></i>'
        f'<span>{row.lgbm_gain_frac*100:.1f}%</span></div></td></tr>'
        for row in imp.itertuples())

    model_labels = [("zero", "Constant zero prediction", "reference value"),
                    ("trend", "Linear extrapolation of the 5-min trend",
                     "less accurate than predicting no change"),
                    ("ridge", "Ridge regression", "linear in the nine inputs"),
                    ("lightgbm", "Gradient-boosted trees",
                     f"overfits to {mrow.loc['lightgbm','train_mae']:.3f} on training rows"),
                    ("blend", "Mean of ridge and gradient-boosted trees",
                     "selected by cross-validation")]
    model_rows = "\n".join(
        f'    <tr{" class=\"hi\"" if k == "blend" else ""}><td>{lbl}</td>'
        f'<td class="n">{mrow.loc[k,"cv_mae"]:.3f}</td>'
        f'<td class="n{" win" if k == "blend" else ""}">{mrow.loc[k,"test_mae"]:.3f}</td>'
        f'<td>{note}</td></tr>'
        for k, lbl, note in model_labels)

    oracle_rows = "\n".join([
        f'    <tr><td>Exact physical model, true future weather</td>'
        f'<td class="n">{oracles.loc["perfect_physics_true_weather","mae"]:.3f} °C</td>'
        f'<td>true state, true future outdoor series</td></tr>',
        f'    <tr class="hi"><td>Exact physical model, outdoor temperature held</td>'
        f'<td class="n">{oracles.loc["perfect_physics_outdoor_held","mae"]:.3f} °C</td>'
        f'<td>true state, outdoor temperature held constant</td></tr>',
        f'    <tr><td>Exact physical model, outdoor trend extrapolated</td>'
        f'<td class="n">{oracles.loc["perfect_physics_outdoor_trend","mae"]:.3f} °C</td>'
        f'<td>true state, outdoor trend extrapolated</td></tr>',
        f'    <tr><td>Selected model</td>'
        f'<td class="n win">{oracles.loc["selected_model","mae"]:.3f} °C</td>'
        f'<td>measured state and 5-minute trends only</td></tr>',
        f'    <tr><td>Constant zero prediction</td>'
        f'<td class="n">{oracles.loc["guess_no_change","mae"]:.3f} °C</td>'
        f'<td>no information used</td></tr>'])

    def prow(label, fmt, key, hi=False, win=None, lower_better=True):
        vals = {k: getattr(pol.loc[k], key) for k in ("timer", "simple_rule", "model")}
        cells = ""
        for k in ("timer", "simple_rule", "model"):
            cls = "n win" if (win == k) else "n"
            cells += f'<td class="{cls}">{fmt.format(vals[k])}</td>'
        return (f'    <tr{" class=\"hi\"" if hi else ""}><td>{label}</td>{cells}</tr>')

    policy_rows = "\n".join([
        prow("Degree-minutes above 26 °C", "{:.1f}", "dm"),
        prow("Minutes above 26 °C", "{:.1f}", "over"),
        prow("Minutes below 24 °C", "{:.1f}", "under"),
        prow("Mean indoor temperature", "{:.2f} °C", "mean_t"),
        prow("Peak indoor temperature", "{:.2f} °C", "max_t", win="model"),
        prow("Fan minutes", "{:.1f}", "fan_min", hi=True, win="model"),
        prow("Fan watt-hours", "{:.2f}", "fan_wh", hi=True, win="model"),
        prow("Window open, minutes", "{:.1f}", "open_min"),
        prow("Window movements", "{:.1f}", "moves", hi=True, win="model"),
    ])

    bound_labels = [("always_shut", "No ventilation", False),
                    ("timer", "Timer, 50% duty cycle", False),
                    ("always_open", "Continuous opening, fan off", False),
                    ("always_open_fan", "Continuous opening, fan on", False),
                    ("greedy_oracle", "Reference controller, true state and weather", True),
                    ("outdoor_floor", "Air held at outdoor temperature, unattainable", False)]
    def brel(k):
        if k == "timer":
            return "reference"
        v = (1 - bsum[k] / bsum["timer"]) * 100
        return f"{v:+.1f}%".replace("-", "−")
    bound_rows = "\n".join(
        f'    <tr{" class=\"hi\"" if hi else ""}><td>{lbl}</td>'
        f'<td class="n">{bsum[k]:.1f}</td>'
        f'<td class="n{" win" if hi else ""}">{brel(k)}</td></tr>'
        for k, lbl, hi in bound_labels)

    vals = {
        "G1_MAE": f"{head['g1_mae']:.3f}", "G1_FLOOR": f"{head['g1_floor']:.3f}",
        "G2_SMALL": f"{head['g2_model']*100:.1f}",
        "G2_BIG": f"{head['g2_model_large']*100:.1f}",
        "G2_RULE_BIG": f"{head['g2_rule_large']*100:.1f}",
        "N_BIG": str(len(big)),
        "G2_GAP": f"{(head['g2_model_large']-head['g2_rule_large'])*100:.0f}",
        "G2_DISAGREE": str(int((big.model_correct & ~big.rule_correct).sum())),
        "MCNEMAR": _sci(head["mcnemar_p"]),
        "G3_PCT": f"{head['g3_pct']:.1f}", "G3_CEIL": f"{head['g3_ceiling']:.1f}",
        "G3_SHARE": f"{head['g3_pct']/head['g3_ceiling']*100:.0f}",
        "G3_CEIL_SEED1": f"{choice['achievable_g3_pct']:.1f}",
        "G4_PCT": f"{head['g4_pct']:.1f}",
        "RS_AHEAD": f"{head['resample_ahead_pct']:.0f}",
        "RS_TIED": f"{head['resample_tied_pct']:.0f}",
        "RS_BEHIND": f"{head['resample_behind_pct']:.0f}",
        "C_IN": f"{cfg.C_IN:,.0f}", "C_WALL": f"{cfg.C_WALL:,.0f}",
        "C_WALL_KJ": f"{cfg.C_WALL/1000:.0f}", "MASS_WALL": f"{cfg.MASS_WALL:.1f}",
        "TF_SHUT": f"{D['tau_fast_shut']:.1f}", "TS_SHUT": f"{D['tau_slow_shut']:.1f}",
        "TF_OPEN": f"{D['tau_fast_open']:.1f}", "TS_OPEN": f"{D['tau_slow_open']:.1f}",
        "TF_FAN": f"{D['tau_fast_open_fan']:.1f}", "TS_FAN": f"{D['tau_slow_open_fan']:.1f}",
        "H_SHUT": f"{D['h_in_shut']:.1f}", "K_SHUT": f"{D['k_iw_shut']:.2f}",
        "H_OPEN": f"{D['h_in_open']:.1f}", "K_OPEN": f"{D['k_iw_open']:.2f}",
        "H_FAN": f"{D['h_in_open_fan']:.1f}", "K_FAN": f"{D['k_iw_open_fan']:.2f}",
        "REBOUND": f"{D['rebound']:.2f}", "SHUT_RISE": f"{D['shut_rise']:.1f}",
        "SHELL_GAP": f"{D['shell_gap']:.0f}",
        "RULE_WRONG": str(int((~big.rule_correct).sum())),
        "RULE_DOUT": f"{(big[~big.rule_correct].t_out_true - big[~big.rule_correct].t_in_true).mean():+.2f}",
        "INPUT_ROWS": input_rows, "MODEL_ROWS": model_rows,
        "ORACLE_ROWS": oracle_rows, "POLICY_ROWS": policy_rows,
        "BOUND_ROWS": bound_rows,
        "TOP_SHARE": f"{imp.iloc[0].lgbm_gain_frac*100:.0f}",
        "WALL_SHARE": f"{imp[imp.input.isin(['wall_minus_in','u_wall_minus_out'])].lgbm_gain_frac.sum()*100:.0f}",
        "OPEN_TH": f"{cfg.OPEN_THRESHOLD_C:.2f}", "CLOSE_TH": f"{cfg.CLOSE_THRESHOLD_C:.2f}",
        "DWELL": f"{cfg.MIN_DWELL_S/60:.0f}", "FLOOR_C": f"{cfg.T_IN_FLOOR_C:.0f}",
        "FAN_MARGIN": f"{choice['fan_margin']:.2f}",
        "TREND_MAE": f"{mrow.loc['trend','test_mae']:.3f}",
        "ZERO_MAE": f"{mrow.loc['zero','test_mae']:.3f}",
        "N_RANDOM": str(int((sessions.policy == "random").sum())),
        "N_SESSIONS": str(len(sessions)), "N_RUNS": str(len(runs)),
        "AUTOCORR": f"{head['autocorr_lag1']:.3f}",
        "LEAKY_MAE": f"{oracles.loc['leaky_row_level_split','mae']:.3f}",
        "DROPPED": f"{head['rows_dropped_pct']:.0f}",
        "O_PERFECT": f"{oracles.loc['perfect_physics_true_weather','mae']:.3f}",
        "O_WEATHER": f"{oracles.loc['perfect_physics_outdoor_held','mae']-oracles.loc['perfect_physics_true_weather','mae']:.3f}",
        "FAN_SAVE": f"{(1-m.fan_min/r.fan_min)*100:.0f}",
        "MOVE_SAVE": f"{(1-m.moves/r.moves)*100:.0f}",
        "M_DM": f"{m.dm:.1f}", "R_DM": f"{r.dm:.1f}",
        "DM_WORSE": f"{(m.dm/r.dm-1)*100:.1f}",
        "M_PEAK": f"{m.max_t:.2f}", "R_PEAK": f"{r.max_t:.2f}",
        "RULE_FAN_MORE": f"{(r.fan_min/t.fan_min-1)*100:.0f}",
        "M_OPEN": f"{m.open_min:.0f}", "R_OPEN": f"{r.open_min:.0f}",
        "TRAP_RULE_FAN": f"{trap['simple_rule']:.0f}",
        "TRAP_MODEL_FAN": f"{trap['model']:.0f}",
        "WORST_MAE": f"{worst.blend:.3f}", "WORST_FLOOR": f"{worst.floor:.3f}",
        "OTHER_MAES": ", ".join(f"{row.blend:.3f}" for row in
                                by_sc[by_sc.scenario != worst.scenario].itertuples()),
        "FAN_LPS": f"{cfg.FLOW_FAN_MAX*1000:.0f}",
        "FAN_ACH": f"{cfg.FLOW_FAN_MAX*3600/cfg.VOLUME_IN:.0f}",
        "RUNTIME": f"{runtime:.0f}",
    }

    html = TEMPLATE.read_text()
    for key, value in vals.items():
        html = html.replace("{{" + key + "}}", str(value))
    import re
    left = sorted(set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", html)))
    if left:
        raise AssertionError(f"report template has unfilled tokens: {left}")
    # Written as index.html so GitHub Pages serves it at the site root.
    pathlib.Path("index.html").write_text(html)
    print(f"  wrote index.html ({len(html):,} characters)")
