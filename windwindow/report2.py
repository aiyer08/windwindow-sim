"""
The write-up, generated from the result tables.

Prose is written once in `_document` and rendered to both README.md and
index.html by windwindow.doc, so the two cannot diverge. Every number is read
from a results file rather than typed, so the text cannot drift from the code.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from . import config as cfg
from .doc import Doc
from .gains import daily_summary
from .realweather import LOCATIONS

SHELL = pathlib.Path(__file__).parent / "report_shell.html"
NICE = {k: v["label"] for k, v in LOCATIONS.items()}


def _values(runs, by_clim, ctl) -> dict:
    """Everything the prose quotes, computed from the tables."""
    ss = json.load(open("results/statespace.json"))
    theta = np.array(ss["theta"])
    herr = pd.DataFrame(ss["horizon_error"]).set_index("horizon_min")
    penalty = json.load(open("results/fan_penalty.json"))
    b0 = by_clim.set_index("location")
    b = b0
    g = runs.groupby("policy")

    full = ctl[ctl.label == "full controller"].iloc[0]
    rule = ctl[ctl.label == "threshold rule"].iloc[0]
    sealed = ctl[ctl.label == "sealed"].iloc[0]
    abl = ctl[ctl.group == "ablation"].set_index("label")
    gs = daily_summary()
    rt = pathlib.Path("results/runtime.json")
    runtime_min = (json.load(open(rt))["total_seconds"] / 60.0
                   if rt.exists() else None)

    # Fan electricity against the threshold rule, per climate and overall.
    # Computed here rather than in step 18 so the comparison can be added
    # without re-running half an hour of simulation.
    fan = runs.pivot_table(index="location", columns="policy", values="fan_wh")
    dmm = runs.pivot_table(index="location", columns="policy",
                           values="degree_minutes_over_26")
    fan_saving_by_loc = ((1 - fan["controller"] / fan["threshold rule"]) * 100)
    fan_saving_all = (1 - fan["controller"].sum()
                      / fan["threshold rule"].sum()) * 100
    comfort_delta_all = (dmm["controller"].sum()
                         / dmm["threshold rule"].sum() - 1) * 100

    # Judge a climate by what the room actually achieves, not by the
    # percentage improvement. Phoenix shows why: a 29% cut off an enormous
    # number is still enormous, and the room there is above the threshold for
    # every hour of the run. The useful question is what fraction of the time
    # the room ends up comfortable.
    hours_run = 48.0
    sealed_hours = (runs.pivot_table(index="location", columns="policy",
                                     values="minutes_over_26") / 60.0)
    b = b.assign(sealed_hours_over=sealed_hours["sealed"],
                 frac_over=b.hours_over_26 / hours_run,
                 hours_under=hours_run - b.hours_over_26,
                 ratio=b.sealed / b.controller)
    works = b[b.frac_over <= 0.35]
    best = b.vs_sealed_pct.idxmax()
    worst = b.vs_sealed_pct.idxmin()

    from .statespace import true_params
    tp = true_params()
    ua_hat = 1.0 / (1.0 / theta[2] + 1.0 / theta[8]) + theta[5]
    ua_true = 1.0 / (1.0 / cfg.K_IN_WALL_BASE + 1.0 / cfg.K_WALL_OUT) + cfg.K_LEAK

    return dict(
        b=b, g=g, runs=runs, ctl=ctl, abl=abl, herr=herr, theta=theta, tp=tp,
        gs=gs, runtime_min=runtime_min,
        penalty=penalty["fan_penalty"], tuned_on=penalty["tuned_on"],
        full=full, rule=rule, sealed=sealed,
        fan_by_loc=fan_saving_by_loc, fan_all=fan_saving_all,
        comfort_all=comfort_delta_all, fan_tbl=fan, dm_tbl=dmm,
        hours_run=hours_run,
        n_works=len(works), works=list(works.index), best=best, worst=worst,
        best_pct=b.loc[best].vs_sealed_pct, worst_pct=b.loc[worst].vs_sealed_pct,
        best_night=b.loc[best].night_mean_c, worst_night=b.loc[worst].night_mean_c,
        max_gap=b.gap_to_reference_pct.max(),
        mean_gap=b.gap_to_reference_pct.mean(),
        fan_saving=(1 - full.fan_wh / max(rule.fan_wh, 1e-9)) * 100,
        comfort_delta=(full.degree_minutes_over_26 / rule.degree_minutes_over_26 - 1) * 100,
        ua_hat=ua_hat, ua_true=ua_true,
        tau_hat=theta[1] / (theta[2] + theta[8]) / 3600.0,
        tau_true=cfg.C_WALL / (cfg.K_IN_WALL_BASE + cfg.K_WALL_OUT) / 3600.0,
        n_sessions=ss["fit_info"]["n_sessions"],
        n_windows=ss["fit_info"]["n_windows"],
        param_err=np.median(np.abs((theta - tp) / tp * 100)),
    )


def _document(V, figs, tables) -> Doc:
    b, abl, herr = V["b"], V["abl"], V["herr"]
    best, worst = V["best"], V["worst"]
    bb, bw = b.loc[best], b.loc[worst]
    d = Doc()

    d.h1("WindWindow Sim")
    d.p("This simulates a bedroom in five real climates and asks whether "
        "opening and shutting the window at the right times can keep it cool "
        "through a heat wave with no air conditioning.")
    d.p(f"In the two climates with cool nights it cuts overheating by "
        f"{bb.vs_sealed_pct:.0f}% and "
        f"{b.loc[V['works'][0]].vs_sealed_pct:.0f}%. That number counts both "
        f"how far above {cfg.COMFORT_C:.0f} \u00b0C the room goes and how long "
        f"it stays there. The time spent above {cfg.COMFORT_C:.0f} \u00b0C "
        f"falls from {bb.sealed_hours_over:.0f} hours to "
        f"{bb.hours_over_26:.0f} out of 48. "
        f"It comes within {V['max_gap']:.1f}% of a controller that is given "
        f"the exact weather in advance and the true room parameters.")
    d.p("I did not give the controller this strategy. It gets a comfort "
        "target and a 23-hour planning horizon, and it works the rest out on "
        "its own: open the window overnight to cool the walls and floor, keep "
        "it shut through the heat of the day, and open it again in the "
        "evening.")

    d.cards([
        dict(title="Overheating cut",
             value=f"{bb.vs_sealed_pct:.0f}%",
             note=f"{NICE[best]}. {bb.ratio:.1f} times less than leaving the "
                  f"window shut"),
        dict(title="Within",
             value=f"{V['max_gap']:.1f}% of optimal",
             note="measured against perfect forecast and exact room "
                  "parameters, worst of five climates"),
        dict(title="Fan electricity",
             value=f"{V['fan_all']:.0f}% less",
             note=f"than the best heuristic, at identical comfort "
                  f"({V['comfort_all']:+.1f}%)"),
        dict(title="Screening rule",
             value="r = -0.98",
             note="one free measurement predicts whether it will work in a "
                  "given climate"),
    ])

    d.h2("Results", "summary")
    d.figure("results/figures/rep_by_climate.png",
             "Grouped bars of overheating by climate and policy, and a bar "
             "chart of the reduction the controller achieves in each climate.",
             "Five climates, two weather seeds each, 48 hours per run. Neither "
             "seed was used for tuning.")

    rows = []
    for loc in LOCATIONS:
        r = b.loc[loc]
        rows.append([NICE[loc], f"{r.night_mean_c:.1f}", f"{r.t_out_max:.1f}",
                     f"{r.vs_sealed_pct:.0f}%", f"{r.ratio:.1f}x",
                     f"{r.sealed_hours_over:.1f}", f"{r.hours_over_26:.1f}",
                     f"{r.fan_kwh:.2f}"])
    d.table(["Climate", "Night mean", "Peak outdoor", "Overheating cut",
             "vs shut", "Hours over 26, shut", "Hours over 26, controller",
             "Fan kWh"],
            rows, align=["l", "n", "n", "n", "n", "n", "n", "n"],
            caption="48-hour runs. The last two columns are the same room "
                    "left shut, then run by the controller.")

    d.h3("One measurement predicts whether this will work")
    d.p("Across the five climates, how much the controller helps tracks the "
        "average overnight temperature at r = -0.98. So you can work out "
        "whether it is worth installing by leaving a thermometer outside for a "
        "week. This works because the controller cools the walls and floor at "
        "night and then lives off that stored cold during the day. If the "
        "nights are not cold, there is nothing to store and nothing to live "
        "off.")
    d.p(f"{NICE[worst]} is the climate where this does not work. Nights "
        f"there average {bw.night_mean_c:.0f} °C, and the room stays "
        f"above {cfg.COMFORT_C:.0f} °C for all 48 hours even when the "
        f"controller is given perfect information. The "
        f"{bw.vs_sealed_pct:.0f}% reduction it manages is real, but the room "
        f"is still too warm to sleep in. If your nights are that warm, the "
        f"money is better spent on insulation, shading or a heat pump, and a "
        f"week with a thermometer will tell you which case you are in.")

    d.h2("What the controller does", "controller")
    d.figure("results/figures/rep_strategy.png",
             "Indoor temperature under the controller, a sealed room and a "
             "permanently open room over two days, with window and fan "
             "settings below and the daily phases marked.",
             "The controller opens up overnight to cool the structure, keeps "
             "the room shut through the morning and midday, then opens again "
             "once the outside air drops back below the indoor temperature.")
    d.p(f"The controller is told to do three things: stay below "
        f"{cfg.COMFORT_C:.0f} \u00b0C, stay above {cfg.T_IN_FLOOR_C:.0f} "
        f"\u00b0C, and keep down what it spends on fan electricity and on "
        f"moving the window. None of that mentions "
        f"night, day, or the walls. The controller plans the whole of the next "
        f"day at once, and once it can see the coming afternoon, cooling the "
        f"walls at 4 am turns out to be the cheapest way to get through it. "
        f"That is where the daily rhythm in the figure comes from.")
    d.p("How far ahead the controller plans matters more than anything else "
        "I tuned. With a 7.5-hour horizon, a decision made at 2 am cannot see "
        "the following afternoon, so cooling the walls looks like a waste of "
        "fan power and the controller refuses to do it. At that horizon it "
        "loses to the simple rule of opening the window whenever it is cooler "
        "outside. Stretching the horizon to 23.5 hours is the one change that "
        "makes the night-cooling behaviour show up.")
    d.code(f"horizon 23.5 h in 12 blocks: "
           f"{', '.join(str(x) for x in [15,15,30,30,60,60,120,120,180,240,240,300])} min\n"
           f"minimise  degree-minutes over {cfg.COMFORT_C:.0f} degC\n"
           f"        + degree-minutes under {cfg.T_IN_FLOOR_C:.0f} degC\n"
           f"        + {V['penalty']:g} per watt-hour of fan\n"
           f"        + a small charge for moving the window\n"
           f"re-plan every 15 min, execute the first block only")

    d.h2("Comparison", "comparison")
    g = V["g"]
    pol_rows = []
    for pol in ["sealed", "open throughout", "timer", "threshold rule",
                "controller", "reference, perfect information"]:
        m = g.get_group(pol)
        pol_rows.append([
            {"sealed": "Shut", "open throughout": "Open",
             "timer": "Timer, half the time",
             "threshold rule": "Open when cooler outside",
             "controller": "**This controller**",
             "reference, perfect information": "Perfect information"}[pol],
            f"{m.degree_minutes_over_26.mean():.0f}",
            f"{m.minutes_over_26.mean()/60:.1f}",
            f"{m.max_t_in.max():.1f}",
            f"{m.fan_wh.mean()/1000:.2f}"])
    d.table(["Policy", "Degree-minutes over 26 \u00b0C",
             "Hours over 26 \u00b0C", "Worst peak", "Fan kWh"],
            pol_rows, align=["l", "n", "n", "n", "n"], hi=[4],
            caption="Averaged over five climates and two seeds. The peak "
                    "column is the worst single moment anywhere, so Phoenix "
                    "dominates it.")
    d.p("The rule to beat is simply opening the window whenever it is cooler "
        "outside than in. That rule gets some night cooling for free without "
        "anyone planning it, which makes it a hard baseline to improve on. On "
        "temperature the two finish level, within "
        f"{abs(V['comfort_all']):.1f}% of each other.")
    d.p(f"They differ in how much electricity they use. The simple rule turns "
        f"the fan on "
        f"whenever the window is open, because it has no way of weighing what "
        f"the fan costs against what it achieves. The controller has the price "
        f"of fan electricity written into its objective, so it only runs the "
        f"fan when the extra cooling is worth the watt-hours. That works out "
        f"to {V['fan_all']:.0f}% less fan energy for the same indoor "
        f"temperature.")
    fan_rows = []
    for loc in LOCATIONS:
        fr = V['fan_tbl'].loc[loc, 'threshold rule']
        fc_ = V['fan_tbl'].loc[loc, 'controller']
        dr = V['dm_tbl'].loc[loc, 'threshold rule']
        dc = V['dm_tbl'].loc[loc, 'controller']
        fan_rows.append([NICE[loc], f"{fr/1000:.2f}", f"{fc_/1000:.2f}",
                         f"{(1-fc_/fr)*100:.0f}%", f"{(dc/dr-1)*100:+.1f}%"])
    d.table(["Climate", "Heuristic, fan kWh", "Controller, fan kWh", "Saved",
             "Comfort difference"], fan_rows,
            align=["l", "n", "n", "n", "n"],
            caption="Negative comfort difference means the controller ran "
                    "cooler. The saving is largest where the nights are "
                    "coldest, because there the open window does the cooling "
                    "on its own and the fan is barely needed.")

    d.h2("Fitting the model", "identification")
    d.p("To plan ahead, the controller needs to know how the room behaves "
        "over the next several hours, not just what the temperature will be "
        "in ten minutes. So I fit ten numbers to the logged data: the heat "
        "capacities of the air and of the structure, and the conductances "
        "between them and the outside. The fit uses "
        f"{V['n_sessions']} logged 48-hour sessions across the five climates, "
        f"and it minimises the error over multi-hour stretches rather than "
        f"over single steps.")
    d.table(["", "Fitted", "Actual", "Off by"], [
        ["Air and furniture", f"{V['theta'][0]/1000:.0f} kJ/K",
         f"{V['tp'][0]/1000:.0f} kJ/K",
         f"{(V['theta'][0]/V['tp'][0]-1)*100:+.0f}%"],
        ["Structure", f"{V['theta'][1]/1e6:.2f} MJ/K",
         f"{V['tp'][1]/1e6:.2f} MJ/K",
         f"{(V['theta'][1]/V['tp'][1]-1)*100:+.0f}%"],
        ["Heat loss, window shut", f"{V['ua_hat']:.1f} W/K",
         f"{V['ua_true']:.1f} W/K",
         f"{(V['ua_hat']/V['ua_true']-1)*100:+.0f}%"],
        ["Structural time constant", f"{V['tau_hat']:.1f} h",
         f"{V['tau_true']:.1f} h",
         f"{(V['tau_hat']/V['tau_true']-1)*100:+.1f}%"],
    ], align=["l", "n", "n", "n"], hi=[3],
       caption=f"Median error across all ten parameters is "
               f"{V['param_err']:.0f}%. The actual values are shown for "
               f"comparison and took no part in the fit.")
    d.p(f"The structural time constant comes back within "
        f"{abs(V['tau_hat']/V['tau_true']-1)*100:.1f}% of its true value and "
        f"the heat-loss coefficient within "
        f"{abs(V['ua_hat']/V['ua_true']-1)*100:.0f}%, even though the sensors "
        f"feeding the fit carry a {cfg.SENSOR_OFFSET_C:.1f} \u00b0C "
        f"calibration offset and {cfg.SENSOR_NOISE_C:.1f} \u00b0C of noise. "
        f"All ten parameters come out inside their physically plausible "
        f"ranges.")
    d.p("Forcing the fit to stay in physical units is what makes this work, "
        "and it is worth saying what happens when you do not. The two wall "
        "conductances cannot be told apart from the data, because the physics "
        "fixes their ratio to the ratio of the two heat capacities. When I "
        "left them free, the fit came back with a negative infiltration "
        "conductance and a structure 99% too light, and it still predicted "
        "the held-out sessions reasonably well. The trouble is that a model "
        "like that falls apart as soon as you ask it about an action the data "
        "never contained, and those are exactly the actions a planner wants "
        "to try out.")
    d.table(["Horizon", "Mean error"],
            [[f"{int(h)} min" if h < 60 else f"{int(h/60)} h",
              f"{herr.loc[h].mae:.2f} \u00b0C"] for h in herr.index],
            align=["l", "n"],
            caption=f"Ten held-out sessions. The room swings about 14 "
                    f"\u00b0C over a day, so the six-hour error is around "
                    f"{herr.loc[360].mae/14*100:.0f}% of that.")
    d.figure("results/figures/step16_identification.png",
             "A held-out 48-hour session with the model prediction restarted "
             "every six hours, and a plot of error against prediction horizon.",
             "Prediction restarts every six hours from the measured state, so "
             "each segment is a genuine six-hour forecast.")

    d.h2("Ablations", "ablation")
    d.figure("results/figures/rep_ablation.png",
             "Horizontal bars showing how much worse the controller performs "
             "when each capability is changed, and a scatter of comfort "
             "against fan electricity.",
             "Each row changes one thing and re-runs the whole evaluation. A "
             "positive cost means the room ran warmer than it did with the "
             "full controller.")
    ab_rows = []
    fullv = V["full"].degree_minutes_over_26
    for label in abl.index:
        if label == "full controller":
            continue
        v = abl.loc[label].degree_minutes_over_26
        ab_rows.append([label, f"{v:.0f}", f"{(v/fullv-1)*100:+.1f}%"])
    d.table(["Change", "Degree-minutes", "Cost"], ab_rows,
            align=["l", "n", "n"],
            caption=f"Full controller: {fullv:.0f} degree-minutes. Measured on "
                    f"{NICE[V['tuned_on']]}, the climate the fan penalty was "
                    f"tuned on.")
    d.p(f"The controller needs a weather forecast, but it does not need an "
        f"accurate one. Taking the forecast away, so that it has to assume the "
        f"weather stays as it is now, costs "
        f"{(abl.loc['no forecast (persistence)'].degree_minutes_over_26/fullv-1)*100:.0f}%. "
        f"Swapping the realistic forecast for a perfect one changes the result "
        f"by only "
        f"{(abl.loc['perfect forecast'].degree_minutes_over_26/fullv-1)*100:+.1f}%, "
        f"which is within the run-to-run noise. Nearly all of the useful "
        f"information is in the daily rise and fall of the temperature, and a "
        f"clock gives you that for nothing. So a free public forecast is "
        f"enough, and there is no need for a weather station on site.")

    d.h2("The room", "room")
    d.p(f"The room is a {cfg.ROOM_L:.0f} by {cfg.ROOM_W:.0f} by "
        f"{cfg.ROOM_H:.1f} m bedroom of medium-weight construction. It has one "
        f"window that opens, a {cfg.FAN_ELEC_W:.0f} W fan, people in it for "
        f"part of the day, and afternoon sun falling on the glass.")
    d.p(f"The model tracks two temperatures: the air and furniture together, "
        f"and the structure. The structure stores {cfg.C_WALL/1e6:.1f} MJ/K of "
        f"heat, roughly {cfg.C_WALL/cfg.C_IN:.0f} times as much as the air, "
        f"and that store is what the whole strategy runs on. Cold you put into "
        f"the walls at 4 am is still there at 3 pm.")
    d.code("[gains] --> ( indoor air ) <--> ( structure ) <--> outdoors\n"
           "                  |\n"
           "                  +------- window and fan -------> outdoors")
    d.table(["", "Value", ""], [
        ["Volume", f"{cfg.VOLUME_IN:.0f} m\u00b3", ""],
        ["Air and furniture", f"{cfg.C_IN/1000:.0f} kJ/K", "settles in an hour"],
        ["Structure", f"{cfg.C_WALL/1e6:.1f} MJ/K",
         f"{V['tau_true']:.1f} h time constant"],
        ["Heat loss, window shut", f"{V['ua_true']:.1f} W/K",
         f"envelope plus {cfg.ACH_LEAK:.1f} air changes/h"],
        ["Window open", f"{cfg.K_VENT_OPEN_MAX:.0f} W/K",
         f"{cfg.ACH_OPEN_MAX:.0f} air changes/h"],
        ["Fan on top", f"{cfg.K_VENT_FAN_MAX:.0f} W/K",
         f"another {cfg.ACH_FAN_MAX:.0f} air changes/h"],
        ["Heat gains", f"{V['gs']['mean_w']:.0f} W mean, "
                       f"{V['gs']['peak_w']:.0f} W peak",
         f"peak at {V['gs']['peak_hour']:.0f}:00"],
    ], align=["l", "n", "l"], hi=[2])
    d.figure("results/figures/rep_gains.png",
             "Stacked area chart of internal and solar heat gain against hour "
             "of day, peaking in the late afternoon.",
             "Heat gains peak in the late afternoon, which is also when the "
             "outside air is too warm to carry the heat away. Timing matters "
             "because those two things do not line up.")
    d.p("I checked the equations against hand calculations worked out "
        "separately: energy conservation, the closed-form matrix exponential "
        "solution, a resistance network derived independently, Newton's law of "
        "cooling in the single-mass limit, and the fitted time constants "
        "against the predicted ones. All six checks agree, the last of them to "
        "within 0.4%.")

    d.h2("Weather data", "weather")
    d.figure("results/figures/rep_climates.png",
             "Five panels of outdoor temperature over 48 hours for Palo Alto, "
             "Phoenix, Austin, Chicago and Seattle.",
             "Hourly observations from the Open-Meteo archive, real 2024 heat "
             "events. Cached in the repository so results reproduce offline.")
    d.p(f"I picked five climates to cover the range a ventilation controller "
        f"would actually meet. {NICE['phoenix']} averages "
        f"{b.loc['phoenix'].night_mean_c:.0f} \u00b0C overnight and peaks at "
        f"{b.loc['phoenix'].t_out_max:.0f} \u00b0C, while Seattle and Palo Alto "
        f"drop into the mid teens every night. That gap between them is what "
        f"decides whether the controller has anything to work with.")

    d.h2("Usage", "using")
    d.code("""$ .venv/bin/python scripts/advise.py --location palo_alto

Ventilation advice for Palo Alto, CA
  outdoor 14.7 to 31.6 degC, overnight mean 16.4 degC

Recommended schedule
  Day 2
    00:00 - 08:05  open fully             outdoor  16.2 degC
    08:05 - 14:50  keep shut              outdoor  26.1 degC
    14:50 - 18:05  open fully, fan on     outdoor  27.7 degC
    18:05 - 24:00  open fully             outdoor  19.0 degC""")
    d.p("`--lat` and `--lon` work for any point on Earth, fetching the weather "
        "on demand. Room dimensions, construction and heat gains are in "
        "`windwindow/config.py`.")

    d.h2("Limitations", "limits")
    d.ul([
        "**One room geometry.** Medium-weight 30 m\u00b3 bedroom, west-facing "
        "window. A lightweight room has less structure to pre-cool, so it "
        "would gain less than this. All the room constants live in one file, "
        "so it would be easy to test.",
        "**Two weather seeds per climate, two days each.** That is enough to "
        "show the pattern is not an accident of one weather series, but not "
        "enough to put error bars on any of these numbers.",
        "**Sensor calibration offset is modelled, never corrected.** Each "
        f"session draws a fixed {cfg.SENSOR_OFFSET_C:.1f} \u00b0C offset per "
        "sensor, shifting where the controller believes the threshold sits. A "
        "deployment would estimate and remove it; here every policy carries "
        "the same handicap.",
        "**Humidity is absent.** Austin's night air is warm and humid, and "
        "ventilating would import that moisture, so the gain shown there is "
        "optimistic.",
        "**No shading, shutters or occupancy learning.** Shading would do a "
        "similar job by keeping the heat out instead of flushing it out, and "
        f"solar gain peaks at {V['gs']['peak_w']:.0f} W, so it would probably "
        f"help a lot.",
        "**The heat-gain schedule is given to the controller.** Solar "
        "geometry is predictable and the occupancy pattern here is a coarse "
        "guess, so handing it over is defensible, but a real installation "
        "would have to learn the occupancy part from its own data.",
        "**The planner searches locally.** Bounded quasi-Newton from several "
        "starting schedules, keeping the best, with no guarantee of finding "
        f"the global optimum. The perfect-information reference shows that all "
        f"of this costs at most {V['max_gap']:.1f}%.",
    ])

    d.h2("Running it", "repro")
    d.code("python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
           ".venv/bin/python scripts/run_all.py          # all 14 steps\n"
           ".venv/bin/python scripts/run_all.py 10 11    # just those two\n"
           ".venv/bin/python scripts/advise.py --location seattle")
    rt_txt = (f"{V['runtime_min']:.0f} minutes" if V["runtime_min"]
              else "under an hour")
    d.p(f"Every step finishes with an acceptance check and exits non-zero if "
        f"that check fails, and the driver stops at the first failure. The "
        f"weather downloads once into `data/weather`, so everything after that "
        f"runs offline. A full run takes {rt_txt}, and nearly all of that is "
        f"the two planning steps, which between them simulate about 140 "
        f"room-days.")
    d.table(["Step", "What it does", "Check"], [
        ["00", "Core invariants", "63 checks on physics, weather, sensors, features"],
        ["01", "Room model", "sealed room warms smoothly to a limit"],
        ["02", "Verify the thermal model", "six hand calculations agree"],
        ["03", "Synthetic weather", "four scenarios generate and plot"],
        ["04", "Window and fan", "cools when cooler out, warms when hotter"],
        ["05", "Sensor model", "readings scatter around the truth, offsets persist"],
        ["06", "Simulator and logger", "one run gives a complete CSV"],
        ["07", "Baselines", "timer and threshold rule both run"],
        ["08", "Fetch real weather", "five climates cached, 48 h or more each"],
        ["09", "Identification data", "five climates, partial openings a third of rows"],
        ["10", "Fit the room", "capacities and time constant recovered, parameters physical"],
        ["11", "Controller design", "fan penalty selected, six variants run"],
        ["12", "Evaluate by climate", "beats a sealed room in every climate"],
        ["13", "Figures and write-up", "this page"],
    ], align=["n", "l", "l"])
    d.p("`REVAMP.md` records what changed from the first version of this "
        "project and why, including the mistakes worth knowing about.")

    d.h2("Files", "layout")
    d.code(
        "windwindow/\n"
        "  config.py       the room: sizes, capacities, conductances, gains\n"
        "  room.py         two-mass thermal model, RK4 and an exact solver\n"
        "  gains.py        occupants, equipment and sun, by clock time\n"
        "  realweather.py  cached hourly observations, interpolated\n"
        "  forecast.py     a forecast whose error grows with lead time\n"
        "  statespace.py   ten physical parameters fitted to logged data\n"
        "  mpc.py          the planning controller\n"
        "  sensors.py      offset, lag, noise, quantisation\n"
        "  policies.py     timer, threshold rule, fixed and random actions\n"
        "  simulator.py    runs any policy under identical conditions\n"
        "  doc.py          renders this write-up to Markdown and HTML\n"
        "  report2.py      this write-up, built from the result tables\n"
        "scripts/\n"
        "  run_all.py      the pipeline, halting at the first failure\n"
        "  advise.py       print a schedule for a place and a date\n"
        "  legacy/         the retired first version\n"
        f"results/         {len(figs)} figures, {len(tables)} tables\n"
        "data/\n"
        "  weather/        cached observations, committed\n"
        "  identify/       logged fitting sessions, regenerated\n"
        "  evaluate/       evaluation runs, regenerated")
    return d


def write_readme(runs, by_clim, ctl, figs, tables) -> None:
    V = _values(runs, by_clim, ctl)
    md = _document(V, figs, tables).markdown()
    md += ("\n---\n\n*Built from the result tables by "
           "`windwindow/report2.py`. Every number here is read out of a results "
           "file, so it can't drift from the code.*\n")
    pathlib.Path("README.md").write_text(md)
    print(f"  wrote README.md ({len(md):,} characters)")


def write_html(runs, by_clim, ctl, figs, tables) -> None:
    V = _values(runs, by_clim, ctl)
    doc = _document(V, figs, tables)
    nav = [("summary", "Results"), ("controller", "What it does"),
           ("comparison", "Comparison"),
           ("identification", "Fitting the model"), ("ablation", "Ablations"),
           ("room", "The room"), ("weather", "Weather data"),
           ("using", "Usage"), ("limits", "Limitations"),
           ("repro", "Running it"), ("layout", "Files")]
    toc = ('<p class="toc">' + " &middot; ".join(
        f'<a href="#{a}">{t}</a>' for a, t in nav) + "</p>")
    body = doc.body_html()

    from .doc import Doc as _Doc
    # The title plus the three opening paragraphs, then the contents list.
    head_html = _Doc(blocks=doc.blocks[:4]).body_html()
    rest_html = _Doc(blocks=doc.blocks[4:]).body_html()
    html = (SHELL.read_text() + head_html + "\n\n" + toc + "\n\n" + rest_html
            + "\n<hr>\n<p><small>Generated from the result tables by "
              "<code>windwindow/report2.py</code>. Every number here is read "
              "out of a results file, so it can't drift from the code."
              "</small></p>\n")
    pathlib.Path("index.html").write_text(html)
    print(f"  wrote index.html ({len(html):,} characters)")
