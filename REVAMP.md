# What changed, and why

The first version answered "did we hit the four targets we set in advance?".
Three of four, and the write-up spent most of its length on the fourth. That
isn't useful to anyone trying to decide whether to build or use this.

Here's what I changed to make it useful, including the parts I didn't see
coming.

## Planned from the start

| # | Change | Why |
|---|--------|-----|
| 1 | Real hourly weather from Open-Meteo, cached, five climates, 48-hour runs | Synthetic three-hour slices never contain the daily cycle a real install faces |
| 2 | A state-space model fitted from logged data | Works at any horizon and any action, instead of one number ten minutes out |
| 3 | A forecast whose error grows the further out you look | Lets the controller act on what's coming, not just what it can measure |
| 4 | A receding-horizon controller over continuous window and fan settings | Can pre-cool the structure and modulate the opening instead of just switching it |
| 5 | Evaluation reframed around absolute outcomes and the gap to a perfect-information reference | Answers "how much does this save me, and how close is it to the best possible" |
| 6 | `scripts/advise.py` | Makes it something you can run, not just a report |

## Things I found out along the way

**The room had to be rescaled, and this mattered more than anything I'd
planned.** The original rig was a 600 x 400 x 400 mm plywood box holding 66 L of
air with a 17.6 W heater: 267 W per cubic metre, roughly fifty times a real
bedroom. So the box was hotter than outdoors at every hour of every day, which
means opening the window was always right and there was no decision left. On
real Palo Alto weather the planning controller and "just leave it open" gave
the same answer to one decimal place. They were the same policy. The room is
now a 4 x 3 x 2.5 m bedroom of medium-weight construction with occupants and
afternoon sun. The box constants are still in
`windwindow/config_box_legacy.py`.

**Heat gains had to vary through the day.** One constant heater can't produce
the situation that makes timing interesting. Solar peaks in the late afternoon
and occupancy in the evening, so the room is loaded hardest exactly when the
outside air is least able to carry the heat away.

**The planning horizon had to cover a whole day.** I tried 7.5 hours first and
it lost to a plain threshold rule. Deciding at 2 am it could only see to about
half nine, so pre-cooling looked like pointless overcooling and it wouldn't do
it. At 23.5 hours the flush-seal-reopen cycle shows up on its own.

**The model had to be parameterised physically.** My unconstrained fit let the
air-side and structure-side wall couplings float independently, even though
physics ties their ratio to the ratio of the heat capacities. It came back with
a negative infiltration conductance and a structure 99% too light, and it still
predicted the validation sessions acceptably. A model like that will tell you
anything once you ask it about an action the data never contained, and that's
the only kind of question a planner ever asks.

## What didn't change

The physics, the sensor model and the simulator loop are as they were. Their
checks against hand calculations still pass, now on the rescaled room. The
timer and threshold-rule baselines are still there as comparisons.

## Rules I stuck to

- Results are measured, not chosen. Nothing was tuned to make a target pass.
- The fan penalty is fixed on one climate and one weather seed, then evaluated
  on climates and seeds it never saw.
- Forecast error is modelled rather than wished away, and the controller only
  ever sees the forecast and the noisy sensors.
- The optimality gap is measured against a controller handed the true room
  parameters and an exact forecast, so "close to optimal" means something.
- Four bugs I hit are written up rather than quietly fixed: the horizon that
  was too short, the warm start that drifted out of step with the clock, the
  unconstrained fit above, and a knee-selection rule that picked the worst
  setting on its own curve.
