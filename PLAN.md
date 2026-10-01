# PRD: WindWindow Sim

A software-only version of WindWindow. A virtual room, virtual weather, and a controller that decides when to open the window.

---

## 1. What we're building

A Python project that simulates a small room with a window and fan. A model predicts whether opening the window will cool the room over the next 10 minutes, and we compare it against simpler strategies.

**Why software-only still works:** the hard part of the original project was never the plywood. It was the decision logic: knowing that hot walls can re-warm the air even when it's cooler outside.

---

## 2. Goals (same as the hardware version)

| # | Goal | How we measure it |
|---|------|-------------------|
| G1 | Predict the 10-min temperature change to within 0.25 °C on average | Mean absolute error (MAE) on held-out sessions |
| G2 | Make the right open/closed call at least 90% of the time, and beat "open if cooler outside" | Accuracy on 30 paired test trials |
| G3 | Spend at least 30% less time above 26 °C than the timer | Degree-minutes over 26 °C |
| G4 | Run the fan at least 30% less than the timer | Fan minutes and Wh |

**Non-goals (for now):** real hardware, humidity, sunlight, multiple rooms, a web interface. These are stretch goals in section 8.

---

## 3. System overview

```
Weather generator -> Virtual room (physics) -> Virtual sensors (noisy)
                          ^                            |
                          |                            v
                   Actions (window, fan) <---- Controller (policy)
                                                       |
                                                       v
                                                   CSV logs -> Model training -> Results
```

---

## 4. Modules and why each one exists

| Module | What it does | Real-world analogy |
|--------|--------------|--------------------|
| **Room model** | Tracks indoor air temp, wall temp, and heat from the heater. Window and fan change how heat flows | Three buckets of water joined by pipes. Temperature is the water level, and opening the window widens a pipe |
| **Weather generator** | Produces outdoor temperature for the scenarios (cool, hot, hot-then-cooling, hot-walls-cold-outside) | A "scenario" button in a video game |
| **Virtual sensors** | Adds noise and small offsets (about 0.3 °C) to the true values | A bathroom scale that's always slightly off |
| **Policies** | Timer, simple rule, and model-based controller | Three different thermostats to compare |
| **Simulator loop** | Steps time forward, asks the policy for an action, logs everything | A referee running each game the same way |
| **Data logger** | Saves 30-second averaged rows to CSV | A lab notebook |
| **Model** | Baselines, ridge regression, and LightGBM | A weather forecaster at 3 skill levels |
| **Evaluator** | Computes the metrics in section 2 | The scoreboard |

---

## 5. Key design decisions (carried over from the hardware project)

- **Random actions when collecting training data.** Otherwise the model never sees what happens when you open the window on a hot day.
- **Split by whole sessions, not random rows.** Rows 30 seconds apart are nearly identical, so a random split leaks test data into training.
- **Keep only rows where the action didn't change during the 10-minute window.** Otherwise the outcome is a mix of two actions.
- **Different open and close thresholds.** Open at 0.15 °C benefit, close at 0.05 °C, and wait at least 5 min between moves. This stops the window from flip-flopping.
- **Same 9 model inputs**, including wall temperature and outdoor trend, which are what make it smarter than "open if cooler outside."

### Decision rule

```
benefit = predict(closed) - predict(open + fan)
open   if benefit > 0.15 °C
close  if benefit < 0.05 °C
wait at least 5 min between moves
close  if T_in < 24 °C
fan on only if it cools > 0.1 °C more than just having the window open
```

### Model inputs

Let `u` = how open the window is (0 to 1), `f` = fan speed (0 to 1), `d` = outdoor minus indoor temperature.

| Input | What it captures |
|-------|------------------|
| `d`, `u·d`, `u·f·d` | Heat moving through the walls, the window, and the fan |
| `Twall - Tin`, heater | Heat stored in the walls, heat from the heater |
| `slope_in`, `slope_out` | Whether indoor and outdoor temps rose or fell over the last 5 min |
| `u·(Twall - Tout)`, `u·slope_out` | Walls re-warming the room, and the outdoor trend, when the window is open |

---

## 6. Step-by-step milestones

Each step ends with a "done when" check. We do not move on until it passes.

| Step | Task | Why | Done when |
|------|------|-----|-----------|
| 0 | Set up the project (folder, Python, libraries) | Everything else depends on a working environment | A "hello world" script runs |
| 1 | Build the room model | It's the foundation. Bad physics makes every result meaningless | A closed room with the heater on warms up smoothly and levels off |
| 2 | Validate the room model | A simulation is only trustworthy if it matches known physics | Results match a hand-calculated heat equation |
| 3 | Build the weather generator | The model needs varied conditions to learn from | All 4 scenarios plot correctly |
| 4 | Add window and fan effects | This is the actual decision the project is about | Opening in cool weather cools the room, and opening in hot weather warms it |
| 5 | Add virtual sensors | Real sensors are noisy, and the controller must cope | Plotted readings wobble around the true values |
| 6 | Build the simulator loop and logger | Runs any policy and saves data | One 3-hour run produces a CSV |
| 7 | Build the timer and simple-rule policies | These are the baselines | Both run and produce results |
| 8 | Generate training data (15+ random-action sessions) | The model learns from this | 15 CSVs saved |
| 9 | Build features and the train/test split | Wrong splitting gives fake-good results | Test sessions never appear in training |
| 10 | Train baselines, ridge, and LightGBM | Shows whether a complex model is worth it | MAE reported for all 4 |
| 11 | Build the model-based controller | This is WindWindow itself | It runs a full simulation |
| 12 | Run the full experiment (3 policies x 4 scenarios x 3 repeats) | Produces the comparison | 36 runs logged |
| 13 | Evaluate against G1 to G4 | The actual answer to "did it work?" | Results table filled in |
| 14 | Write up and make figures | Turns the work into something shareable | Page or README complete |

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| The simulator is too simple and results don't mean anything | Validate in Step 2 and use real weather data as input |
| The model "cheats" by seeing test data | Session-level split (Step 9) |
| The simulator is so easy that every policy looks great | Add sensor noise, delays, and wall heat storage |
| Scope creeps | Keep section 8 items off the list until Step 13 is done |

---

## 8. Stretch goals (only after Step 13)

- Real hourly weather from an API
- Humidity and sunlight
- Different room sizes, to test whether the model generalizes
- An interactive web page with an animated room
- Report prediction error and paired-trial accuracy honestly (the original page marks these as not yet done)

---

## 9. Assumptions to confirm

1. **Python first**, with the web page as a stretch goal.
2. Room specs stay the same: 600 x 400 x 400 mm, about 66 L inside, 17.6 W heater.
3. Simulated time runs faster than real time, so a 3-hour run takes seconds.

---

## 10. Working agreement

- We build **one small chunk of code at a time**.
- You confirm each chunk works before we move on.
- For each step, I explain **why** we're doing it, not just what to type.