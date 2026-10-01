# Retired steps

These scripts belong to the first version of the project, which simulated the
original 600 x 400 x 400 mm plywood box and learned a single number: the
change in indoor temperature over the following ten minutes.

They are kept because the work is sound and because the greedy controller they
build is a useful comparison point, but they are no longer part of the
pipeline, for two reasons.

The box made the control problem vacuous. A 17.6 W load in 66 L of air is
267 W per cubic metre, about fifty times a real bedroom, so the box sat above
the outdoor temperature at every hour of every day. Ventilating was therefore
always correct and there was nothing to decide. Measured on real Palo Alto
weather, a planning controller and a policy that simply held the window open
scored identically to one decimal place, because they were the same policy.

The ten-minute predictor cannot plan. It compares two actions over the next
ten minutes, which is the right thing to do if nothing beyond ten minutes
matters. In a room whose structure stores 1.5 MJ/K it does: heat removed
overnight is capacity to absorb the following afternoon, and that payoff
arrives hours after the decision that earns it.

The constants for the box are preserved in `windwindow/config_box_legacy.py`.
To run these steps, point `windwindow/config.py` at those values first.
