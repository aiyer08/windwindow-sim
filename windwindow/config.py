"""
Physical constants for the simulated room.

WHAT CHANGED, AND WHY
---------------------
This used to be the original hardware rig: a 600 x 400 x 400 mm plywood box
holding 66 L of air, with a 17.6 W heater inside. Those constants are still in
config_box_legacy.py.

The box made the control problem pointless. 17.6 W in 66 L is 267 W per cubic
metre. A real bedroom is closer to 5. So the box sat above the outdoor
temperature at every hour of every day, which means ventilating was always the
right answer and there was nothing to decide. On real Palo Alto weather a
planning controller and a policy that just held the window open scored the same
to one decimal place, because they were the same policy.

So this file describes a real room instead: a 4 x 3 x 2.5 m bedroom of
medium-weight construction, with a window that opens, a fan, occupants, and sun
on the window in the afternoon. In that room the answer genuinely varies. A
room pre-cooled overnight to 22 degC should be shut tight against a 31 degC
afternoon, and the same room should be flushed hard at 5 am. Which applies
depends on the time of day, the thermal mass, and what the weather is about to
do, and that's what makes it worth planning.

The equations didn't change, so the verification in step 2 still applies. Only
the constants and the heat-gain model are different.

The room is two lumps of thermal mass joined by conductive paths:

    [gains] --> ( indoor air ) <--> ( structure ) <--> outdoors
                      |
                      +------- window and fan -------> outdoors

A thermal conductance is in watts per kelvin: heat flow per degree of
difference. A thermal capacitance is in joules per kelvin: energy to warm the
mass by one degree.
"""

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
ROOM_L, ROOM_W, ROOM_H = 4.0, 3.0, 2.5       # m
VOLUME_IN = ROOM_L * ROOM_W * ROOM_H          # 30 m^3

# Envelope exposed to outdoors: four walls and the ceiling. The floor is
# excluded because it faces another conditioned storey rather than outside.
AREA_WALL = 2 * (ROOM_L * ROOM_H) + 2 * (ROOM_W * ROOM_H) + ROOM_L * ROOM_W

# --------------------------------------------------------------------------
# Material properties
# --------------------------------------------------------------------------
RHO_AIR, CP_AIR = 1.2, 1005.0                 # kg/m^3, J/(kg.K)
R_ENVELOPE = 3.0                              # m^2.K/W, insulated construction
R_HALF_WALL = R_ENVELOPE / 2.0                # node sits at the mid-plane

# Surface film coefficients, W/(m^2.K). Still air insulates; moving air
# scrubs the boundary layer away, so the structure exchanges heat with the
# room several times faster once the window is open. Indoor values are lower
# than in the original box because room-scale air velocities are lower.
H_IN = 3.0                                    # still air indoors
H_IN_FORCED = 12.0                            # air moving across the surfaces
H_OUT = 20.0                                  # outdoors, wind exposed
MIX_WINDOW = 0.40                             # fraction of full mixing from the window alone

# --------------------------------------------------------------------------
# Capacitances, J/K
# --------------------------------------------------------------------------
C_AIR = RHO_AIR * CP_AIR * VOLUME_IN          # ~36 kJ/K
MASS_FITTINGS_J_K = 110_000                   # furniture, books, fittings
C_IN = C_AIR + MASS_FITTINGS_J_K              # ~146 kJ/K

# Plaster, framing and the inner leaf of the envelope. 1.5 MJ/K gives a
# 7.4 hour structural time constant and damps the outdoor daily swing to
# about a third indoors, which is what medium-weight construction does.
C_WALL = 1_500_000.0

# --------------------------------------------------------------------------
# Conductances, W/K
# --------------------------------------------------------------------------
K_IN_WALL_BASE = AREA_WALL / (1.0 / H_IN + R_HALF_WALL)     # ~25.6 W/K
K_WALL_OUT = AREA_WALL / (R_HALF_WALL + 1.0 / H_OUT)        # ~30.3 W/K

# Infiltration with the window shut, at 0.8 air changes per hour.
ACH_LEAK = 0.8
K_LEAK = RHO_AIR * CP_AIR * (ACH_LEAK * VOLUME_IN / 3600.0)  # ~8 W/K

# --------------------------------------------------------------------------
# Window and fan
# --------------------------------------------------------------------------
# Single-sided natural ventilation through an openable window of about 1.2 m2.
# The buoyancy correlation Q = (1/3)*A*sqrt(g*dT*H/T) gives roughly 20 air
# changes an hour at a 5 K difference; 14 is the conservative figure used here.
ACH_OPEN_MAX = 14.0
# A 25 W window fan moving about 500 m^3/h, which is 16 air changes an hour.
ACH_FAN_MAX = 16.0
K_VENT_OPEN_MAX = RHO_AIR * CP_AIR * (ACH_OPEN_MAX * VOLUME_IN / 3600.0)
K_VENT_FAN_MAX = RHO_AIR * CP_AIR * (ACH_FAN_MAX * VOLUME_IN / 3600.0)

# --------------------------------------------------------------------------
# Heat gains
# --------------------------------------------------------------------------
# Occupancy and equipment. A sleeping or resting person is about 80 W, and
# evening activity adds lighting, cooking heat and electronics.
GAIN_BASE_W = 70.0
GAIN_EVENING_W = 110.0
GAIN_EVENING_HOUR = 20.0
GAIN_EVENING_WIDTH_H = 3.0

# Afternoon sun through the window. A 1.5 m2 west-facing window with blinds
# part-drawn gives an effective aperture near 0.5 m2; clear-sky irradiance on
# a west facade peaks around 600 W/m2 in late afternoon.
SOLAR_PEAK_W = 250.0
SOLAR_PEAK_HOUR = 16.0
SOLAR_WIDTH_H = 3.6
SOLAR_FIRST_HOUR, SOLAR_LAST_HOUR = 6.0, 20.0

# Kept for the legacy box scripts and as the default constant gain.
HEATER_W = GAIN_BASE_W
FAN_ELEC_W = 25.0                             # a 25 W window fan
FAN_HEAT_FRAC = 1.0                           # its power ends up as heat in the room

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
DT_PHYSICS = 5.0            # s, integration step
DT_SENSOR = 15.0            # s, sensor polling interval
DT_CONTROL = 300.0          # s, control and logging interval (5 minutes)
HORIZON_S = 600.0           # s, the legacy ten-minute prediction horizon
SLOPE_WINDOW_S = 1800.0     # s, window for trend features
COMFORT_C = 26.0            # degC, the overheating threshold
# Night ventilation deliberately runs a room cool so its structure can absorb
# the next day's heat, so the floor has to sit below normal comfort or it
# blocks the strategy. 17 degC is a reasonable minimum for a sleeping or
# unoccupied room.
T_IN_FLOOR_C = 17.0

# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
SENSOR_NOISE_C = 0.10
SENSOR_OFFSET_C = 0.30
SENSOR_TAU_S = 60.0
SENSOR_QUANT_C = 0.0625

# --------------------------------------------------------------------------
# Legacy greedy-controller thresholds (kept for the baseline comparison)
# --------------------------------------------------------------------------
OPEN_THRESHOLD_C = 0.15
CLOSE_THRESHOLD_C = 0.05
MIN_DWELL_S = 900.0
FAN_MARGIN_C = 0.10
