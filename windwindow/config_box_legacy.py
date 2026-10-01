"""
Physical constants for the virtual room.

Everything here is derived from the room specs in PLAN.md section 9:
a 600 x 400 x 400 mm box, ~66 L of air inside, a 17.6 W heater.

The room is modelled as two "buckets of heat" joined by pipes:

    [heater] -> (T_in: inside air) <=> (T_wall: the plywood shell) <=> outdoors
                       |
                       +--------- window/fan opening ------------> outdoors

A bucket's "level" is its temperature. How wide a pipe is, is a thermal
conductance in watts per kelvin (W/K): how many watts flow for each degree
of temperature difference. How big a bucket is, is a thermal capacitance in
joules per kelvin (J/K): how many joules it takes to warm it by one degree.
"""

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
BOX_L, BOX_W, BOX_H = 0.600, 0.400, 0.400   # m, external dimensions
WALL_THICKNESS = 0.009                       # m, 9 mm plywood
VOLUME_IN = 0.066                            # m^3, ~66 L of air inside (PLAN sec. 9)

# Internal surface area of the shell: 2 of each face.
AREA_WALL = 2 * (BOX_L * BOX_W) + 2 * (BOX_L * BOX_H) + 2 * (BOX_W * BOX_H)  # 1.28 m^2

# --------------------------------------------------------------------------
# Material properties
# --------------------------------------------------------------------------
RHO_AIR, CP_AIR = 1.2, 1005.0        # kg/m^3, J/(kg.K)
RHO_PLY, CP_PLY, K_PLY = 600.0, 1600.0, 0.13   # kg/m^3, J/(kg.K), W/(m.K)

# Surface heat-transfer coefficients, W/(m^2.K).
# A "film coefficient" is how well heat crosses the thin layer of air stuck to
# a surface. Still air insulates; moving air scrubs that layer away, so the
# same wall exchanges heat several times faster when air is blowing over it.
H_IN = 4.0            # inside face, still air (window shut)
H_IN_FORCED = 20.0    # inside face with air moving over it at ~1.5 m/s
H_OUT = 8.0           # outside face, convection plus radiation
MIX_WINDOW = 0.35     # an open window alone reaches this fraction of full forced flow

# --------------------------------------------------------------------------
# Capacitances (bucket sizes), J/K
# --------------------------------------------------------------------------
C_AIR = RHO_AIR * CP_AIR * VOLUME_IN          # ~80 J/K -- the air alone is tiny
MASS_FITTINGS = 1.5                            # kg of internal mounts, sensors, trim
C_IN = C_AIR + MASS_FITTINGS * CP_PLY          # ~2480 J/K

# The shell: a lumped node at the mid-plane of the plywood.
MASS_WALL = AREA_WALL * WALL_THICKNESS * RHO_PLY   # ~6.9 kg
C_WALL = MASS_WALL * CP_PLY                        # ~11060 J/K

# --------------------------------------------------------------------------
# Conductances (pipe widths), W/K
# --------------------------------------------------------------------------
# Half the plywood sits between the wall node and each surface.
R_HALF_WALL = (WALL_THICKNESS / 2) / K_PLY     # m^2.K/W

# Inside air <-> wall node: inside air film in series with half the plywood.
K_IN_WALL_BASE = AREA_WALL / (1.0 / H_IN + R_HALF_WALL)     # ~4.50 W/K

# Wall node <-> outdoors: half the plywood in series with the outside film.
K_WALL_OUT = AREA_WALL / (R_HALF_WALL + 1.0 / H_OUT)        # ~8.02 W/K

# Inside air <-> outdoors with the window shut: cracks plus the pane itself.
K_LEAK = 0.30                                                # W/K

# --------------------------------------------------------------------------
# Window and fan
# --------------------------------------------------------------------------
# A moving air stream carries heat at rho*cp*(volume flow rate).
#
# FLOW_OPEN_MAX: buoyancy-driven flow through a single opening. The standard
# correlation Q ~ (1/3)*A*sqrt(g*dT*H/T) gives about 3 L/s for a 250x200 mm
# window with a 5 K indoor-outdoor difference, and more as the difference
# grows, so 4 L/s at full opening is in the right range.
#
# FLOW_FAN_MAX: a 40 mm, 2.5 W axial fan is rated around 8 CFM (3.8 L/s) in
# free air and delivers less than that against the resistance of a small
# opening. 3 L/s is 164 air changes an hour in a 66 L box, which is already
# brisk. (An earlier draft used 8 L/s -- 436 air changes an hour -- which is
# not something a 2.5 W fan can do.)
FLOW_OPEN_MAX = 0.0040      # m^3/s with the window fully open, no fan
FLOW_FAN_MAX = 0.0030       # m^3/s of extra flow with the fan at full speed
K_VENT_OPEN_MAX = RHO_AIR * CP_AIR * FLOW_OPEN_MAX          # ~4.82 W/K
K_VENT_FAN_MAX = RHO_AIR * CP_AIR * FLOW_FAN_MAX            # ~9.65 W/K

# THE KEY NON-OBVIOUS EFFECT.
# Opening the window does not only let outside air in. It also sets the inside
# air moving, which scrubs away the insulating film on the inside wall surface
# and makes heat move between the walls and the air several times faster. So
# hot walls dump their stored heat into the room much more quickly when the
# window is open. This is handled by h_in rising from H_IN to H_IN_FORCED
# (see room.h_in_eff), not by a fudge factor.
#
# Because the inside film (1/4 = 0.25 m^2.K/W still) is much larger than half
# the plywood (0.035 m^2.K/W), the wall coupling is film-limited: it is very
# sensitive to airflow. K_in_wall runs about 4.5 W/K shut, 9.2 W/K with the
# window open, and 15.1 W/K with the window open and the fan running.

# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------
HEATER_W = 17.6             # W, the heater from PLAN sec. 9
FAN_ELEC_W = 2.5            # W drawn by the fan at full speed
FAN_HEAT_FRAC = 1.0         # all of that ends up as heat in the room air

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
DT_PHYSICS = 1.0            # s, physics integration step
DT_SENSOR = 5.0             # s, how often the sensors are polled
DT_CONTROL = 30.0           # s, control + logging tick (PLAN: 30-second rows)
HORIZON_S = 600.0           # s, the 10-minute prediction horizon
SLOPE_WINDOW_S = 300.0      # s, the 5-minute window for slope_in / slope_out
COMFORT_C = 26.0            # degC, the threshold for goal G3

# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
SENSOR_NOISE_C = 0.10       # degC, per-reading random noise (1 sigma)
SENSOR_OFFSET_C = 0.30      # degC, per-session fixed calibration error (1 sigma)
SENSOR_TAU_S = 20.0         # s, sensor thermal lag
SENSOR_QUANT_C = 0.0625     # degC, digital resolution (a 12-bit DS18B20 step)

# --------------------------------------------------------------------------
# Controller thresholds (PLAN sec. 5, "Decision rule")
# --------------------------------------------------------------------------
OPEN_THRESHOLD_C = 0.15     # open when predicted benefit exceeds this
CLOSE_THRESHOLD_C = 0.05    # close when predicted benefit drops below this
MIN_DWELL_S = 300.0         # s, at least 5 min between moves
T_IN_FLOOR_C = 24.0         # degC, always close below this
FAN_MARGIN_C = 0.10         # the fan must beat window-only by this much
