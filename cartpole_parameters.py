"""
Configuration Parameters for CartPole Fuzzy Controller

This module centralizes all tunable parameters for the CartPole simulation
and fuzzy controller. Modifying these values affects controller behavior.

PARAMETER CATEGORIES
--------------------
1. Simulation Control: Timing and episode limits
2. CartPole Physical Limits: Environment boundaries from Gymnasium
3. Fuzzy Input Ranges: Universe of discourse for each fuzzy variable
4. Convergence Thresholds: Criteria for "balanced" state
5. Display Settings: Console output frequency

USAGE
-----
Import specific parameters:
    from cartpole_parameters import MAX_SIMULATION_STEPS, DT

Or import all:
    from cartpole_parameters import *

MODIFICATION GUIDELINES
-----------------------
- Changing fuzzy input ranges requires corresponding changes in fuzzy_controller.py
- Simulation parameters can be modified independently
- Physical limits should match Gymnasium's CartPole if using standard environment
"""

# =============================================================================
# SIMULATION CONTROL PARAMETERS
# =============================================================================
# These control how the simulation runs, independent of the physics or controller.

# Time step for simulation display and data logging (seconds)
# Matches Gymnasium's CartPole internal time step (tau = 0.02)
# 50 Hz update rate = 20ms per step
DT = 0.02

# Maximum steps per episode before forced termination
# At 50 Hz, 1000 steps = 20 seconds of simulated time
# Longer episodes test sustained balance; shorter episodes speed up training
MAX_SIMULATION_STEPS = 1000

# Default number of episodes for batch runs
# Used by scripts that run multiple episodes for statistics
NUM_EPISODES = 10

# =============================================================================
# CARTPOLE PHYSICAL STATE LIMITS
# =============================================================================
# These match Gymnasium's CartPole environment internal limits.
# Used for observation space bounds, NOT for fuzzy controller inputs.
# The fuzzy controller uses separate (often narrower) ranges below.

# Cart position limits (meters)
# Gymnasium allows ±4.8m for observation space, but terminates at ±2.4m
CART_POSITION_MIN = -4.8
CART_POSITION_MAX = 4.8

# Cart velocity limits (m/s)
# Theoretical max based on force and physics; rarely reached in practice
CART_VELOCITY_MIN = -10.0
CART_VELOCITY_MAX = 10.0

# Pole angle limits (radians)
# ±0.418 rad = ±24 degrees (observation space)
# Episode terminates at ±0.2095 rad = ±12 degrees
POLE_ANGLE_MIN = -0.418
POLE_ANGLE_MAX = 0.418

# Pole angular velocity limits (rad/s)
# Practical limits during normal operation
POLE_ANGULAR_VELOCITY_MIN = -5.0
POLE_ANGULAR_VELOCITY_MAX = 5.0

# =============================================================================
# FUZZY CONTROLLER INPUT RANGES (Universe of Discourse)
# =============================================================================
# These define the range of values the fuzzy controller considers.
# Ranges are often extended beyond physical limits to provide "headroom"
# for membership functions at extreme values.

# Pole angle input range for fuzzy controller (radians)
# Extended beyond ±0.2095 (12°) failure threshold to allow meaningful
# membership values near the failure boundary
# ±0.5 rad = ±28.6 degrees
ANGLE_ERROR_RANGE_MIN = -0.5
ANGLE_ERROR_RANGE_MAX = 0.5

# Angular velocity input range for fuzzy controller (rad/s)
# Covers typical operation range during balancing
# Extreme velocities beyond this are clipped
DELTA_ANGLE_RANGE_MIN = -3.0
DELTA_ANGLE_RANGE_MAX = 3.0

# Force output range for fuzzy controller (Newtons)
# Matches CartPole's maximum applicable force
# Symmetric for bidirectional control
CONTROL_RANGE_MIN = -10.0
CONTROL_RANGE_MAX = 10.0

# Cart position input range for fuzzy controller (meters)
# Set to exactly match termination boundaries (±2.4m)
# No extension needed - positions beyond this trigger immediate failure
CART_POSITION_RANGE_MIN = -2.4
CART_POSITION_RANGE_MAX = 2.4

# Cart velocity input range for fuzzy controller (m/s)
# Narrower than physical max - typical velocities during control stay within ±1 m/s
# Extended to ±3 to handle aggressive recovery maneuvers
CART_VELOCITY_RANGE_MIN = -3.0
CART_VELOCITY_RANGE_MAX = 3.0

# =============================================================================
# CONVERGENCE THRESHOLDS
# =============================================================================
# Define what constitutes a "balanced" or "stable" state.
# Used for determining when the controller has achieved its goal.

# Angle threshold for considering pole "vertical" (radians)
# 0.02 rad = ~1.1 degrees - very tight tolerance
CONVERGENCE_THRESHOLD_ANGLE = 0.02

# Angular velocity threshold for considering pole "stationary" (rad/s)
# Pole must be both near-vertical AND barely moving to be "converged"
CONVERGENCE_THRESHOLD_ANGULAR_VEL = 0.1

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
# Control console output verbosity during simulation.

# Print status every N steps during simulation
# Lower values = more verbose output
# Set to MAX_SIMULATION_STEPS to only print at end
DISPLAY_INTERVAL = 50
