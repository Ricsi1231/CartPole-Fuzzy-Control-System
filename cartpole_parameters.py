"""
Configuration Parameters for CartPole Fuzzy Controller

This module contains all tunable parameters for the fuzzy logic control system.
Parameters are organized into logical groups for easy modification and tuning.
"""

# =============================================================================
# OBSERVATION ARRAY INDICES
# Maps the Gymnasium CartPole observation array to meaningful variable names.
# The observation array from env.step() returns: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# =============================================================================
OBS_CART_POS = 0    # Index 0: Cart position (meters from center)
OBS_CART_VEL = 1    # Index 1: Cart velocity (m/s, positive = moving right)
OBS_POLE_ANGLE = 2  # Index 2: Pole angle (radians, positive = tilted right)
OBS_POLE_VEL = 3    # Index 3: Pole angular velocity (rad/s, positive = rotating right)

# =============================================================================
# SIMULATION TIMING AND EPISODE PARAMETERS
# Control the simulation loop timing and episode structure.
# =============================================================================
DT = 0.02                     # Time step in seconds (20ms) - used for integral control accumulation
MAX_SIMULATION_STEPS = 1000   # Maximum steps per episode (1000 steps = 20 seconds at DT=0.02)
NUM_EPISODES = 1             # Default number of episodes to run

# =============================================================================
# ENVIRONMENT STATE LIMITS (from Gymnasium CartPole)
# These are the physical boundaries of the CartPole system.
# Exceeding these limits causes episode termination.
# =============================================================================
CART_POSITION_MIN = -4.8            # Cart left boundary (meters)
CART_POSITION_MAX = 4.8             # Cart right boundary (meters)
CART_VELOCITY_MIN = -10.0           # Maximum leftward cart speed (m/s)
CART_VELOCITY_MAX = 10.0            # Maximum rightward cart speed (m/s)
POLE_ANGLE_MIN = -0.418             # Maximum left tilt (~24 degrees) - episode fails beyond this
POLE_ANGLE_MAX = 0.418              # Maximum right tilt (~24 degrees) - episode fails beyond this
POLE_ANGULAR_VELOCITY_MIN = -5.0    # Maximum angular speed in left direction (rad/s)
POLE_ANGULAR_VELOCITY_MAX = 5.0     # Maximum angular speed in right direction (rad/s)

# =============================================================================
# FUZZY VARIABLE UNIVERSE OF DISCOURSE
# These define the input/output ranges for fuzzy membership function quantization.
# The STEP values determine the resolution (smaller = finer granularity but more computation).
# Input values are clipped to these ranges before fuzzy inference.
# =============================================================================

# Pole Angle fuzzy variable universe (radians)
ANGLE_ERROR_RANGE_MIN = -1    # Minimum angle for fuzzy processing
ANGLE_ERROR_RANGE_MAX = 1     # Maximum angle for fuzzy processing
ANGLE_STEP = 0.01             # Quantization step (200 discrete points in universe)

# Angular Velocity fuzzy variable universe (rad/s)
DELTA_ANGLE_RANGE_MIN = -3.0  # Minimum angular velocity for fuzzy processing
DELTA_ANGLE_RANGE_MAX = 3.0   # Maximum angular velocity for fuzzy processing
DELTA_ANGLE_STEP = 0.1        # Quantization step (60 discrete points)

# Force Output fuzzy variable universe (Newtons)
CONTROL_RANGE_MIN = -10.0     # Maximum left push force
CONTROL_RANGE_MAX = 10.0      # Maximum right push force
CONTROL_STEP = 0.1            # Quantization step (200 discrete points)

# Cart Position fuzzy variable universe (meters)
CART_POSITION_RANGE_MIN = -3.0  # Left boundary for fuzzy processing
CART_POSITION_RANGE_MAX = 3.0   # Right boundary for fuzzy processing
CART_POSITION_STEP = 0.1        # Quantization step (60 discrete points)

# Cart Velocity fuzzy variable universe (m/s)
CART_VELOCITY_RANGE_MIN = -1.0  # Maximum left speed for fuzzy processing
CART_VELOCITY_RANGE_MAX = 1.0   # Maximum right speed for fuzzy processing
CART_VELOCITY_STEP = 0.1        # Quantization step (20 discrete points)

# =============================================================================
# CONVERGENCE AND STABILITY THRESHOLDS
# Define when the pole is considered "stable" for control mode decisions.
# =============================================================================
CONVERGENCE_THRESHOLD_ANGLE = 0.02        # Angle threshold for stability (~1.15 degrees)
CONVERGENCE_THRESHOLD_ANGULAR_VEL = 0.1   # Angular velocity threshold for stability (rad/s)

# =============================================================================
# DISPLAY PARAMETERS
# Control console output frequency during simulation.
# =============================================================================
DISPLAY_INTERVAL = 50   # Print status every N steps (50 steps = 1 second at DT=0.02)
SEPARATOR_WIDTH = 60    # Width of separator lines in console output

# =============================================================================
# INITIAL STATE RANDOMIZATION
# Random starting conditions for each episode to test controller robustness.
# The pole starts nearly vertical with small random perturbations.
# =============================================================================
INIT_CART_POS_MIN = -0.3      # Initial cart position minimum (meters)
INIT_CART_POS_MAX = 0.3       # Initial cart position maximum (meters)
INIT_CART_VEL_MIN = -0.1      # Initial cart velocity minimum (m/s)
INIT_CART_VEL_MAX = 0.1       # Initial cart velocity maximum (m/s)
INIT_POLE_ANGLE_MIN = -0.03   # Initial pole angle minimum (~1.7 degrees)
INIT_POLE_ANGLE_MAX = 0.03    # Initial pole angle maximum (~1.7 degrees)
INIT_POLE_VEL_MIN = -0.1      # Initial pole angular velocity minimum (rad/s)
INIT_POLE_VEL_MAX = 0.1       # Initial pole angular velocity maximum (rad/s)

# =============================================================================
# MEMBERSHIP FUNCTION PARAMETERS - POLE ANGLE (radians)
# Define how pole angle is classified into fuzzy linguistic terms.
# Trapezoidal [a,b,c,d]: membership=1 between b and c, 0 outside a and d
# Triangular [a,b,c]: peak membership=1 at b, 0 at a and c
# Designed with proper overlap for smooth fuzzy transitions.
# =============================================================================
ANGLE_NL = [-1.0, -1.0, -0.10, -0.03]   # Negative Large: pole tilted far left (trapezoidal)
ANGLE_NS = [-0.08, -0.04, 0.0]          # Negative Small: pole slightly tilted left (triangular)
ANGLE_Z = [-0.03, 0.0, 0.03]            # Zero: pole nearly vertical (triangular, peak at 0)
ANGLE_PS = [0.0, 0.04, 0.08]            # Positive Small: pole slightly tilted right (triangular)
ANGLE_PL = [0.03, 0.10, 1.0, 1.0]       # Positive Large: pole tilted far right (trapezoidal)

# =============================================================================
# MEMBERSHIP FUNCTION PARAMETERS - ANGULAR VELOCITY (rad/s)
# Define how pole rotation speed is classified into fuzzy linguistic terms.
# =============================================================================
ANG_VEL_N = [-3.0, -3.0, -0.2, -0.02]  # Negative: pole rotating left/counterclockwise (trapezoidal)
ANG_VEL_Z = [-0.08, 0.0, 0.08]         # Zero: pole nearly stationary (triangular, peak at 0)
ANG_VEL_P = [0.02, 0.2, 3.0, 3.0]      # Positive: pole rotating right/clockwise (trapezoidal)

# =============================================================================
# MEMBERSHIP FUNCTION PARAMETERS - CART POSITION (meters)
# Define how cart horizontal position is classified for centering control.
# =============================================================================
CART_POS_N = [-3.0, -3.0, -1.0, -0.5]  # Negative: cart is left of center (trapezoidal)
CART_POS_Z = [-0.7, 0.0, 0.7]          # Zero: cart is centered (triangular, ±0.7m tolerance)
CART_POS_P = [0.5, 1.0, 3.0, 3.0]      # Positive: cart is right of center (trapezoidal)

# =============================================================================
# MEMBERSHIP FUNCTION PARAMETERS - CART VELOCITY (m/s)
# Define how cart movement speed is classified for damping control.
# Designed with proper symmetric overlap for smooth fuzzy transitions.
# =============================================================================
CART_VEL_N = [-1.0, -1.0, -0.2, 0.0]    # Negative: cart moving left (trapezoidal)
CART_VEL_Z = [-0.2, 0.0, 0.2]           # Zero: cart nearly stationary (triangular)
CART_VEL_P = [0.0, 0.2, 1.0, 1.0]       # Positive: cart moving right (trapezoidal)

# =============================================================================
# MEMBERSHIP FUNCTION PARAMETERS - FORCE OUTPUT (Newtons)
# Define the output force levels produced by defuzzification.
# =============================================================================
FORCE_NL = [-10, -10, -7, -3]  # Negative Large: strong left push (trapezoidal)
FORCE_NS = [-5, -2.5, 0]       # Negative Small: weak left push (triangular)
FORCE_Z = [-0.3, 0, 0.3]       # Zero: no force applied (triangular)
FORCE_PS = [0, 2.5, 5]         # Positive Small: weak right push (triangular)
FORCE_PL = [3, 7, 10, 10]      # Positive Large: strong right push (trapezoidal)

# =============================================================================
# INTEGRAL CONTROLLER PARAMETERS
# Secondary controller for cart position drift correction.
# Only active when pole is nearly vertical (below INTEGRAL_ANGLE_THRESHOLD).
# This helps center the cart without destabilizing pole balance.
# =============================================================================
INTEGRAL_GAIN = 0.0               # Gain for integral control (0.0 = disabled, increase to enable centering)
INTEGRAL_LIMIT = 6.0              # Anti-windup limit: max integral accumulation (meters*seconds)
INTEGRAL_ANGLE_THRESHOLD = 0.1    # Angle threshold to enable integral (~5.7 degrees)
INTEGRAL_DECAY = 0.95             # Decay factor when pole unstable (prevents sudden jumps)
