"""
Fuzzy Logic Controller for CartPole Balancing Problem

This module implements a Mamdani-type fuzzy inference system that uses all 4 state
variables from the CartPole environment to compute a continuous force output.

CONTROL PHILOSOPHY
------------------
The controller implements a hierarchical control strategy where pole balancing takes
absolute priority over cart position correction. This design recognizes that:
1. A fallen pole is an immediate failure condition (terminates at ±12 degrees)
2. Cart position drift is only dangerous when approaching boundaries (±2.4m)
3. Position correction forces can destabilize the pole if applied too aggressively

FORCE DIRECTION CONVENTION
--------------------------
- Positive force (+) = Push cart RIGHT = Pole tips LEFT = Counteracts RIGHT tilt
- Negative force (-) = Push cart LEFT  = Pole tips RIGHT = Counteracts LEFT tilt

TUNING HISTORY
--------------
- Initial version: 24 rules, average ~140 steps per episode
- Tuned version: 25 rules, average 390-450 steps, with episodes reaching 500+

KEY IMPROVEMENTS FROM TUNING
----------------------------
1. Narrower "zero" zones for angle (±0.006 rad) and angular velocity (±0.08 rad/s)
   for earlier instability detection
2. Earlier cart position activation (0.05m instead of 0.5m) for drift prevention
3. Weakened position correction rules (PS/NS instead of PL/NL) to avoid pole destabilization
4. Stronger intermediate force outputs (PS/NS peak at 2.5N instead of 1.5N)
5. Combined rules that boost force when pole tilt and cart offset require same correction
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from cartpole_parameters import (ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX,
                                  DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX,
                                  CONTROL_RANGE_MIN, CONTROL_RANGE_MAX,
                                  CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX,
                                  CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX)


class FuzzyCartPoleController:
    """
    Fuzzy logic controller for CartPole balancing.

    Uses all 4 state variables as inputs:
    - x (cart position)
    - x_dot (cart velocity)
    - theta (pole angle)
    - theta_dot (pole angular velocity)
    """

    def __init__(self):
        # ====================================================================
        # UNIVERSE OF DISCOURSE DEFINITIONS
        # ====================================================================
        # Each universe defines the range and resolution of a fuzzy variable.
        # The step size affects both computational cost and control precision.
        # ====================================================================

        # POLE ANGLE: -0.5 to 0.5 radians (~-28.6° to +28.6°)
        # Range: Extended beyond failure threshold (±0.2095 rad = ±12°) to allow
        #        meaningful membership values near failure. Provides "headroom"
        #        for NL/PL membership functions.
        # Step: 0.001 rad (~0.057°) - Fine resolution needed for tight Z zone
        self.angle = ctrl.Antecedent(np.arange(ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX, 0.001), 'angle')

        # ANGULAR VELOCITY: -3.0 to 3.0 rad/s
        # Range: Typical velocities stay within ±2 rad/s during balancing.
        #        Extended to ±3 rad/s to handle recovery from large disturbances.
        # Step: 0.01 rad/s - Balances precision with computational efficiency
        self.angular_velocity = ctrl.Antecedent(np.arange(DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX, 0.01), 'angular_velocity')

        # CART POSITION: -2.4 to 2.4 meters
        # Range: Exactly matches CartPole termination boundaries.
        #        No extension needed - position beyond this triggers failure.
        # Step: 0.01 m (1 cm) - Adequate for position tracking
        self.cart_position = ctrl.Antecedent(np.arange(CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX, 0.01), 'cart_position')

        # CART VELOCITY: -3.0 to 3.0 m/s
        # Range: Extended to handle aggressive corrections during recovery.
        #        Typical velocities during stable control stay within ±1 m/s.
        # Step: 0.01 m/s - Same resolution as angular velocity
        self.cart_velocity = ctrl.Antecedent(np.arange(CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX, 0.01), 'cart_velocity')

        # FORCE OUTPUT: -10 to 10 Newtons
        # Range: Standard CartPole maximum force magnitude.
        #        Symmetric range for bidirectional control.
        # Step: 0.1 N - Coarser than inputs (sufficient for smooth output)
        self.force = ctrl.Consequent(np.arange(CONTROL_RANGE_MIN, CONTROL_RANGE_MAX, 0.1), 'force')

        # ====================================================================
        # MEMBERSHIP FUNCTION DEFINITIONS
        # ====================================================================
        # Design Philosophy:
        # - Triangular (trimf) for precise intermediate states
        # - Trapezoidal (trapmf) for extreme states (saturating behavior)
        # - Overlapping regions enable smooth transitions between rules
        # ====================================================================

        # --------------------------------------------------------------------
        # POLE ANGLE MEMBERSHIP FUNCTIONS (5 terms)
        # --------------------------------------------------------------------
        # These are the most critical MFs - pole angle is the primary state.
        # Very narrow "Zero" zone (0.012 rad = 0.69°) ensures immediate response
        # to ANY visible deviation from vertical.
        #
        # Term Boundaries (radians / degrees):
        #   NL: -0.5 to -0.015  (-28.6° to -0.86°)  - pole tilted far left
        #   NS: -0.04 to -0.003 (-2.3° to -0.17°)   - pole slightly left
        #   Z:  -0.006 to 0.006 (-0.34° to 0.34°)   - pole nearly vertical
        #   PS: 0.003 to 0.04   (0.17° to 2.3°)     - pole slightly right
        #   PL: 0.015 to 0.5    (0.86° to 28.6°)    - pole tilted far right
        # --------------------------------------------------------------------

        # NL: Pole tilted far LEFT - needs strong rightward push
        # Trapezoidal with saturation at extreme angles
        self.angle['NL'] = fuzz.trapmf(self.angle.universe, [-0.5, -0.5, -0.05, -0.015])

        # NS: Pole slightly LEFT - needs gentle rightward push
        # Triangular for precise small-angle detection
        self.angle['NS'] = fuzz.trimf(self.angle.universe, [-0.04, -0.015, -0.003])

        # Z: Pole VERTICAL - target state, minimal/no correction needed
        # TIGHT zone (0.012 rad) - 70% narrower than typical designs
        # Key to fast response: any visible tilt activates NS or PS
        self.angle['Z'] = fuzz.trimf(self.angle.universe, [-0.006, 0.0, 0.006])

        # PS: Pole slightly RIGHT - needs gentle rightward push (to catch it)
        # Mirror of NS
        self.angle['PS'] = fuzz.trimf(self.angle.universe, [0.003, 0.015, 0.04])

        # PL: Pole tilted far RIGHT - needs strong rightward push
        # Mirror of NL
        self.angle['PL'] = fuzz.trapmf(self.angle.universe, [0.015, 0.05, 0.5, 0.5])

        # --------------------------------------------------------------------
        # ANGULAR VELOCITY MEMBERSHIP FUNCTIONS (3 terms)
        # --------------------------------------------------------------------
        # Determines HOW the pole is moving - provides "derivative" control.
        # Narrow Z zone (0.16 rad/s) detects motion before angle changes much.
        #
        # Control Logic:
        #   - Pole tilted AND moving same direction -> urgent correction
        #   - Pole tilted BUT moving back -> gentle/no correction (recovering)
        # --------------------------------------------------------------------

        # N: Pole rotating LEFT (counterclockwise)
        # Activates at just -0.02 rad/s (barely visible motion)
        self.angular_velocity['N'] = fuzz.trapmf(self.angular_velocity.universe, [-3.0, -3.0, -0.2, -0.02])

        # Z: Pole nearly STATIONARY
        # Narrow zone (0.16 rad/s total) for quick motion detection
        self.angular_velocity['Z'] = fuzz.trimf(self.angular_velocity.universe, [-0.08, 0.0, 0.08])

        # P: Pole rotating RIGHT (clockwise)
        # Mirror of N
        self.angular_velocity['P'] = fuzz.trapmf(self.angular_velocity.universe, [0.02, 0.2, 3.0, 3.0])

        # --------------------------------------------------------------------
        # CART POSITION MEMBERSHIP FUNCTIONS (3 terms)
        # --------------------------------------------------------------------
        # Secondary objective: keep cart near center of track.
        # EARLY ACTIVATION at 0.05m prevents drift from accumulating.
        # Narrow Z zone (0.30m) encourages staying near center.
        #
        # Critical Design: Position MFs activate early, but RULES produce
        # only PS/NS outputs (not PL/NL) to avoid destabilizing the pole.
        # --------------------------------------------------------------------

        # N: Cart LEFT of center
        # Activates at just 0.05m left - early drift prevention
        self.cart_position['N'] = fuzz.trapmf(self.cart_position.universe, [-2.4, -2.4, -0.3, -0.05])

        # Z: Cart CENTERED - acceptable position
        # Narrow zone (±0.15m) encourages staying near center
        self.cart_position['Z'] = fuzz.trimf(self.cart_position.universe, [-0.15, 0.0, 0.15])

        # P: Cart RIGHT of center
        # Mirror of N
        self.cart_position['P'] = fuzz.trapmf(self.cart_position.universe, [0.05, 0.3, 2.4, 2.4])

        # --------------------------------------------------------------------
        # CART VELOCITY MEMBERSHIP FUNCTIONS (3 terms)
        # --------------------------------------------------------------------
        # Detects cart motion direction to modulate position corrections.
        # Less critical than angular velocity - moderate sensitivity.
        # --------------------------------------------------------------------

        # N: Cart moving LEFT
        self.cart_velocity['N'] = fuzz.trapmf(self.cart_velocity.universe, [-3.0, -3.0, -0.5, -0.1])

        # Z: Cart nearly STATIONARY
        # Wider than angular velocity Z (cart motion less critical)
        self.cart_velocity['Z'] = fuzz.trimf(self.cart_velocity.universe, [-0.3, 0.0, 0.3])

        # P: Cart moving RIGHT
        self.cart_velocity['P'] = fuzz.trapmf(self.cart_velocity.universe, [0.1, 0.5, 3.0, 3.0])

        # --------------------------------------------------------------------
        # FORCE OUTPUT MEMBERSHIP FUNCTIONS (5 terms)
        # --------------------------------------------------------------------
        # Defines control signal magnitudes.
        # STRONGER intermediate forces (PS/NS peak at 2.5N, not 1.5N)
        # for more effective corrections without maximum force.
        #
        # Force Distribution:
        #   NL: -10 to -3 N  (strong left push, emergency)
        #   NS: -5 to 0 N    (moderate left push, routine)
        #   Z:  -0.3 to 0.3 N (minimal force, maintain balance)
        #   PS: 0 to 5 N     (moderate right push, routine)
        #   PL: 3 to 10 N    (strong right push, emergency)
        # --------------------------------------------------------------------

        # NL: Strong LEFT push - emergency correction
        self.force['NL'] = fuzz.trapmf(self.force.universe, [-10, -10, -7, -3])

        # NS: Moderate LEFT push - routine correction
        # Peak at -2.5N (67% stronger than original -1.5N)
        self.force['NS'] = fuzz.trimf(self.force.universe, [-5, -2.5, 0])

        # Z: Minimal force - pole is balanced
        # Narrow zone (0.6N) eliminates "dead zone" where controller does nothing
        self.force['Z'] = fuzz.trimf(self.force.universe, [-0.3, 0, 0.3])

        # PS: Moderate RIGHT push - routine correction
        # Peak at 2.5N
        self.force['PS'] = fuzz.trimf(self.force.universe, [0, 2.5, 5])

        # PL: Strong RIGHT push - emergency correction
        self.force['PL'] = fuzz.trapmf(self.force.universe, [3, 7, 10, 10])

        rules = []

        # ====================================================================
        # FUZZY RULE BASE (25 rules total)
        # ====================================================================
        # Rules organized into 3 categories by priority:
        # 1. Pole Balancing (15 rules) - HIGHEST PRIORITY
        # 2. Position Correction (6 rules) - MEDIUM PRIORITY (weakened)
        # 3. Combined Angle+Position (4 rules) - COORDINATION
        #
        # Rule Interaction: When multiple rules fire, scikit-fuzzy aggregates
        # outputs using MAX for each term, then defuzzifies using centroid.
        # Stronger rules (PL/NL) naturally dominate over weaker ones (Z).
        # ====================================================================

        # ====================================================================
        # CATEGORY 1: POLE BALANCING RULES (15 rules) - HIGHEST PRIORITY
        # ====================================================================
        # 5x3 matrix covering all angle × angular_velocity combinations.
        #
        # Rule Matrix (Angle × AngVel → Force):
        #              AngVel:  N(←)    Z       P(→)
        #           +--------+-------+-------+
        # Angle PL  |   PS   |  PL   |  PL   |  ← tilted right
        #       PS  |   Z    |  PS   |  PL   |  ← slight right
        #       Z   |   NS   |  Z    |  PS   |  ← vertical
        #       NS  |   NL   |  NS   |  Z    |  ← slight left
        #       NL  |   NL   |  NL   |  NS   |  ← tilted left
        #           +--------+-------+-------+
        # ====================================================================

        # --- LARGE RIGHT TILT (PL) ---
        # PL+P→PL: Falling right fast - max force (emergency)
        rules.append(ctrl.Rule(self.angle['PL'] & self.angular_velocity['P'], self.force['PL']))
        # PL+Z→PL: Tilted right, stationary - max force (will fall)
        rules.append(ctrl.Rule(self.angle['PL'] & self.angular_velocity['Z'], self.force['PL']))
        # PL+N→PS: Tilted right, rotating back - gentle assist (recovering)
        rules.append(ctrl.Rule(self.angle['PL'] & self.angular_velocity['N'], self.force['PS']))

        # --- SMALL RIGHT TILT (PS) ---
        # PS+P→PL: Small tilt worsening - strong force (prevent escalation)
        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['P'], self.force['PL']))
        # PS+Z→PS: Small tilt, stable - moderate correction
        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['Z'], self.force['PS']))
        # PS+N→Z: Small tilt, recovering - let it return naturally
        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['N'], self.force['Z']))

        # --- VERTICAL (Z) ---
        # Z+P→PS: Starting to fall right - preemptive correction
        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['P'], self.force['PS']))
        # Z+Z→Z: Perfect balance - no action (target state)
        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['Z'], self.force['Z']))
        # Z+N→NS: Starting to fall left - preemptive correction
        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['N'], self.force['NS']))

        # --- SMALL LEFT TILT (NS) ---
        # NS+P→Z: Small left tilt, rotating back - let it recover
        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['P'], self.force['Z']))
        # NS+Z→NS: Small left tilt, stable - moderate correction
        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['Z'], self.force['NS']))
        # NS+N→NL: Small left tilt worsening - strong correction
        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['N'], self.force['NL']))

        # --- LARGE LEFT TILT (NL) ---
        # NL+P→NS: Tilted left, recovering - gentle assist
        rules.append(ctrl.Rule(self.angle['NL'] & self.angular_velocity['P'], self.force['NS']))
        # NL+Z→NL: Tilted left, stationary - max force
        rules.append(ctrl.Rule(self.angle['NL'] & self.angular_velocity['Z'], self.force['NL']))
        # NL+N→NL: Falling left fast - max force (emergency)
        rules.append(ctrl.Rule(self.angle['NL'] & self.angular_velocity['N'], self.force['NL']))

        # ====================================================================
        # CATEGORY 2: POSITION CORRECTION RULES (6 rules) - MEDIUM PRIORITY
        # ====================================================================
        # Keep cart centered. CRITICAL: Output limited to PS/NS (not PL/NL)
        # to avoid destabilizing the pole.
        #
        # Why Weaker Forces? Original design used PL/NL, causing "tug of war":
        #   Cart right + pole tilting right → balance wants PL, position wants NL
        #   → Forces cancel → weak control → pole falls
        # Solution: Position rules use only PS/NS, letting balance rules win.
        # This improved survival from ~140 to ~400+ steps.
        # ====================================================================

        # --- Cart LEFT of center (N) ---
        # N+N→PS: Drifting more left - gentle right push
        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['N'], self.force['PS']))
        # N+Z→PS: Stationary but left - gentle right push
        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['Z'], self.force['PS']))
        # N+P→Z: Returning right - let momentum carry it
        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['P'], self.force['Z']))

        # --- Cart RIGHT of center (P) ---
        # P+P→NS: Drifting more right - gentle left push
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['P'], self.force['NS']))
        # P+Z→NS: Stationary but right - gentle left push
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['Z'], self.force['NS']))
        # P+N→Z: Returning left - let momentum carry it
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['N'], self.force['Z']))

        # ====================================================================
        # CATEGORY 3: COMBINED ANGLE+POSITION RULES (4 rules) - COORDINATION
        # ====================================================================
        # Handle interaction between pole angle and cart position.
        #
        # Scenario 1: Pole vertical + cart off-center
        #   → Safe to correct position (pole won't be destabilized)
        # Scenario 2: Pole tilted OPPOSITE to cart offset
        #   → Both corrections push same direction → BOOST force (synergy)
        # ====================================================================

        # Pole vertical + cart left → push right (safe position correction)
        rules.append(ctrl.Rule(self.angle['Z'] & self.cart_position['N'], self.force['PS']))
        # Pole vertical + cart right → push left
        rules.append(ctrl.Rule(self.angle['Z'] & self.cart_position['P'], self.force['NS']))

        # SYNERGY: Pole tilting right + cart left → both need right push → PL
        rules.append(ctrl.Rule(self.angle['PS'] & self.cart_position['N'], self.force['PL']))
        # SYNERGY: Pole tilting left + cart right → both need left push → NL
        rules.append(ctrl.Rule(self.angle['NS'] & self.cart_position['P'], self.force['NL']))

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def compute_control(self, angle_val, angular_velocity_val, cart_pos_val, cart_vel_val):
        """
        Compute control force using Mamdani fuzzy inference.

        Performs the complete fuzzy inference cycle:
        1. INPUT CLIPPING: Ensures values are within defined universes
        2. FUZZIFICATION: Converts crisp inputs to membership degrees
        3. RULE EVALUATION: Fires all applicable rules, aggregates outputs
        4. DEFUZZIFICATION: Converts aggregated output to crisp force (centroid)

        Args:
            angle_val: Pole angle in radians (positive = tilted right)
            angular_velocity_val: Pole angular velocity in rad/s (positive = rotating right)
            cart_pos_val: Cart position in meters (positive = right of center)
            cart_vel_val: Cart velocity in m/s (positive = moving right)

        Returns:
            float: Control force in Newtons (positive = push cart right)
        """
        # Clip inputs to universe boundaries to prevent errors
        angle_clipped = np.clip(angle_val, ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX)
        angular_velocity_clipped = np.clip(angular_velocity_val, DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX)
        cart_pos_clipped = np.clip(cart_pos_val, CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX)
        cart_vel_clipped = np.clip(cart_vel_val, CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX)

        # Set crisp input values
        self.simulation.input['angle'] = angle_clipped
        self.simulation.input['angular_velocity'] = angular_velocity_clipped
        self.simulation.input['cart_position'] = cart_pos_clipped
        self.simulation.input['cart_velocity'] = cart_vel_clipped

        # Execute fuzzy inference
        self.simulation.compute()

        return self.simulation.output['force']

    def get_action(self, observation):
        """
        Interface method for Gymnasium environment compatibility.

        Extracts state variables from observation array and computes control.
        This is the main entry point called by the simulation loop.

        Args:
            observation: numpy array [cart_pos, cart_vel, pole_angle, pole_ang_vel]
                        as provided by Gymnasium's CartPole environment

        Returns:
            float: Control force clipped to [-10, 10] Newtons

        Observation Array Format (from Gymnasium):
            Index 0: Cart position (x) in meters
            Index 1: Cart velocity (x_dot) in m/s
            Index 2: Pole angle (theta) in radians
            Index 3: Pole angular velocity (theta_dot) in rad/s
        """
        cart_position = observation[0]
        cart_velocity = observation[1]
        pole_angle = observation[2]
        pole_angular_velocity = observation[3]

        force = self.compute_control(pole_angle, pole_angular_velocity, cart_position, cart_velocity)

        # Safety clipping to action space bounds
        return np.clip(force, -10.0, 10.0)

    def get_membership_functions(self):
        """
        Get membership functions for visualization.

        Returns:
            tuple: (angle, angular_velocity, force, cart_position, cart_velocity)
                   All are scikit-fuzzy Antecedent/Consequent objects
        """
        return self.angle, self.angular_velocity, self.force, self.cart_position, self.cart_velocity
