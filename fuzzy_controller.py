"""
Fuzzy Logic Controller for CartPole Balancing Problem

This module implements a Mamdani-type fuzzy inference system that uses all 4 state
variables from the CartPole environment to compute a continuous force output.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from cartpole_parameters import (ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX, ANGLE_STEP,
                                  DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX, DELTA_ANGLE_STEP,
                                  CONTROL_RANGE_MIN, CONTROL_RANGE_MAX, CONTROL_STEP,
                                  CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX, CART_POSITION_STEP,
                                  CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX, CART_VELOCITY_STEP,
                                  ANGLE_NL, ANGLE_NS, ANGLE_Z, ANGLE_PS, ANGLE_PL,
                                  ANG_VEL_N, ANG_VEL_Z, ANG_VEL_P,
                                  CART_POS_N, CART_POS_Z, CART_POS_P,
                                  CART_VEL_N, CART_VEL_Z, CART_VEL_P,
                                  FORCE_NL, FORCE_NS, FORCE_Z, FORCE_PS, FORCE_PL,
                                  INTEGRAL_GAIN, INTEGRAL_LIMIT, INTEGRAL_ANGLE_THRESHOLD, INTEGRAL_DECAY,
                                  DT, OBS_CART_POS, OBS_CART_VEL, OBS_POLE_ANGLE, OBS_POLE_VEL)


class FuzzyCartPoleController:
    """
    Fuzzy logic controller for CartPole balancing.

    Uses all 4 state variables as inputs:
    - x (cart position)
    - x_dot (cart velocity)
    - theta (pole angle)
    - theta_dot (pole angular velocity)

    Includes integral control on position to eliminate steady-state drift.
    """

    def __init__(self):
        self.position_integral = 0.0
        self.integral_gain = INTEGRAL_GAIN
        self.integral_limit = INTEGRAL_LIMIT
        self.angle = ctrl.Antecedent(np.arange(ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX, ANGLE_STEP), 'angle')
        self.angular_velocity = ctrl.Antecedent(np.arange(DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX, DELTA_ANGLE_STEP), 'angular_velocity')
        self.cart_position = ctrl.Antecedent(np.arange(CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX, CART_POSITION_STEP), 'cart_position')
        self.cart_velocity = ctrl.Antecedent(np.arange(CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX, CART_VELOCITY_STEP), 'cart_velocity')
        self.force = ctrl.Consequent(np.arange(CONTROL_RANGE_MIN, CONTROL_RANGE_MAX, CONTROL_STEP), 'force')

        self.angle['NL'] = fuzz.trapmf(self.angle.universe, ANGLE_NL)
        self.angle['NS'] = fuzz.trimf(self.angle.universe, ANGLE_NS)
        self.angle['Z'] = fuzz.trimf(self.angle.universe, ANGLE_Z)
        self.angle['PS'] = fuzz.trimf(self.angle.universe, ANGLE_PS)
        self.angle['PL'] = fuzz.trapmf(self.angle.universe, ANGLE_PL)

        self.angular_velocity['N'] = fuzz.trapmf(self.angular_velocity.universe, ANG_VEL_N)
        self.angular_velocity['Z'] = fuzz.trimf(self.angular_velocity.universe, ANG_VEL_Z)
        self.angular_velocity['P'] = fuzz.trapmf(self.angular_velocity.universe, ANG_VEL_P)

        self.cart_position['N'] = fuzz.trapmf(self.cart_position.universe, CART_POS_N)
        self.cart_position['Z'] = fuzz.trimf(self.cart_position.universe, CART_POS_Z)
        self.cart_position['P'] = fuzz.trapmf(self.cart_position.universe, CART_POS_P)

        self.cart_velocity['N'] = fuzz.trapmf(self.cart_velocity.universe, CART_VEL_N)
        self.cart_velocity['Z'] = fuzz.trimf(self.cart_velocity.universe, CART_VEL_Z)
        self.cart_velocity['P'] = fuzz.trapmf(self.cart_velocity.universe, CART_VEL_P)

        self.force['NL'] = fuzz.trapmf(self.force.universe, FORCE_NL)
        self.force['NS'] = fuzz.trimf(self.force.universe, FORCE_NS)
        self.force['Z'] = fuzz.trimf(self.force.universe, FORCE_Z)
        self.force['PS'] = fuzz.trimf(self.force.universe, FORCE_PS)
        self.force['PL'] = fuzz.trapmf(self.force.universe, FORCE_PL)

        rules = []

        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['N'], self.force['PL']))
        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['Z'], self.force['PL']))
        rules.append(ctrl.Rule(self.cart_position['N'] & self.cart_velocity['P'], self.force['PS']))
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['P'], self.force['NL']))
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['Z'], self.force['NL']))
        rules.append(ctrl.Rule(self.cart_position['P'] & self.cart_velocity['N'], self.force['NS']))

        rules.append(ctrl.Rule(self.angle['PL'] & (self.angular_velocity['P'] | self.angular_velocity['Z']), self.force['PL']))
        rules.append(ctrl.Rule(self.angle['PL'] & self.angular_velocity['N'], self.force['PS']))

        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['P'], self.force['PL']))
        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['Z'], self.force['PS']))
        rules.append(ctrl.Rule(self.angle['PS'] & self.angular_velocity['N'], self.force['Z']))

        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['P'], self.force['PS']))
        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['Z'], self.force['Z']))
        rules.append(ctrl.Rule(self.angle['Z'] & self.angular_velocity['N'], self.force['NS']))

        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['P'], self.force['Z']))
        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['Z'], self.force['NS']))
        rules.append(ctrl.Rule(self.angle['NS'] & self.angular_velocity['N'], self.force['NL']))

        rules.append(ctrl.Rule(self.angle['NL'] & self.angular_velocity['P'], self.force['NS']))
        rules.append(ctrl.Rule(self.angle['NL'] & (self.angular_velocity['Z'] | self.angular_velocity['N']), self.force['NL']))

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def compute_control(self, angle_val, angular_velocity_val, cart_pos_val, cart_vel_val):
        """
        Compute control force using Mamdani fuzzy inference.

        Args:
            angle_val: Pole angle in radians (positive = tilted right)
            angular_velocity_val: Pole angular velocity in rad/s (positive = rotating right)
            cart_pos_val: Cart position in meters (positive = right of center)
            cart_vel_val: Cart velocity in m/s (positive = moving right)

        Returns:
            float: Control force in Newtons (positive = push cart right)
        """
        angle_clipped = np.clip(angle_val, ANGLE_ERROR_RANGE_MIN, ANGLE_ERROR_RANGE_MAX)
        angular_velocity_clipped = np.clip(angular_velocity_val, DELTA_ANGLE_RANGE_MIN, DELTA_ANGLE_RANGE_MAX)
        cart_pos_clipped = np.clip(cart_pos_val, CART_POSITION_RANGE_MIN, CART_POSITION_RANGE_MAX)
        cart_vel_clipped = np.clip(cart_vel_val, CART_VELOCITY_RANGE_MIN, CART_VELOCITY_RANGE_MAX)

        self.simulation.input['angle'] = angle_clipped
        self.simulation.input['angular_velocity'] = angular_velocity_clipped
        self.simulation.input['cart_position'] = cart_pos_clipped
        self.simulation.input['cart_velocity'] = cart_vel_clipped

        self.simulation.compute()

        return self.simulation.output['force']

    def reset_integral(self):
        """Reset the position integral for a new episode."""
        self.position_integral = 0.0

    def get_action(self, observation):
        """
        Interface method for Gymnasium environment compatibility.

        Args:
            observation: numpy array [cart_pos, cart_vel, pole_angle, pole_ang_vel]

        Returns:
            float: Control force clipped to [-10, 10] Newtons
        """
        cart_position = observation[OBS_CART_POS]
        cart_velocity = observation[OBS_CART_VEL]
        pole_angle = observation[OBS_POLE_ANGLE]
        pole_angular_velocity = observation[OBS_POLE_VEL]

        fuzzy_force = self.compute_control(pole_angle, pole_angular_velocity, cart_position, cart_velocity)

        angle_magnitude = abs(pole_angle)
        if angle_magnitude < INTEGRAL_ANGLE_THRESHOLD:
            self.position_integral += cart_position * DT
            self.position_integral = np.clip(self.position_integral, -self.integral_limit, self.integral_limit)
            integral_force = -self.integral_gain * self.position_integral
        else:
            integral_force = 0.0
            self.position_integral *= INTEGRAL_DECAY

        total_force = fuzzy_force + integral_force
        clipped_force = np.clip(total_force, CONTROL_RANGE_MIN, CONTROL_RANGE_MAX)

        """print(f"[DEBUG] pos={cart_position:+.3f} vel={cart_velocity:+.3f} "
              f"angle={np.degrees(pole_angle):+.2f}° ang_vel={pole_angular_velocity:+.3f} "
              f"| fuzzy={fuzzy_force:+.3f} int={integral_force:+.3f} out={clipped_force:+.3f}")
        """
              
        return clipped_force

    def get_membership_functions(self):
        """
        Get membership functions for visualization.

        Returns:
            tuple: (angle, angular_velocity, force, cart_position, cart_velocity)
        """
        return self.angle, self.angular_velocity, self.force, self.cart_position, self.cart_velocity
