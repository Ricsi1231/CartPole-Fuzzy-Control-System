"""
Custom CartPole Environment with Continuous Action Space

This module extends Gymnasium's CartPoleEnv to accept continuous force values
instead of discrete left/right actions. This enables smooth force control
from the fuzzy logic controller.

KEY MODIFICATION
----------------
- Original CartPole: Discrete actions {0: push left, 1: push right} with fixed 10N force
- This version: Continuous force in range [-10, +10] Newtons

HOW IT WORKS
------------
The step() method intercepts the continuous force value, extracts the magnitude
and direction, then calls the parent CartPole physics with the appropriate
force_mag setting.

FORCE DIRECTION CONVENTION
--------------------------
- Positive force (+) = Push cart RIGHT (action=1 in parent)
- Negative force (-) = Push cart LEFT  (action=0 in parent)

PHYSICS DELEGATION
------------------
The parent CartPoleEnv.step() uses self.force_mag for the force magnitude.
By setting this attribute before calling super().step(), we achieve
variable-magnitude continuous control while reusing the parent's physics.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CartPoleContinuousEnv(CartPoleEnv):
    """
    CartPole environment modified for continuous force control.

    Extends Gymnasium's CartPoleEnv to accept continuous force values in the
    range [-10, +10] Newtons instead of discrete {0, 1} actions.

    This enables fuzzy logic controllers and other continuous control methods
    to apply precisely calculated forces rather than binary push left/right.

    Attributes:
        action_space: Box(-10, 10) instead of Discrete(2)
        force_mag: Dynamically set each step based on input force magnitude

    Inherits all other attributes from CartPoleEnv:
        - gravity: 9.8 m/s^2
        - masscart: 1.0 kg
        - masspole: 0.1 kg
        - total_mass: 1.1 kg
        - length: 0.5 m (half pole length)
        - polemass_length: 0.05 kg*m
        - tau: 0.02 s (time step)
        - theta_threshold_radians: 12 degrees (failure angle)
        - x_threshold: 2.4 m (failure position)
    """

    def __init__(self, render_mode=None):
        """
        Initialize the continuous CartPole environment.

        Calls parent constructor, then replaces the discrete action space
        with a continuous Box space.

        Args:
            render_mode: Rendering mode for visualization
                - None: No rendering (fastest)
                - "human": Real-time window display
                - "rgb_array": Returns RGB array for recording

        Action Space Change:
            Original: Discrete(2) -> {0: left, 1: right}
            Modified: Box(low=-10.0, high=10.0, shape=(1,), dtype=float32)
        """
        super().__init__(render_mode=render_mode)
        # Replace discrete action space with continuous force range
        # Force range matches typical CartPole maximum force magnitude
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        """
        Execute one simulation step with continuous force input.

        Converts the continuous force value to the format expected by the
        parent CartPole physics engine, then delegates the actual simulation.

        Args:
            action: Force to apply to the cart
                - Can be numpy array (shape (1,)) or scalar
                - Range: [-10.0, +10.0] Newtons
                - Positive = push right, Negative = push left

        Returns:
            observation: numpy array [cart_pos, cart_vel, pole_angle, pole_ang_vel]
            reward: 1.0 for each step the pole remains upright
            terminated: True if pole fell (angle > 12°) or cart out of bounds (|x| > 2.4m)
            truncated: True if episode reached maximum steps
            info: Empty dict (for Gymnasium compatibility)

        Implementation Details:
            1. Extract scalar force from array if needed
            2. Clip force to valid range [-10, 10]
            3. Determine direction: positive -> right (1), negative -> left (0)
            4. Set force_mag attribute for parent physics
            5. Call parent step() with direction
        """
        # Handle both array input (from action space) and scalar input
        if isinstance(action, np.ndarray):
            force = float(action[0])
        else:
            force = float(action)

        # Safety clip to action space bounds
        force = np.clip(force, -10.0, 10.0)

        # Convert continuous force to direction for parent class
        # Parent uses: force = self.force_mag if action == 1 else -self.force_mag
        # So action=1 means push right (positive), action=0 means push left (negative)
        direction = 1 if force >= 0 else 0

        # Set force magnitude for parent's physics calculation
        # Parent will apply: force_mag in direction specified by action
        self.force_mag = abs(force)

        # Delegate to parent for actual physics simulation
        return super().step(direction)
