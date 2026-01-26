"""
Custom CartPole Environment with Continuous Action Space

This module extends Gymnasium's CartPoleEnv to accept continuous force values
instead of discrete left/right actions.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class CartPoleContinuousEnv(CartPoleEnv):
    """
    CartPole environment modified for continuous force control.

    Extends Gymnasium's CartPoleEnv to accept continuous force values in the
    range [-10, +10] Newtons instead of discrete {0, 1} actions.
    """

    def __init__(self, render_mode=None):
        """
        Initialize the continuous CartPole environment.

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__(render_mode=render_mode)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        """
        Execute one simulation step with continuous force input.

        Args:
            action: Force to apply to the cart [-10.0, +10.0] Newtons

        Returns:
            observation, reward, terminated, truncated, info
        """
        if isinstance(action, np.ndarray):
            force = float(action[0])
        else:
            force = float(action)

        force = np.clip(force, -10.0, 10.0)
        direction = 1 if force >= 0 else 0
        self.force_mag = abs(force)

        return super().step(direction)
