"""
Main Entry Point for CartPole Fuzzy Controller Simulation

This module provides the main simulation loop for running the fuzzy logic
controller on the CartPole environment. It handles episode management,
data collection, and result visualization.
"""

import sys
import numpy as np
from cartpole_env import CartPoleContinuousEnv
from fuzzy_controller import FuzzyCartPoleController
from visualization import (plot_membership_functions, plot_simulation_results,
                          plot_control_surface, plot_final_summary,
                          plot_episode_comparison)
from cartpole_parameters import (MAX_SIMULATION_STEPS, NUM_EPISODES, DISPLAY_INTERVAL,
                                  INIT_CART_POS_MIN, INIT_CART_POS_MAX,
                                  INIT_CART_VEL_MIN, INIT_CART_VEL_MAX,
                                  INIT_POLE_ANGLE_MIN, INIT_POLE_ANGLE_MAX,
                                  INIT_POLE_VEL_MIN, INIT_POLE_VEL_MAX,
                                  OBS_CART_POS, OBS_CART_VEL, OBS_POLE_ANGLE, OBS_POLE_VEL,
                                  DT, SEPARATOR_WIDTH)


def simulate_cartpole_fuzzy(controller, episode_num, render=True):
    """
    Simulate a single episode of CartPole balancing with the fuzzy controller.

    Args:
        controller: FuzzyCartPoleController instance
        episode_num: Current episode number
        render: Whether to render the environment visually

    Returns:
        tuple: (time_steps, angles, angular_velocities, cart_positions,
                actions, rewards, total_reward, steps)
    """
    # =========================================================================
    # Environment Setup
    # Create the CartPole environment with optional rendering
    # =========================================================================
    render_mode = "human" if render else None
    env = CartPoleContinuousEnv(render_mode=render_mode)
    observation, info = env.reset()

    # Reset the integral controller for the new episode
    controller.reset_integral()

    # =========================================================================
    # Random Initial State
    # Randomize starting conditions to test controller robustness
    # =========================================================================
    random_cart_pos = np.random.uniform(INIT_CART_POS_MIN, INIT_CART_POS_MAX)
    random_cart_vel = np.random.uniform(INIT_CART_VEL_MIN, INIT_CART_VEL_MAX)
    random_pole_angle = np.random.uniform(INIT_POLE_ANGLE_MIN, INIT_POLE_ANGLE_MAX)
    random_pole_vel = np.random.uniform(INIT_POLE_VEL_MIN, INIT_POLE_VEL_MAX)

    # Apply randomized initial state to the environment
    env.state = (random_cart_pos, random_cart_vel, random_pole_angle, random_pole_vel)
    observation = np.array(env.state, dtype=np.float32)

    # =========================================================================
    # Data Collection Arrays
    # Store simulation data for analysis and visualization
    # =========================================================================
    time_steps = [0.0]
    angles = [observation[OBS_POLE_ANGLE]]
    angular_velocities = [observation[OBS_POLE_VEL]]
    cart_positions = [observation[OBS_CART_POS]]
    actions = [0]
    rewards = [0.0]

    total_reward = 0
    step = 0

    # =========================================================================
    # Episode Info Display
    # =========================================================================
    print(f"\nStarting CartPole simulation (Episode {episode_num}):")
    print(f"Controller: Fuzzy Logic")
    print(f"Initial pole angle: {np.degrees(observation[OBS_POLE_ANGLE]):.2f} degrees")
    print(f"Initial angular velocity: {observation[OBS_POLE_VEL]:.2f} rad/s")
    print(f"Max steps: {MAX_SIMULATION_STEPS}\n")

    # =========================================================================
    # Main Simulation Loop
    # Run until pole falls, cart goes out of bounds, or max steps reached
    # =========================================================================
    while step < MAX_SIMULATION_STEPS:
        # Get control action from fuzzy controller
        action = controller.get_action(observation)

        # Apply action to environment and get new state
        observation, reward, terminated, truncated, info = env.step(action)

        # Update counters and accumulate reward
        step += 1
        total_reward += reward

        # Record data for analysis
        time_steps.append(step * DT)
        angles.append(observation[OBS_POLE_ANGLE])
        angular_velocities.append(observation[OBS_POLE_VEL])
        cart_positions.append(observation[OBS_CART_POS])
        actions.append(action)
        rewards.append(reward)

        # Periodic status display
        if step % DISPLAY_INTERVAL == 0:
            print(f"Step {step}: Angle={np.degrees(observation[OBS_POLE_ANGLE]):.2f}deg, "
                  f"AngVel={observation[OBS_POLE_VEL]:.2f}rad/s, "
                  f"CartPos={observation[OBS_CART_POS]:.2f}m, Force={action:.2f}")

        # Check for episode termination
        if terminated or truncated:
            if terminated:
                print(f"\nPole fell at step {step}!")
            else:
                print(f"\nReached maximum steps ({step})!")
            break

    # =========================================================================
    # Episode Summary
    # =========================================================================
    print(f"Total reward: {total_reward}")
    print(f"Final pole angle: {np.degrees(observation[OBS_POLE_ANGLE]):.2f} degrees")
    print(f"Final cart position: {observation[OBS_CART_POS]:.2f} m")

    env.close()

    return time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, step


def main():
    """
    Main entry point - runs continuous CartPole simulation with fuzzy controller.

    Usage: python main.py [show_plots]
        show_plots: 'true' or 'false' (default: true)
    """
    # =========================================================================
    # Parse Command Line Arguments
    # =========================================================================
    show_plots = True

    if len(sys.argv) > 1:
        show_plots = sys.argv[1].lower() != 'false'

    # =========================================================================
    # Welcome Banner
    # =========================================================================
    print("=" * SEPARATOR_WIDTH)
    print("CartPole Fuzzy Logic Controller")
    print("Real-time Simulation")
    print("Press Ctrl+C to stop")
    print("=" * SEPARATOR_WIDTH)

    # =========================================================================
    # Initialize Controller
    # =========================================================================
    controller = FuzzyCartPoleController()

    # =========================================================================
    # Display Membership Functions and Control Surface (Optional)
    # =========================================================================
    if show_plots:
        print("\nDisplaying membership functions...")
        plot_membership_functions(controller)

        print("\nDisplaying control surface...")
        plot_control_surface(controller)

    # =========================================================================
    # Episode Tracking
    # =========================================================================
    episode_rewards = []
    episode_steps = []
    episode = 0

    # =========================================================================
    # Main Episode Loop
    # Runs continuously until user interrupts with Ctrl+C
    # =========================================================================
    try:
        while True:
            episode += 1
            print(f"\n{'=' * SEPARATOR_WIDTH}")
            print(f"Episode {episode}")
            print("=" * SEPARATOR_WIDTH)

            # Run single episode and collect results
            time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, steps = simulate_cartpole_fuzzy(
                controller, episode, render=True
            )

            # Store episode results for summary statistics
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            print(f"\nEpisode {episode} finished - Steps: {steps}, Reward: {total_reward}")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    # =========================================================================
    # Final Summary Statistics
    # Display aggregate performance metrics across all episodes
    # =========================================================================
    if len(episode_steps) > 0:
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("Simulation Summary")
        print("=" * SEPARATOR_WIDTH)
        print(f"Total episodes: {len(episode_steps)}")
        print(f"Average steps: {np.mean(episode_steps):.1f}")
        print(f"Average reward: {np.mean(episode_rewards):.1f}")
        print(f"Best episode: {np.argmax(episode_steps) + 1} with {np.max(episode_steps)} steps")
        print("=" * SEPARATOR_WIDTH)


if __name__ == "__main__":
    main()
