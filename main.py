"""
Main Entry Point for CartPole Fuzzy Controller Simulation

This module orchestrates the simulation loop, connecting the fuzzy controller
to the CartPole environment and managing episode execution.

USAGE
-----
    python3 main.py          # Run with membership function and control surface plots
    python3 main.py false    # Run without plots (faster startup)

SIMULATION FLOW
---------------
1. Initialize fuzzy controller (loads all 25 rules and membership functions)
2. Display membership function plots (optional, interactive)
3. Display 3D control surface plot (optional, interactive)
4. Run episodes continuously until Ctrl+C
5. Print summary statistics on exit

EPISODE EXECUTION
-----------------
Each episode:
1. Creates fresh CartPole environment
2. Randomizes initial state (cart position, velocity, pole angle, angular velocity)
3. Runs control loop until termination (pole falls or max steps reached)
4. Logs results and continues to next episode

TERMINATION CONDITIONS
----------------------
- Pole angle exceeds ±12 degrees (0.2095 radians)
- Cart position exceeds ±2.4 meters
- Maximum simulation steps reached (default 1000)
- User presses Ctrl+C (graceful shutdown)

OUTPUT
------
Console output includes:
- Episode number and initial conditions
- Periodic state updates (every DISPLAY_INTERVAL steps)
- Episode summary (steps survived, total reward)
- Final summary statistics across all episodes
"""

import sys
import numpy as np
from cartpole_env import CartPoleContinuousEnv
from fuzzy_controller import FuzzyCartPoleController
from visualization import (plot_membership_functions, plot_simulation_results,
                          plot_control_surface, plot_final_summary,
                          plot_episode_comparison)
from cartpole_parameters import (MAX_SIMULATION_STEPS, NUM_EPISODES, DISPLAY_INTERVAL)


def simulate_cartpole_fuzzy(controller, episode_num, render=True):
    """
    Simulate a single episode of CartPole balancing with the fuzzy controller.

    Runs the control loop in real-time, applying fuzzy logic control at each
    time step until the episode terminates.

    Args:
        controller: FuzzyCartPoleController instance (pre-initialized)
        episode_num: Current episode number (for logging)
        render: Whether to render the environment visually
            - True: Opens pygame window showing cart and pole
            - False: Headless mode (faster, for batch runs)

    Returns:
        tuple: (time_steps, angles, angular_velocities, cart_positions,
                actions, rewards, total_reward, steps)
            - time_steps: List of simulation times (seconds)
            - angles: List of pole angles at each step (radians)
            - angular_velocities: List of pole angular velocities (rad/s)
            - cart_positions: List of cart positions (meters)
            - actions: List of forces applied (Newtons)
            - rewards: List of rewards received (1.0 per step)
            - total_reward: Sum of all rewards (equals steps for CartPole)
            - steps: Total steps survived before termination

    Episode Flow:
        1. Create environment with optional rendering
        2. Reset to random initial state
        3. Run control loop:
           a. Get observation from environment
           b. Compute control action via fuzzy inference
           c. Apply action and get next state
           d. Log data and check termination
        4. Close environment and return results
    """
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    # Create environment with optional visual rendering
    render_mode = "human" if render else None
    env = CartPoleContinuousEnv(render_mode=render_mode)

    # Get initial observation from environment reset
    observation, info = env.reset()

    # =========================================================================
    # RANDOM INITIAL CONDITIONS
    # =========================================================================
    # Randomize starting state to test controller robustness.
    # These ranges are conservative - controller should handle them easily.
    #
    # Cart position: ±0.5m (well within ±2.4m boundary)
    # Cart velocity: ±0.2 m/s (slow initial movement)
    # Pole angle: ±0.05 rad = ±2.9° (slightly tilted, not challenging)
    # Pole velocity: ±0.2 rad/s (slow initial rotation)
    random_cart_pos = np.random.uniform(-0.5, 0.5)
    random_cart_vel = np.random.uniform(-0.2, 0.2)
    random_pole_angle = np.random.uniform(-0.05, 0.05)
    random_pole_vel = np.random.uniform(-0.2, 0.2)

    # Directly set environment state (bypasses normal reset)
    env.state = (random_cart_pos, random_cart_vel, random_pole_angle, random_pole_vel)
    observation = np.array(env.state, dtype=np.float32)

    # =========================================================================
    # DATA COLLECTION ARRAYS
    # =========================================================================
    # Store simulation history for analysis and visualization.
    # Pre-populated with initial values (step 0).
    time_steps = [0.0]                        # Simulation time in seconds
    angles = [observation[2]]                  # Pole angle (radians)
    angular_velocities = [observation[3]]      # Pole angular velocity (rad/s)
    cart_positions = [observation[0]]          # Cart position (meters)
    actions = [0]                              # Control force applied (Newtons)
    rewards = [0.0]                            # Reward received (1.0 per step)

    total_reward = 0
    step = 0

    # =========================================================================
    # EPISODE HEADER
    # =========================================================================
    print(f"\nStarting CartPole simulation (Episode {episode_num}):")
    print(f"Controller: Fuzzy Logic")
    print(f"Initial pole angle: {np.degrees(observation[2]):.2f} degrees")
    print(f"Initial angular velocity: {observation[3]:.2f} rad/s")
    print(f"Max steps: {MAX_SIMULATION_STEPS}\n")

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    # Run until termination (pole falls, cart out of bounds, or max steps)
    while step < MAX_SIMULATION_STEPS:
        # ---------------------------------------------------------------------
        # FUZZY CONTROL: Compute force from current state
        # ---------------------------------------------------------------------
        # Controller performs:
        # 1. Fuzzification of all 4 state variables
        # 2. Rule evaluation (25 rules)
        # 3. Defuzzification to get crisp force output
        action = controller.get_action(observation)

        # ---------------------------------------------------------------------
        # ENVIRONMENT STEP: Apply force and get next state
        # ---------------------------------------------------------------------
        observation, reward, terminated, truncated, info = env.step(action)

        # Update counters
        step += 1
        total_reward += reward

        # ---------------------------------------------------------------------
        # DATA LOGGING
        # ---------------------------------------------------------------------
        time_steps.append(step * 0.02)  # Convert step to time (20ms per step)
        angles.append(observation[2])
        angular_velocities.append(observation[3])
        cart_positions.append(observation[0])
        actions.append(action)
        rewards.append(reward)

        # Periodic status output
        if step % DISPLAY_INTERVAL == 0:
            print(f"Step {step}: Angle={np.degrees(observation[2]):.2f}deg, "
                  f"AngVel={observation[3]:.2f}rad/s, "
                  f"CartPos={observation[0]:.2f}m, Force={action:.2f}")

        # ---------------------------------------------------------------------
        # TERMINATION CHECK
        # ---------------------------------------------------------------------
        # terminated: Pole fell (angle > 12°) or cart out of bounds (|x| > 2.4m)
        # truncated: Reached maximum steps (success!)
        if terminated or truncated:
            if terminated:
                print(f"\nPole fell at step {step}!")
            else:
                print(f"\nReached maximum steps ({step})!")
            break

    # =========================================================================
    # EPISODE SUMMARY
    # =========================================================================
    print(f"Total reward: {total_reward}")
    print(f"Final pole angle: {np.degrees(observation[2]):.2f} degrees")
    print(f"Final cart position: {observation[0]:.2f} m")

    # Clean up environment resources
    env.close()

    return time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, step


def main():
    """
    Main entry point - runs continuous CartPole simulation with fuzzy controller.

    Initializes the fuzzy controller, optionally displays visualization plots,
    then runs episodes indefinitely until the user interrupts with Ctrl+C.

    Command Line Arguments:
        sys.argv[1]: 'false' to skip initial plots (default: show plots)

    Execution Flow:
        1. Parse command line arguments
        2. Create FuzzyCartPoleController (loads rules and membership functions)
        3. Display membership function plot (blocks until closed)
        4. Display control surface plot (blocks until closed)
        5. Enter episode loop (runs until Ctrl+C)
        6. Print summary statistics

    Graceful Shutdown:
        Catches KeyboardInterrupt (Ctrl+C) and prints summary statistics
        for all completed episodes before exiting.
    """
    # =========================================================================
    # COMMAND LINE ARGUMENT PARSING
    # =========================================================================
    # Optional argument to skip plots for faster startup
    # Usage: python3 main.py false
    show_plots = True

    if len(sys.argv) > 1:
        show_plots = sys.argv[1].lower() != 'false'

    # =========================================================================
    # STARTUP BANNER
    # =========================================================================
    print("=" * 60)
    print("CartPole Fuzzy Logic Controller")
    print("Real-time Simulation")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # =========================================================================
    # CONTROLLER INITIALIZATION
    # =========================================================================
    # Creates the fuzzy inference system:
    # - 4 input variables (angle, angular_velocity, cart_position, cart_velocity)
    # - 1 output variable (force)
    # - 25 fuzzy rules
    # - Centroid defuzzification
    controller = FuzzyCartPoleController()

    # =========================================================================
    # OPTIONAL VISUALIZATION
    # =========================================================================
    # Display plots if enabled (each blocks until user closes the window)
    if show_plots:
        print("\nDisplaying membership functions...")
        plot_membership_functions(controller)

        print("\nDisplaying control surface...")
        plot_control_surface(controller)

    # =========================================================================
    # EPISODE STATISTICS TRACKING
    # =========================================================================
    # Accumulate results across all episodes for summary
    episode_rewards = []  # Total reward per episode
    episode_steps = []    # Steps survived per episode
    episode = 0

    # =========================================================================
    # MAIN EPISODE LOOP
    # =========================================================================
    # Runs indefinitely until user presses Ctrl+C
    try:
        while True:
            episode += 1
            print(f"\n{'='*60}")
            print(f"Episode {episode}")
            print("=" * 60)

            # Run single episode and collect results
            time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, steps = simulate_cartpole_fuzzy(
                controller, episode, render=True
            )

            # Store episode statistics
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            print(f"\nEpisode {episode} finished - Steps: {steps}, Reward: {total_reward}")

    except KeyboardInterrupt:
        # =====================================================================
        # GRACEFUL SHUTDOWN
        # =====================================================================
        # User pressed Ctrl+C - print summary and exit cleanly
        print("\n\nSimulation stopped by user.")

    # =========================================================================
    # FINAL SUMMARY STATISTICS
    # =========================================================================
    # Print aggregate statistics across all completed episodes
    if len(episode_steps) > 0:
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        print(f"Total episodes: {len(episode_steps)}")
        print(f"Average steps: {np.mean(episode_steps):.1f}")
        print(f"Average reward: {np.mean(episode_rewards):.1f}")
        print(f"Best episode: {np.argmax(episode_steps) + 1} with {np.max(episode_steps)} steps")
        print("=" * 60)


if __name__ == "__main__":
    main()
