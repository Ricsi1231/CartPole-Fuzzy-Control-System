"""
Main Entry Point for CartPole Fuzzy Controller Simulation
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
                                  INIT_POLE_VEL_MIN, INIT_POLE_VEL_MAX)


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
    render_mode = "human" if render else None
    env = CartPoleContinuousEnv(render_mode=render_mode)
    observation, info = env.reset()

    controller.reset_integral()

    random_cart_pos = np.random.uniform(INIT_CART_POS_MIN, INIT_CART_POS_MAX)
    random_cart_vel = np.random.uniform(INIT_CART_VEL_MIN, INIT_CART_VEL_MAX)
    random_pole_angle = np.random.uniform(INIT_POLE_ANGLE_MIN, INIT_POLE_ANGLE_MAX)
    random_pole_vel = np.random.uniform(INIT_POLE_VEL_MIN, INIT_POLE_VEL_MAX)

    env.state = (random_cart_pos, random_cart_vel, random_pole_angle, random_pole_vel)
    observation = np.array(env.state, dtype=np.float32)

    time_steps = [0.0]
    angles = [observation[2]]
    angular_velocities = [observation[3]]
    cart_positions = [observation[0]]
    actions = [0]
    rewards = [0.0]

    total_reward = 0
    step = 0

    print(f"\nStarting CartPole simulation (Episode {episode_num}):")
    print(f"Controller: Fuzzy Logic")
    print(f"Initial pole angle: {np.degrees(observation[2]):.2f} degrees")
    print(f"Initial angular velocity: {observation[3]:.2f} rad/s")
    print(f"Max steps: {MAX_SIMULATION_STEPS}\n")

    while step < MAX_SIMULATION_STEPS:
        action = controller.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        step += 1
        total_reward += reward

        time_steps.append(step * 0.02)
        angles.append(observation[2])
        angular_velocities.append(observation[3])
        cart_positions.append(observation[0])
        actions.append(action)
        rewards.append(reward)

        if step % DISPLAY_INTERVAL == 0:
            print(f"Step {step}: Angle={np.degrees(observation[2]):.2f}deg, "
                  f"AngVel={observation[3]:.2f}rad/s, "
                  f"CartPos={observation[0]:.2f}m, Force={action:.2f}")

        if terminated or truncated:
            if terminated:
                print(f"\nPole fell at step {step}!")
            else:
                print(f"\nReached maximum steps ({step})!")
            break

    print(f"Total reward: {total_reward}")
    print(f"Final pole angle: {np.degrees(observation[2]):.2f} degrees")
    print(f"Final cart position: {observation[0]:.2f} m")

    env.close()

    return time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, step


def main():
    """
    Main entry point - runs continuous CartPole simulation with fuzzy controller.
    """
    show_plots = True

    if len(sys.argv) > 1:
        show_plots = sys.argv[1].lower() != 'false'

    print("=" * 60)
    print("CartPole Fuzzy Logic Controller")
    print("Real-time Simulation")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    controller = FuzzyCartPoleController()

    if show_plots:
        print("\nDisplaying membership functions...")
        # plot_membership_functions(controller)

        print("\nDisplaying control surface...")
        # plot_control_surface(controller)

    episode_rewards = []
    episode_steps = []
    episode = 0

    try:
        while True:
            episode += 1
            print(f"\n{'='*60}")
            print(f"Episode {episode}")
            print("=" * 60)

            time_steps, angles, angular_velocities, cart_positions, actions, rewards, total_reward, steps = simulate_cartpole_fuzzy(
                controller, episode, render=True
            )

            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            print(f"\nEpisode {episode} finished - Steps: {steps}, Reward: {total_reward}")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

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
