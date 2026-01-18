"""
Visualization Functions for Fuzzy Controller Analysis
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skfuzzy import control as ctrl


def plot_membership_functions(controller):
    """
    Plot membership functions for all 5 fuzzy variables.

    Args:
        controller: FuzzyCartPoleController instance
    """
    angle, angular_velocity, force, cart_position, cart_velocity = controller.get_membership_functions()

    colors = {'NL': '#8B0000', 'NS': '#E74C3C', 'Z': '#27AE60', 'PS': '#3498DB', 'PL': '#00008B', 'N': '#8B0000', 'P': '#00008B'}
    labels = {'NL': 'Neg Large', 'NS': 'Neg Small', 'Z': 'Zero', 'PS': 'Pos Small', 'PL': 'Pos Large', 'N': 'Negative', 'P': 'Positive'}

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    ax1 = axes[0, 0]
    for term_name in ['NL', 'NS', 'Z', 'PS', 'PL']:
        if term_name in angle.terms:
            mf = angle[term_name].mf
            ax1.fill_between(angle.universe, mf, alpha=0.3, color=colors[term_name])
            ax1.plot(angle.universe, mf, linewidth=2, color=colors[term_name], label=labels[term_name])
    ax1.set_xlabel('Pole Angle (radians)', fontsize=10)
    ax1.set_ylabel('Membership', fontsize=10)
    ax1.set_title('Pole Angle', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlim(-0.15, 0.15)  # Zoom to show membership function detail
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for term_name in ['NL', 'NS', 'N', 'Z', 'PS', 'PL', 'P']:
        if term_name in angular_velocity.terms:
            mf = angular_velocity[term_name].mf
            ax2.fill_between(angular_velocity.universe, mf, alpha=0.3, color=colors[term_name])
            ax2.plot(angular_velocity.universe, mf, linewidth=2, color=colors[term_name], label=labels[term_name])
    ax2.set_xlabel('Angular Velocity (rad/s)', fontsize=10)
    ax2.set_ylabel('Membership', fontsize=10)
    ax2.set_title('Angular Velocity', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(angular_velocity.universe.min(), angular_velocity.universe.max())
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for term_name in ['NL', 'NS', 'N', 'Z', 'PS', 'PL', 'P']:
        if term_name in cart_position.terms:
            mf = cart_position[term_name].mf
            ax3.fill_between(cart_position.universe, mf, alpha=0.3, color=colors[term_name])
            ax3.plot(cart_position.universe, mf, linewidth=2, color=colors[term_name], label=labels[term_name])
    ax3.set_xlabel('Cart Position (m)', fontsize=10)
    ax3.set_ylabel('Membership', fontsize=10)
    ax3.set_title('Cart Position', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_xlim(cart_position.universe.min(), cart_position.universe.max())
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for term_name in ['NL', 'NS', 'N', 'Z', 'PS', 'PL', 'P']:
        if term_name in cart_velocity.terms:
            mf = cart_velocity[term_name].mf
            ax4.fill_between(cart_velocity.universe, mf, alpha=0.3, color=colors[term_name])
            ax4.plot(cart_velocity.universe, mf, linewidth=2, color=colors[term_name], label=labels[term_name])
    ax4.set_xlabel('Cart Velocity (m/s)', fontsize=10)
    ax4.set_ylabel('Membership', fontsize=10)
    ax4.set_title('Cart Velocity', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_ylim(-0.05, 1.1)
    ax4.set_xlim(cart_velocity.universe.min(), cart_velocity.universe.max())
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[2, 0]
    for term_name in ['NL', 'NS', 'Z', 'PS', 'PL']:
        if term_name in force.terms:
            mf = force[term_name].mf
            ax5.fill_between(force.universe, mf, alpha=0.3, color=colors[term_name])
            ax5.plot(force.universe, mf, linewidth=2, color=colors[term_name], label=labels[term_name])
    ax5.set_xlabel('Force (Control Signal)', fontsize=10)
    ax5.set_ylabel('Membership', fontsize=10)
    ax5.set_title('Force Output', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_ylim(-0.05, 1.1)
    ax5.set_xlim(force.universe.min(), force.universe.max())
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.grid(True, alpha=0.3)

    axes[2, 1].axis('off')

    fig.suptitle('Fuzzy Controller Membership Functions (All 5 Variables)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig('membership_functions.png', dpi=150, bbox_inches='tight')
    print("Saved membership functions to: membership_functions.png")

    plt.show(block=True)


def plot_simulation_results(time_steps, angles, angular_velocities, cart_positions, actions, rewards):
    """
    Plot time-series data from a single simulation episode.

    Args:
        time_steps: List of simulation times
        angles: List of pole angles
        angular_velocities: List of pole angular velocities
        cart_positions: List of cart positions
        actions: List of control forces
        rewards: List of rewards
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(time_steps, angles, 'b-', linewidth=2, label='Pole Angle')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Target (Vertical)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (radians)')
    axes[0, 0].set_title('Pole Angle Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(time_steps, angular_velocities, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity Over Time')
    axes[0, 1].grid(True)

    axes[1, 0].plot(time_steps, cart_positions, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Cart Position (m)')
    axes[1, 0].set_title('Cart Position Over Time')
    axes[1, 0].grid(True)

    axes[1, 1].step(time_steps, actions, 'c-', linewidth=1.5, where='post')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Action (0=Left, 1=Right)')
    axes[1, 1].set_title('Actions Taken')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_control_surface(controller):
    """
    Plot 3D control surface showing force output as a function of angle and angular velocity.

    Args:
        controller: FuzzyCartPoleController instance
    """
    angle_range = np.linspace(-0.3, 0.3, 30)
    angular_velocity_range = np.linspace(-2.0, 2.0, 30)
    x, y = np.meshgrid(angle_range, angular_velocity_range)
    z = np.zeros_like(x, dtype=float)

    sim = ctrl.ControlSystemSimulation(controller.control_system)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sim.input['angle'] = x[i, j]
            sim.input['angular_velocity'] = y[i, j]
            sim.input['cart_position'] = 0.0
            sim.input['cart_velocity'] = 0.0

            sim.compute()

            z[i, j] = sim.output['force']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    ax.set_xlabel('Pole Angle (radians)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_zlabel('Force (Control Signal)')
    ax.set_title('Fuzzy Control Surface (cart at center, stationary)')

    ax.view_init(30, 200)

    fig.colorbar(surf)

    plt.show()


def plot_final_summary(total_reward, steps, episode_num):
    """
    Plot summary bar chart for a single episode.

    Args:
        total_reward: Total reward accumulated
        steps: Number of steps survived
        episode_num: Episode number
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = [steps, total_reward]
    labels = ['Steps\nSurvived', 'Total\nReward']
    colors = ['blue', 'green']

    bars = ax.bar(labels, metrics, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, metric in zip(bars, metrics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{metric:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'CartPole Episode {episode_num} Summary (Fuzzy Controller)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_episode_comparison(episode_rewards, episode_steps):
    """
    Plot comparison charts across multiple episodes.

    Args:
        episode_rewards: List of total rewards per episode
        episode_steps: List of steps survived per episode
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    episodes = range(1, len(episode_rewards) + 1)

    axes[0].bar(episodes, episode_steps, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Steps (500)')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Steps Survived', fontsize=11)
    axes[0].set_title('Steps per Episode', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(episodes, episode_rewards, color='green', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Reward (500)')
    axes[1].set_xlabel('Episode', fontsize=11)
    axes[1].set_ylabel('Total Reward', fontsize=11)
    axes[1].set_title('Reward per Episode', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    avg_steps = np.mean(episode_steps)
    avg_reward = np.mean(episode_rewards)
    fig.suptitle(f'Fuzzy Controller Performance (Avg Steps: {avg_steps:.1f}, Avg Reward: {avg_reward:.1f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
