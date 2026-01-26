"""
Visualization Functions for Fuzzy Controller Analysis

This module provides plotting functions to visualize the fuzzy logic controller's
membership functions, control surface, and simulation results.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

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
    # =========================================================================
    # Get Membership Functions from Controller
    # =========================================================================
    angle, angular_velocity, force, cart_position, cart_velocity = controller.get_membership_functions()

    # =========================================================================
    # Color and Label Configuration
    # Consistent styling across all plots
    # =========================================================================
    colors = {
        'NL': '#8B0000',  # Dark red for Negative Large
        'NS': '#E74C3C',  # Light red for Negative Small
        'Z': '#27AE60',   # Green for Zero
        'PS': '#3498DB',  # Light blue for Positive Small
        'PL': '#00008B',  # Dark blue for Positive Large
        'N': '#8B0000',   # Dark red for Negative (3-term systems)
        'P': '#00008B'    # Dark blue for Positive (3-term systems)
    }
    labels = {
        'NL': 'Neg Large',
        'NS': 'Neg Small',
        'Z': 'Zero',
        'PS': 'Pos Small',
        'PL': 'Pos Large',
        'N': 'Negative',
        'P': 'Positive'
    }

    # =========================================================================
    # Create Figure with 3x2 Subplot Grid
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # =========================================================================
    # Plot 1: Pole Angle Membership Functions (5 terms)
    # Zoomed to show detail in the active control region
    # =========================================================================
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
    ax1.axvline(x=0, color='black', linewidth=0.5, linestyle='--')  # Vertical reference (upright pole)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: Angular Velocity Membership Functions (3 terms)
    # =========================================================================
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

    # =========================================================================
    # Plot 3: Cart Position Membership Functions (3 terms)
    # =========================================================================
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

    # =========================================================================
    # Plot 4: Cart Velocity Membership Functions (3 terms)
    # =========================================================================
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
    ax4.axvline(x=0, color='black', linewidth=0.5, linestyle='--')  # Stationary reference
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Force Output Membership Functions (5 terms)
    # =========================================================================
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

    # Hide the unused subplot (bottom right)
    axes[2, 1].axis('off')

    # =========================================================================
    # Final Layout and Save
    # =========================================================================
    fig.suptitle('Fuzzy Controller Membership Functions (All 5 Variables)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file for reference
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
    # =========================================================================
    # Create 2x2 Subplot Grid for Time-Series Data
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # =========================================================================
    # Plot 1: Pole Angle Over Time
    # Shows how well the controller maintains vertical position
    # =========================================================================
    axes[0, 0].plot(time_steps, angles, 'b-', linewidth=2, label='Pole Angle')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Target (Vertical)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (radians)')
    axes[0, 0].set_title('Pole Angle Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # =========================================================================
    # Plot 2: Angular Velocity Over Time
    # Shows pole rotation dynamics
    # =========================================================================
    axes[0, 1].plot(time_steps, angular_velocities, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity Over Time')
    axes[0, 1].grid(True)

    # =========================================================================
    # Plot 3: Cart Position Over Time
    # Shows cart drift behavior (known issue)
    # =========================================================================
    axes[1, 0].plot(time_steps, cart_positions, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Cart Position (m)')
    axes[1, 0].set_title('Cart Position Over Time')
    axes[1, 0].grid(True)

    # =========================================================================
    # Plot 4: Control Actions Over Time
    # Shows the force applied by the controller
    # =========================================================================
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
    Plot three 3D control surfaces:
    1. Force vs Pole Angle and Angular Velocity (pole balancing)
    2. Force vs Cart Position and Cart Velocity (cart centering)
    3. Force vs Cart Position and Pole Angle (combined behavior)

    Args:
        controller: FuzzyCartPoleController instance
    """
    # =========================================================================
    # Create Figure with Three Subplots (1 row, 3 columns)
    # =========================================================================
    fig = plt.figure(figsize=(18, 6))

    # Create fuzzy simulation instance
    sim = ctrl.ControlSystemSimulation(controller.control_system)

    # =========================================================================
    # Surface 1: Pole Angle vs Angular Velocity
    # Shows the pole balancing behavior (cart held at center, stationary)
    # =========================================================================
    ax1 = fig.add_subplot(131, projection='3d')

    # Define input ranges for pole balancing surface
    angle_range = np.linspace(-0.15, 0.15, 30)
    angular_velocity_range = np.linspace(-1.0, 1.0, 30)
    x1, y1 = np.meshgrid(angle_range, angular_velocity_range)
    z1 = np.zeros_like(x1, dtype=float)

    # Compute force output for each point
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            sim.input['angle'] = x1[i, j]
            sim.input['angular_velocity'] = y1[i, j]
            sim.input['cart_position'] = 0.0  # Cart at center
            sim.input['cart_velocity'] = 0.0  # Cart stationary
            sim.compute()
            z1[i, j] = sim.output['force']

    # Plot surface
    surf1 = ax1.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='viridis',
                             linewidth=0.4, antialiased=True)

    ax1.set_xlabel('Pole Angle (rad)')
    ax1.set_ylabel('Angular Vel (rad/s)')
    ax1.set_zlabel('Force (N)')
    ax1.set_title('Pole Balancing\n(cart centered)', fontweight='bold', fontsize=10)
    ax1.view_init(30, 220)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # =========================================================================
    # Surface 2: Cart Position vs Cart Velocity
    # Shows the cart centering behavior (pole held vertical, stationary)
    # =========================================================================
    ax2 = fig.add_subplot(132, projection='3d')

    # Define input ranges for cart centering surface
    position_range = np.linspace(-2.0, 2.0, 30)
    velocity_range = np.linspace(-0.8, 0.8, 30)
    x2, y2 = np.meshgrid(position_range, velocity_range)
    z2 = np.zeros_like(x2, dtype=float)

    # Compute force output for each point
    for i in range(x2.shape[0]):
        for j in range(x2.shape[1]):
            sim.input['angle'] = 0.0  # Pole vertical
            sim.input['angular_velocity'] = 0.0  # Pole stationary
            sim.input['cart_position'] = x2[i, j]
            sim.input['cart_velocity'] = y2[i, j]
            sim.compute()
            z2[i, j] = sim.output['force']

    # Plot surface
    surf2 = ax2.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='plasma',
                             linewidth=0.4, antialiased=True)

    ax2.set_xlabel('Cart Position (m)')
    ax2.set_ylabel('Cart Velocity (m/s)')
    ax2.set_zlabel('Force (N)')
    ax2.set_title('Cart Centering\n(pole vertical)', fontweight='bold', fontsize=10)
    ax2.view_init(30, 220)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # =========================================================================
    # Surface 3: Cart Position vs Pole Angle
    # Shows combined behavior (velocities held at zero)
    # =========================================================================
    ax3 = fig.add_subplot(133, projection='3d')

    # Define input ranges for combined surface
    position_range_3 = np.linspace(-2.0, 2.0, 30)
    angle_range_3 = np.linspace(-0.15, 0.15, 30)
    x3, y3 = np.meshgrid(position_range_3, angle_range_3)
    z3 = np.zeros_like(x3, dtype=float)

    # Compute force output for each point
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            sim.input['angle'] = y3[i, j]  # Pole angle on Y-axis
            sim.input['angular_velocity'] = 0.0  # Pole stationary
            sim.input['cart_position'] = x3[i, j]  # Cart position on X-axis
            sim.input['cart_velocity'] = 0.0  # Cart stationary
            sim.compute()
            z3[i, j] = sim.output['force']

    # Plot surface
    surf3 = ax3.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='coolwarm',
                             linewidth=0.4, antialiased=True)

    ax3.set_xlabel('Cart Position (m)')
    ax3.set_ylabel('Pole Angle (rad)')
    ax3.set_zlabel('Force (N)')
    ax3.set_title('Position vs Angle\n(velocities zero)', fontweight='bold', fontsize=10)
    ax3.view_init(30, 220)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    # =========================================================================
    # Final Layout
    # =========================================================================
    fig.suptitle('Fuzzy Controller Control Surfaces', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save to file
    plt.savefig('control_surfaces.png', dpi=150, bbox_inches='tight')
    print("Saved control surfaces to: control_surfaces.png")

    plt.show()


def plot_final_summary(total_reward, steps, episode_num):
    """
    Plot summary bar chart for a single episode.

    Args:
        total_reward: Total reward accumulated
        steps: Number of steps survived
        episode_num: Episode number
    """
    # =========================================================================
    # Create Bar Chart for Episode Metrics
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = [steps, total_reward]
    labels = ['Steps\nSurvived', 'Total\nReward']
    colors = ['blue', 'green']

    # Create bars
    bars = ax.bar(labels, metrics, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on top of bars
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
    # =========================================================================
    # Create Side-by-Side Comparison Charts
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    episodes = range(1, len(episode_rewards) + 1)

    # =========================================================================
    # Plot 1: Steps Survived per Episode
    # =========================================================================
    axes[0].bar(episodes, episode_steps, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Steps (500)')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Steps Survived', fontsize=11)
    axes[0].set_title('Steps per Episode', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: Total Reward per Episode
    # =========================================================================
    axes[1].bar(episodes, episode_rewards, color='green', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Reward (500)')
    axes[1].set_xlabel('Episode', fontsize=11)
    axes[1].set_ylabel('Total Reward', fontsize=11)
    axes[1].set_title('Reward per Episode', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # =========================================================================
    # Add Summary Statistics to Title
    # =========================================================================
    avg_steps = np.mean(episode_steps)
    avg_reward = np.mean(episode_rewards)
    fig.suptitle(f'Fuzzy Controller Performance (Avg Steps: {avg_steps:.1f}, Avg Reward: {avg_reward:.1f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()