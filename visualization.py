"""
Visualization Functions for Fuzzy Controller Analysis

This module provides plotting functions to visualize the fuzzy controller's
internal structure and simulation performance.

VISUALIZATIONS PROVIDED
-----------------------
1. Membership Functions: Shows all fuzzy sets for each variable
2. Control Surface: 3D plot of force output vs angle and angular velocity
3. Simulation Results: Time-series plots of state variables during an episode
4. Episode Summary: Bar charts of episode performance
5. Episode Comparison: Multi-episode comparison charts

DEPENDENCIES
------------
- matplotlib with TkAgg backend for interactive plots
- numpy for data manipulation
- skfuzzy for control system simulation (control surface only)

USAGE
-----
All functions take the controller instance and/or simulation data as input:

    from visualization import plot_membership_functions, plot_control_surface
    from fuzzy_controller import FuzzyCartPoleController

    controller = FuzzyCartPoleController()
    plot_membership_functions(controller)
    plot_control_surface(controller)

FILE OUTPUT
-----------
- plot_membership_functions() saves to 'membership_functions.png'
- Other functions only display, no file save
"""

# Force TkAgg backend for consistent cross-platform display
# Must be set before importing pyplot
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skfuzzy import control as ctrl


def plot_membership_functions(controller):
    """
    Plot membership functions for all 5 fuzzy variables.

    Creates a 3x2 grid showing the membership functions for:
    - Pole Angle (5 terms: NL, NS, Z, PS, PL)
    - Angular Velocity (3 terms: N, Z, P)
    - Cart Position (3 terms: N, Z, P)
    - Cart Velocity (3 terms: N, Z, P)
    - Force Output (5 terms: NL, NS, Z, PS, PL)

    Args:
        controller: FuzzyCartPoleController instance with initialized
                   membership functions

    Output:
        - Displays interactive matplotlib window
        - Saves PNG to 'membership_functions.png' (150 DPI)

    Visual Design:
        - Color scheme: Red tones for negative, green for zero, blue for positive
        - Filled regions with alpha=0.3 for overlap visibility
        - Solid lines for precise boundary identification
        - Consistent y-axis [0, 1.1] for membership degree comparison

    Layout:
        +-------------------+-------------------+
        |   Pole Angle      | Angular Velocity  |
        +-------------------+-------------------+
        |   Cart Position   |   Cart Velocity   |
        +-------------------+-------------------+
        |   Force Output    |     (empty)       |
        +-------------------+-------------------+
    """
    # =========================================================================
    # GET FUZZY VARIABLES FROM CONTROLLER
    # =========================================================================
    angle, angular_velocity, force, cart_position, cart_velocity = controller.get_membership_functions()

    # =========================================================================
    # COLOR AND LABEL SCHEMES
    # =========================================================================
    # Consistent colors across all plots:
    # - Dark red (#8B0000) for large negative
    # - Light red (#E74C3C) for small negative
    # - Green (#27AE60) for zero/center
    # - Light blue (#3498DB) for small positive
    # - Dark blue (#00008B) for large positive
    colors = {'NL': '#8B0000', 'NS': '#E74C3C', 'Z': '#27AE60', 'PS': '#3498DB', 'PL': '#00008B', 'N': '#8B0000', 'P': '#00008B'}
    labels = {'NL': 'Neg Large', 'NS': 'Neg Small', 'Z': 'Zero', 'PS': 'Pos Small', 'PL': 'Pos Large', 'N': 'Negative', 'P': 'Positive'}

    # =========================================================================
    # CREATE FIGURE AND SUBPLOTS
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # -------------------------------------------------------------------------
    # SUBPLOT 1: Pole Angle (top-left)
    # -------------------------------------------------------------------------
    # Most critical variable - 5 terms with tight Z zone
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
    ax1.set_xlim(angle.universe.min(), angle.universe.max())
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # SUBPLOT 2: Angular Velocity (top-right)
    # -------------------------------------------------------------------------
    # Provides "derivative" control - 3 terms with tight Z
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

    # -------------------------------------------------------------------------
    # SUBPLOT 3: Cart Position (middle-left)
    # -------------------------------------------------------------------------
    # Secondary objective - 3 terms with early activation
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

    # -------------------------------------------------------------------------
    # SUBPLOT 4: Cart Velocity (middle-right)
    # -------------------------------------------------------------------------
    # Least critical - 3 terms with moderate sensitivity
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
    ax4.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # SUBPLOT 5: Force Output (bottom-left)
    # -------------------------------------------------------------------------
    # Controller output - 5 terms with stronger PS/NS
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

    # -------------------------------------------------------------------------
    # SUBPLOT 6: Empty (bottom-right)
    # -------------------------------------------------------------------------
    # Hide unused subplot for clean layout
    axes[2, 1].axis('off')

    # =========================================================================
    # FINALIZE AND DISPLAY
    # =========================================================================
    fig.suptitle('Fuzzy Controller Membership Functions (All 5 Variables)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file for documentation/reports
    plt.savefig('membership_functions.png', dpi=150, bbox_inches='tight')
    print("Saved membership functions to: membership_functions.png")

    # Display interactive window (blocks until closed)
    plt.show(block=True)


def plot_simulation_results(time_steps, angles, angular_velocities, cart_positions, actions, rewards):
    """
    Plot time-series data from a single simulation episode.

    Creates a 2x2 grid showing how state variables evolved during the episode.
    Useful for analyzing controller behavior and identifying failure patterns.

    Args:
        time_steps: List of simulation times (seconds)
        angles: List of pole angles at each step (radians)
        angular_velocities: List of pole angular velocities (rad/s)
        cart_positions: List of cart positions (meters)
        actions: List of control forces applied (Newtons)
        rewards: List of rewards received (not plotted, for compatibility)

    Layout:
        +-------------------+-------------------+
        |   Pole Angle      | Angular Velocity  |
        +-------------------+-------------------+
        |   Cart Position   |   Control Action  |
        +-------------------+-------------------+

    Interpretation:
        - Pole Angle: Should oscillate around 0 (red dashed line)
        - Angular Velocity: Oscillation indicates active control
        - Cart Position: Drift indicates position correction issues
        - Control Action: Step plot shows force magnitude and direction
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # -------------------------------------------------------------------------
    # SUBPLOT 1: Pole Angle vs Time (top-left)
    # -------------------------------------------------------------------------
    # Target is 0 (vertical) - deviations should be corrected quickly
    axes[0, 0].plot(time_steps, angles, 'b-', linewidth=2, label='Pole Angle')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Target (Vertical)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (radians)')
    axes[0, 0].set_title('Pole Angle Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # -------------------------------------------------------------------------
    # SUBPLOT 2: Angular Velocity vs Time (top-right)
    # -------------------------------------------------------------------------
    # Shows rate of pole rotation - spikes indicate recovery maneuvers
    axes[0, 1].plot(time_steps, angular_velocities, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity Over Time')
    axes[0, 1].grid(True)

    # -------------------------------------------------------------------------
    # SUBPLOT 3: Cart Position vs Time (bottom-left)
    # -------------------------------------------------------------------------
    # Should stay near center (0) - drift toward ±2.4m indicates issues
    axes[1, 0].plot(time_steps, cart_positions, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Cart Position (m)')
    axes[1, 0].set_title('Cart Position Over Time')
    axes[1, 0].grid(True)

    # -------------------------------------------------------------------------
    # SUBPLOT 4: Control Action vs Time (bottom-right)
    # -------------------------------------------------------------------------
    # Step plot shows discrete control decisions
    # Positive = push right, Negative = push left
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

    Creates a wireframe/surface plot showing how the fuzzy controller maps
    (angle, angular_velocity) pairs to force outputs. Cart position and velocity
    are held constant at 0 to isolate the pole-balancing behavior.

    Args:
        controller: FuzzyCartPoleController instance with initialized control system

    Fixed Inputs (for visualization):
        - cart_position = 0.0 (centered)
        - cart_velocity = 0.0 (stationary)

    This isolates the core pole-balancing response, which is the primary control
    objective. Position correction behavior requires varying cart_position.

    Surface Interpretation:
        - X-axis: Pole angle (radians)
        - Y-axis: Angular velocity (rad/s)
        - Z-axis: Control force (Newtons)
        - Positive Z (yellow/green): Push cart right
        - Negative Z (purple/blue): Push cart left

    Expected Pattern:
        - Diagonal ridge from (-angle, +velocity) to (+angle, -velocity)
        - Strong positive force in upper-right (pole falling right)
        - Strong negative force in lower-left (pole falling left)
        - Near-zero force along anti-diagonal (pole recovering naturally)

    View Angle:
        Elevation 30°, Azimuth 200° for optimal visibility of surface features
    """
    # =========================================================================
    # CREATE SAMPLE GRID
    # =========================================================================
    # Grid spans typical operating ranges for pole balancing
    # 30x30 points provides smooth surface while keeping computation fast
    angle_range = np.linspace(-0.3, 0.3, 30)           # ±17° (well within failure)
    angular_velocity_range = np.linspace(-2.0, 2.0, 30)  # ±2 rad/s (typical range)
    x, y = np.meshgrid(angle_range, angular_velocity_range)
    z = np.zeros_like(x, dtype=float)

    # =========================================================================
    # COMPUTE FORCE OUTPUT FOR EACH GRID POINT
    # =========================================================================
    # Create fresh simulation instance to avoid state pollution
    sim = ctrl.ControlSystemSimulation(controller.control_system)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Set inputs for this grid point
            sim.input['angle'] = x[i, j]
            sim.input['angular_velocity'] = y[i, j]
            sim.input['cart_position'] = 0.0   # Fixed: cart at center
            sim.input['cart_velocity'] = 0.0   # Fixed: cart stationary

            # Compute fuzzy inference
            sim.compute()

            # Store force output
            z[i, j] = sim.output['force']

    # =========================================================================
    # CREATE 3D SURFACE PLOT
    # =========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with viridis colormap (perceptually uniform)
    # rstride/cstride=1 for smooth surface
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    # =========================================================================
    # CONFIGURE AXES AND LABELS
    # =========================================================================
    ax.set_xlabel('Pole Angle (radians)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_zlabel('Force (Control Signal)')
    ax.set_title('Fuzzy Control Surface (cart at center, stationary)')

    # Set viewing angle for optimal surface visibility
    # Elevation 30° shows both peaks, Azimuth 200° shows diagonal structure
    ax.view_init(30, 200)

    # Add colorbar for force magnitude reference
    fig.colorbar(surf)

    plt.show()


def plot_final_summary(total_reward, steps, episode_num):
    """
    Plot summary bar chart for a single episode.

    Creates a simple bar chart comparing steps survived and total reward
    for the completed episode.

    Args:
        total_reward: Total reward accumulated during episode
        steps: Number of steps survived before termination
        episode_num: Episode number (for title)

    Note:
        In CartPole, reward = 1.0 per step, so total_reward equals steps.
        Both are plotted for consistency with other RL environments where
        this relationship may not hold.

    Bar Colors:
        - Blue: Steps survived
        - Green: Total reward
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = [steps, total_reward]
    labels = ['Steps\nSurvived', 'Total\nReward']
    colors = ['blue', 'green']

    # Create bar chart
    bars = ax.bar(labels, metrics, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on top of each bar
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

    Creates side-by-side bar charts comparing steps and rewards across
    all completed episodes. Useful for assessing controller consistency.

    Args:
        episode_rewards: List of total rewards per episode
        episode_steps: List of steps survived per episode

    Layout:
        +-------------------+-------------------+
        |  Steps/Episode    | Rewards/Episode   |
        +-------------------+-------------------+

    Reference Lines:
        - Red dashed line at 500: Indicates typical max episode length
        - Bars reaching this line indicate successful full episodes

    Statistics:
        - Figure title shows average steps and average reward
        - Helps identify if controller performance is stable or variable

    Interpretation:
        - Consistent tall bars: Stable controller
        - Variable height bars: Controller struggles with some initial conditions
        - All bars near max: Excellent performance
        - Early terminations (short bars): Identify problematic episodes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate episode numbers for x-axis
    episodes = range(1, len(episode_rewards) + 1)

    # -------------------------------------------------------------------------
    # LEFT: Steps per Episode
    # -------------------------------------------------------------------------
    axes[0].bar(episodes, episode_steps, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Steps (500)')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Steps Survived', fontsize=11)
    axes[0].set_title('Steps per Episode', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # RIGHT: Rewards per Episode
    # -------------------------------------------------------------------------
    axes[1].bar(episodes, episode_rewards, color='green', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=500, color='r', linestyle='--', linewidth=2, label='Max Reward (500)')
    axes[1].set_xlabel('Episode', fontsize=11)
    axes[1].set_ylabel('Total Reward', fontsize=11)
    axes[1].set_title('Reward per Episode', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # SUMMARY STATISTICS IN TITLE
    # -------------------------------------------------------------------------
    avg_steps = np.mean(episode_steps)
    avg_reward = np.mean(episode_rewards)
    fig.suptitle(f'Fuzzy Controller Performance (Avg Steps: {avg_steps:.1f}, Avg Reward: {avg_reward:.1f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
