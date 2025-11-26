"""
Demo: Random Multi-Robot Planner with all robots starting at origin.

Shows how robots disperse from (0, 0) using random walk.

UNITS:
  - All distances in METERS
  - All times in SECONDS
  - Speeds in METERS/SECOND
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import SyntheticEnvironment
from src.baselines.random_planner import RandomMultiRobotPlanner


# Visualization constants
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ROBOT_SIZE = 100
ROBOT_EDGE = 1.5
ROBOT_FONT = 8
CIRCLE_LINE = 1.2
CIRCLE_ALPHA = 0.6


def setup_plot_style():
    """Configure matplotlib for clean visualizations."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'font.family': 'sans-serif'
    })


def demo_random_planner_from_origin():
    """Demonstrate random planner with all robots at origin."""
    
    print("=" * 70)
    print("Random Multi-Robot Planner Demo - Origin Start")
    print("=" * 70)
    
    # Create Gaussian mixture environment (search & rescue scenario)
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,  # 1 coord unit = 1 meter
        n_components=5,      # 5 hotspots
        covs=12.0            # Covariance scale
    )
    
    print(f"\nEnvironment: Gaussian Mixture (Search & Rescue Scenario)")
    print(f"  Bounds: {bounds[0].tolist()} × {bounds[1].tolist()} meters")
    print(f"  Hotspots: 5 Gaussian components")
    print(f"  Component width: 12.0m standard deviation")
    
    # Create 4 robots ALL starting at origin
    n_robots = 4
    origin = np.array([0.0, 0.0])
    time_budget = 30.0  # 30 seconds
    speed = 5.0  # 5 m/s
    
    robots = []
    colors = ['red', 'blue', 'green', 'orange']
    
    print(f"\n{'Robot Setup':-^70}")
    print(f"Number of robots: {n_robots}")
    print(f"Starting position: {origin} (ALL robots at same point)")
    print(f"Time budget: {time_budget}s per robot")
    print(f"Max speed: {speed} m/s")
    print(f"Max possible distance: {time_budget * speed} meters")
    
    for i in range(n_robots):
        robot = Robot(
            robot_id=i,
            initial_position=origin.copy(),
            budget_type=BudgetType.TIME,
            initial_budget=time_budget,
            max_speed=speed,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create random planner with local random walk
    step_size = 15.0  # 15 meter max step
    config = {
        'step_size': step_size,
        'max_attempts': 100,
        'seed': 123
    }
    
    print(f"\n{'Planner Configuration':-^70}")
    print(f"Type: Local Random Walk")
    print(f"Step size: {step_size} meters (max distance per step)")
    print(f"Strategy: Each robot picks random direction/distance within step_size")
    
    planner = RandomMultiRobotPlanner(
        robots=robots,
        environment=env,
        config=config
    )
    
    # Execute mission
    print(f"\n{'Executing Mission':-^70}")
    results = planner.execute_mission(max_iterations=100, verbose=True)
    
    # Print results
    print(f"\n{'Mission Results':-^70}")
    print(f"Total iterations: {results['stats']['iterations']}")
    print(f"Total measurements: {results['total_measurements']}")
    print(f"Total distance: {results['total_distance']:.1f}m")
    print(f"Average per robot: {results['total_distance']/n_robots:.1f}m")
    
    print(f"\n{'Per-Robot Statistics':-^70}")
    for i, robot in enumerate(robots):
        final_pos = robot.position
        distance_from_origin = np.linalg.norm(final_pos - origin)
        measurements = results['stats']['measurements_taken'][robot.id]
        distance = results['stats']['total_distance'][robot.id]
        
        print(f"\nRobot {robot.id} ({COLORS[i]}):")
        print(f"  Measurements taken: {measurements}")
        print(f"  Distance traveled: {distance:.1f}m")
        print(f"  Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        print(f"  Distance from origin: {distance_from_origin:.1f}m")
        print(f"  Budget used: {robot.initial_budget - robot.remaining_budget:.1f}s / {robot.initial_budget}s")
    
    # Visualize
    print(f"\n{'Creating Visualization':-^70}")
    
    setup_plot_style()
    
    # Create figure with 3 rows x 3 columns (matching assignment_demo layout)
    fig = plt.figure(figsize=(20, 16))
    
    # Evaluate environment grid
    X, Y, Z = env.evaluate_grid(resolution=100)
    
    # Row 1: 3D plots
    # 1. Ground truth 3D
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    # Plot sample positions on 3D surface
    all_sample_positions = []
    for robot in robots:
        measurements = results['robot_measurements'][robot.id]
        for pos, val, time in measurements:
            all_sample_positions.append(pos)
    
    if all_sample_positions:
        all_sample_positions = np.array(all_sample_positions)
        sample_values = env.evaluate(all_sample_positions)
        ax1.scatter(all_sample_positions[:, 0], all_sample_positions[:, 1], sample_values,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax1.set_xlabel('X', labelpad=8)
    ax1.set_ylabel('Y', labelpad=8)
    ax1.set_zlabel('Value', labelpad=8)
    ax1.set_title('Ground Truth + Samples', fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=220)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.05)
    
    # 2. Statistics 3D (or another 3D view)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    # Plot trajectories in 3D
    for i, robot in enumerate(robots):
        trajectory = results['robot_trajectories'][robot.id]
        measurements = results['robot_measurements'][robot.id]
        color = COLORS[i]
        
        if measurements:
            meas_pos = np.array([m[0] for m in measurements])
            meas_vals = env.evaluate(meas_pos)
            ax2.scatter(meas_pos[:, 0], meas_pos[:, 1], meas_vals,
                       c=color, s=30, marker='o', edgecolors='black', 
                       linewidths=0.8, zorder=10, alpha=0.8)
    
    ax2.set_xlabel('X', labelpad=8)
    ax2.set_ylabel('Y', labelpad=8)
    ax2.set_zlabel('Value', labelpad=8)
    ax2.set_title('Random Walk Samples', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=220)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.05)
    
    # 3. Another 3D view
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    # Create a coverage/density surface
    coverage_grid = np.zeros_like(Z)
    for robot in robots:
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            for pos, val, time in measurements:
                xi = int((pos[0] - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0]) * 99)
                yi = int((pos[1] - bounds[1, 0]) / (bounds[1, 1] - bounds[1, 0]) * 99)
                if 0 <= xi < 100 and 0 <= yi < 100:
                    coverage_grid[yi, xi] += 1
    
    surf3 = ax3.plot_surface(X, Y, coverage_grid, cmap='YlOrRd',
                              linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel('X', labelpad=8)
    ax3.set_ylabel('Y', labelpad=8)
    ax3.set_zlabel('Visits', labelpad=8)
    ax3.set_title('Coverage Density', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=220)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.05)
    
    # 4. All robot trajectories (2D)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.35)
    
    for robot, color in zip(robots, COLORS):
        robot_id = robot.id
        trajectory = results['robot_trajectories'][robot_id]
        measurements = results['robot_measurements'][robot_id]
        
        # Draw robot starting position
        ax4.scatter(trajectory[0, 0], trajectory[0, 1],
                   s=ROBOT_SIZE, c=color, marker='o', edgecolors='black',
                   linewidths=ROBOT_EDGE, label=f'Robot {robot_id}', zorder=10)
        
        # Draw trajectory
        ax4.plot(trajectory[:, 0], trajectory[:, 1],
                c=color, linewidth=2, alpha=0.7, linestyle='--', zorder=5)
        
        # Draw measurement locations
        if measurements:
            meas_pos = np.array([m[0] for m in measurements])
            ax4.scatter(meas_pos[:, 0], meas_pos[:, 1], s=80, c=color, marker='X',
                       edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
    
    ax4.set_xlim(bounds[0, 0], bounds[0, 1])
    ax4.set_ylim(bounds[1, 0], bounds[1, 1])
    ax4.set_title('All Robot Trajectories', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.2)
    
    # Row 2 & 3: Individual robot details (subplots 5-7 for first row, then continue if needed)
    subplot_positions = [5, 6, 7] + ([8, 9] if n_robots > 3 else [])
    for idx, robot in enumerate(robots[:min(5, n_robots)]):
        ax = fig.add_subplot(3, 3, subplot_positions[idx])
        
        # Plot environment with YlOrRd for variance/uncertainty feel
        ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.3)
        
        robot_id = robot.id
        trajectory = results['robot_trajectories'][robot_id]
        measurements = results['robot_measurements'][robot_id]
        color = COLORS[idx % len(COLORS)]
        
        # Draw reachability circle
        initial_budget = robot.initial_budget
        circle = Circle(trajectory[0], initial_budget,
                       fill=False, edgecolor=color, linewidth=CIRCLE_LINE,
                       linestyle='--', alpha=CIRCLE_ALPHA)
        ax.add_patch(circle)
        
        # Draw candidates (simulated as light gray dots around trajectory)
        if len(measurements) > 0:
            ax.scatter([m[0][0] for m in measurements], [m[0][1] for m in measurements],
                      c='lightgray', s=20, marker='.', alpha=0.4, label='Samples')
        
        # Draw robot position
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                  s=ROBOT_SIZE, c=color, marker='o', edgecolors='black',
                  linewidths=ROBOT_EDGE, zorder=10)
        ax.annotate(f'R{robot_id}', xy=trajectory[0],
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=ROBOT_FONT,
                   fontweight='bold', color='white', zorder=11)
        
        # Draw trajectory and targets
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   c=color, linewidth=2, alpha=0.7, linestyle='--', zorder=5)
            
            # Mark measurement points as targets
            for i, (pos, val, time) in enumerate(measurements):
                ax.scatter(pos[0], pos[1], s=80, c=color, marker='X',
                          edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
        
        distance_traveled = results['stats']['total_distance'][robot_id]
        
        ax.set_xlim(bounds[0, 0], bounds[0, 1])
        ax.set_ylim(bounds[1, 0], bounds[1, 1])
        ax.set_title(f'Robot {robot_id}: {len(measurements)} targets, '
                    f'{len(measurements)} samples', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/demo_random_planner.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("   Saved to: results/demo_random_planner.png")
    
    plt.show()
    
    print(f"\n{'Demo Complete':-^70}")
    print("\nKey Observations:")
    print("  • All robots started at (0, 0)")
    print("  • Random walk causes natural dispersion")
    print("  • No coordination - paths may overlap")
    print("  • Coverage is random (not targeting hotspots)")
    print("  • This is the LOWER BOUND baseline")
    

if __name__ == "__main__":
    demo_random_planner_from_origin()
