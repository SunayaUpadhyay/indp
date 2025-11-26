"""
Compare KrigingBelieverAssignment vs SequentialGreedyIGPlanner.

Both planners use the same Kriging Believer mechanism. This experiment
tests if they produce similar results with identical settings from assignment_demo.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from matplotlib.patches import Circle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.environment import create_environment
from src.core.robot import Robot, BudgetType
from src.core.belief import create_gp_belief
from src.baselines.sequential_greedy_planner import SequentialGreedyIGPlanner
from src.planning.candidates.candidate_generator import CandidateGenerator, CandidateSet
from src.planning.assignment.kriging_believer import KrigingBelieverAssignment


# Visualization constants (matching assignment_demo)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
ROBOT_SIZE = 100
ROBOT_EDGE = 1.5
ROBOT_FONT = 5
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


def create_robots(n_robots: int, env, budget: float, start_pos=None):
    """Create robots with identical starting positions."""
    robots = []
    robot_speed_ms = 2.0  # 2 m/s = 7.2 km/h (ground robot)
    
    if start_pos is None:
        start_pos = np.array([0.0, 0.0])
    
    for i in range(n_robots):
        robot = Robot(
            robot_id=i,
            initial_position=start_pos.copy(),
            max_speed=robot_speed_ms,
            initial_budget=budget,
            budget_type=BudgetType.TIME,
            environment=env
        )
        robots.append(robot)
    
    return robots


def run_sequential_greedy(env, robots, gp_belief, candidate_sets, sensor_time=1.0):
    """Run SequentialGreedyIGPlanner using candidate sets."""
    print("\n" + "="*70)
    print("RUNNING SEQUENTIAL GREEDY PLANNER")
    print("="*70)
    
    start_time = time.time()
    
    # Reset robots
    for robot in robots:
        robot.reset()
    
    # Extract all unique candidates from candidate sets
    all_candidates = set()
    for cand_set in candidate_sets.values():
        for point in cand_set.get_feasible_points():
            all_candidates.add(tuple(point))
    
    all_candidates = np.array([list(p) for p in all_candidates])
    
    print(f"  Using {len(all_candidates)} candidates from quadtree generation")
    
    config = {
        'candidate_resolution': 1,  # Will be overridden
        'seed': 42,
        'sensor_time': sensor_time,
        'time_limit': 100.0,  # Same 100s mission time limit as Kriging Believer
        'min_time_threshold': 3.0
    }
    
    planner = SequentialGreedyIGPlanner(
        robots=robots,
        environment=env,
        gp_belief=gp_belief.copy(),
        config=config
    )
    
    # Override candidate grid with same candidates from quadtree
    planner.candidate_grid = all_candidates
    print(f"  Overrode planner's candidate grid with {len(all_candidates)} quadtree candidates")

    # Provide per-robot feasible candidate sets to mirror assignment inputs
    robot_candidates = {
        rid: cand_set.get_feasible_points()
        for rid, cand_set in candidate_sets.items()
    }
    planner.set_robot_candidates(robot_candidates)
    
    results = planner.execute_mission(max_iterations=1000, verbose=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\nSequential Greedy Results:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.2f}m")
    print(f"  Iterations: {results.get('iterations', 'N/A')}")
    print(f"  Measurements per robot: {results['stats']['measurements_taken']}")
    print(f"  RUNTIME: {runtime:.3f} seconds")
    
    # Add runtime to results
    results['runtime'] = runtime
    
    # Debug: print unique waypoint positions
    for robot_id, traj in results['robot_trajectories'].items():
        if len(traj) > 0:
            unique_positions = np.unique(np.array(traj), axis=0)
            print(f"  Robot {robot_id}: {len(traj)} waypoints, {len(unique_positions)} unique positions")
    
    return results


def run_kriging_believer(env, robots, gp_belief, candidate_sets, sensor_time=1.0, time_limit=None):
    """Run KrigingBelieverAssignment."""
    print("\n" + "="*70)
    print("RUNNING KRIGING BELIEVER ASSIGNMENT")
    print("="*70)
    
    start_time = time.time()
    
    # Reset robots
    for robot in robots:
        robot.reset()
    
    # Calculate time limit if not provided
    if time_limit is None:
        time_limit = robots[0].initial_budget
    
    # Run KB assignment
    kb = KrigingBelieverAssignment(
        time_limit=time_limit,
        environment=env,
        min_time_threshold=3.0,  # Match assignment_demo
        sensor_time=sensor_time,
        verbose=False
    )
    
    assignments, samples = kb.assign_targets(
        robots=robots,
        candidate_sets=candidate_sets,
        gp_belief=gp_belief.copy(),
        environment_sampler=lambda pos: env.evaluate(pos.reshape(1, -1))[0]
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    stats = kb.get_statistics()
    
    print(f"\nKriging Believer Results:")
    print(f"  Total measurements: {stats['total_samples']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Measurements per robot: {[s['samples_collected'] for s in stats['robot_stats'].values()]}")
    print(f"  RUNTIME: {runtime:.3f} seconds")
    
    # Add runtime to stats
    stats['runtime'] = runtime
    
    return assignments, samples, stats


def plot_comparison(env, seq_results, kb_assignments, kb_samples, robots_seq, robots_kb, candidate_sets, bounds, seq_gp, kb_gp):
    """Plot comparison of both planners (matching assignment_demo style)."""
    setup_plot_style()
    
    # Evaluate grids
    resolution = 100
    X_true, Y_true, true_values = env.evaluate_grid(resolution=resolution)
    
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]
    
    # Create figure with 2 columns (Sequential Greedy | Kriging Believer)
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    
    # === SEQUENTIAL GREEDY COLUMN ===
    
    # 1. Sequential Greedy - Ground truth + samples
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf1 = ax1.plot_surface(X_true, Y_true, true_values, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    all_seq_positions = []
    for robot_id in seq_results['robot_measurements'].keys():
        measurements = seq_results['robot_measurements'][robot_id]
        if isinstance(measurements, list) and len(measurements) > 0:
            if isinstance(measurements[0], dict):
                for m in measurements:
                    all_seq_positions.append(m['position'])
            else:  # Tuples: (position, value, timestamp)
                for m in measurements:
                    all_seq_positions.append(m[0] if isinstance(m, tuple) else m.position)
    if len(all_seq_positions) > 0:
        all_seq_positions = np.array(all_seq_positions)
        sample_values = env.evaluate(all_seq_positions)
        ax1.scatter(all_seq_positions[:, 0], all_seq_positions[:, 1], sample_values,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax1.set_xlabel('X', labelpad=8)
    ax1.set_ylabel('Y', labelpad=8)
    ax1.set_zlabel('Value', labelpad=8)
    ax1.set_title(f'Sequential Greedy: Ground Truth + Samples\n({seq_results["total_measurements"]} samples)', 
                  fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=220)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.05)
    
    # 2. Sequential Greedy - GP Mean
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    seq_mean, _ = seq_gp.predict(points, return_std=False)
    seq_mean = seq_mean.reshape(X.shape)
    surf2 = ax2.plot_surface(X, Y, seq_mean, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    if len(all_seq_positions) > 0:
        gp_sample_mean, _ = seq_gp.predict(all_seq_positions)
        ax2.scatter(all_seq_positions[:, 0], all_seq_positions[:, 1], gp_sample_mean,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax2.set_xlabel('X', labelpad=8)
    ax2.set_ylabel('Y', labelpad=8)
    ax2.set_zlabel('Value', labelpad=8)
    ax2.set_title('Sequential Greedy: GP Belief (Final)', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=220)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.05)
    
    # 3. Sequential Greedy - GP Variance
    ax3 = fig.add_subplot(gs[2, 0], projection='3d')
    _, seq_std = seq_gp.predict(points, return_std=True)
    seq_variance = (seq_std ** 2).reshape(X.shape)
    surf3 = ax3.plot_surface(X, Y, seq_variance, cmap='YlOrRd',
                              linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel('X', labelpad=8)
    ax3.set_ylabel('Y', labelpad=8)
    ax3.set_zlabel('Variance', labelpad=8)
    ax3.set_title('Sequential Greedy: GP Uncertainty', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=220)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.05)
    
    # 4. Sequential Greedy - Trajectories
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.contourf(X, Y, seq_variance, levels=20, cmap='YlOrRd', alpha=0.35)
    
    for robot, color in zip(robots_seq, COLORS):
        robot_id = robot.id
        traj = seq_results['robot_trajectories'][robot_id]
        
        if len(traj) > 0:
            traj = np.array(traj)
            # Starting position
            ax4.scatter(traj[0, 0], traj[0, 1], s=ROBOT_SIZE, c=color, marker='o',
                       edgecolors='black', linewidths=ROBOT_EDGE, label=f'Robot {robot_id}', zorder=10)
            
            # Trajectory
            ax4.plot(traj[:, 0], traj[:, 1], c=color, linewidth=2, alpha=0.7,
                    linestyle='--', zorder=5)
            
            # Waypoints
            for i, waypoint in enumerate(traj[1:], 1):
                ax4.scatter(waypoint[0], waypoint[1], s=80, c=color, marker='X',
                           edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
    
    ax4.set_xlim(bounds[0, 0], bounds[0, 1])
    ax4.set_ylim(bounds[1, 0], bounds[1, 1])
    ax4.set_title('Sequential Greedy: All Robot Trajectories', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.2)
    
    # === KRIGING BELIEVER COLUMN ===
    
    # 5. Kriging Believer - Ground truth + samples
    ax5 = fig.add_subplot(gs[0, 1], projection='3d')
    surf5 = ax5.plot_surface(X_true, Y_true, true_values, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    all_kb_positions = []
    for robot_samples in kb_samples.values():
        for pos, val, time in robot_samples:
            all_kb_positions.append(pos)
    if len(all_kb_positions) > 0:
        all_kb_positions = np.array(all_kb_positions)
        sample_values = env.evaluate(all_kb_positions)
        ax5.scatter(all_kb_positions[:, 0], all_kb_positions[:, 1], sample_values,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax5.set_xlabel('X', labelpad=8)
    ax5.set_ylabel('Y', labelpad=8)
    ax5.set_zlabel('Value', labelpad=8)
    total_kb_samples = sum(len(s) for s in kb_samples.values())
    ax5.set_title(f'Kriging Believer: Ground Truth + Samples\n({total_kb_samples} samples)', 
                  fontweight='bold', pad=10)
    ax5.view_init(elev=25, azim=220)
    fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10, pad=0.05)
    
    # 6. Kriging Believer - GP Mean
    ax6 = fig.add_subplot(gs[1, 1], projection='3d')
    kb_mean, _ = kb_gp.predict(points, return_std=False)
    kb_mean = kb_mean.reshape(X.shape)
    surf6 = ax6.plot_surface(X, Y, kb_mean, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    if len(all_kb_positions) > 0:
        gp_sample_mean, _ = kb_gp.predict(all_kb_positions)
        ax6.scatter(all_kb_positions[:, 0], all_kb_positions[:, 1], gp_sample_mean,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax6.set_xlabel('X', labelpad=8)
    ax6.set_ylabel('Y', labelpad=8)
    ax6.set_zlabel('Value', labelpad=8)
    ax6.set_title('Kriging Believer: GP Belief (Final)', fontweight='bold', pad=10)
    ax6.view_init(elev=25, azim=220)
    fig.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10, pad=0.05)
    
    # 7. Kriging Believer - GP Variance
    ax7 = fig.add_subplot(gs[2, 1], projection='3d')
    _, kb_std = kb_gp.predict(points, return_std=True)
    kb_variance = (kb_std ** 2).reshape(X.shape)
    surf7 = ax7.plot_surface(X, Y, kb_variance, cmap='YlOrRd',
                              linewidth=0, antialiased=True, alpha=0.95)
    ax7.set_xlabel('X', labelpad=8)
    ax7.set_ylabel('Y', labelpad=8)
    ax7.set_zlabel('Variance', labelpad=8)
    ax7.set_title('Kriging Believer: GP Uncertainty', fontweight='bold', pad=10)
    ax7.view_init(elev=25, azim=220)
    fig.colorbar(surf7, ax=ax7, shrink=0.5, aspect=10, pad=0.05)
    
    # 8. Kriging Believer - Trajectories
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.contourf(X, Y, kb_variance, levels=20, cmap='YlOrRd', alpha=0.35)
    
    for robot, color in zip(robots_kb, COLORS):
        robot_id = robot.id
        robot_assignments = kb_assignments[robot_id]
        robot_samples = kb_samples[robot_id]
        
        # Starting position
        start_pos = robot.trajectory[0].position
        ax8.scatter(start_pos[0], start_pos[1], s=ROBOT_SIZE, c=color, marker='o',
                   edgecolors='black', linewidths=ROBOT_EDGE, label=f'Robot {robot_id}', zorder=10)
        
        # Trajectory
        if robot_assignments:
            trajectory_points = [start_pos] + robot_assignments
            trajectory_points = np.array(trajectory_points)
            ax8.plot(trajectory_points[:, 0], trajectory_points[:, 1], c=color,
                    linewidth=2, alpha=0.7, linestyle='--', zorder=5)
            
            # Targets
            for i, target in enumerate(robot_assignments):
                ax8.scatter(target[0], target[1], s=80, c=color, marker='X',
                           edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
    
    ax8.set_xlim(bounds[0, 0], bounds[0, 1])
    ax8.set_ylim(bounds[1, 0], bounds[1, 1])
    ax8.set_title('Kriging Believer: All Robot Trajectories', fontweight='bold')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_aspect('equal')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.2)
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/kb_vs_sequential_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("\n   Saved to: results/kb_vs_sequential_comparison.png")
    plt.show()
    """Plot comparison of both planners."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Ground truth heatmap
    bounds = env.bounds
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    Z = env.evaluate(grid).reshape(X.shape)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots_seq)))
    
    # Sequential Greedy - Trajectories
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    
    for i, robot in enumerate(robots_seq):
        if len(robot.trajectory) > 1:
            # Extract trajectory positions from robot states
            traj = np.array([state.position for state in robot.trajectory])
            ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], 
                   label=f'Robot {robot.id}', markersize=6, linewidth=2, alpha=0.8)
            # Start position (square)
            ax.plot(traj[0, 0], traj[0, 1], 's', color=colors[i], markersize=12, 
                   edgecolor='white', linewidth=2)
            # End position (X)
            ax.plot(traj[-1, 0], traj[-1, 1], 'X', color=colors[i], markersize=12,
                   edgecolor='white', linewidth=2)
    
    ax.set_title(f'Sequential Greedy - Trajectories\n({seq_results["total_measurements"]} measurements)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Sequential Greedy - Measurement Points
    ax = fig.add_subplot(gs[1, 0])
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
    
    for i, robot_id in enumerate(seq_results['robot_measurements'].keys()):
        measurements = seq_results['robot_measurements'][robot_id]
        if len(measurements) > 0:
            if isinstance(measurements[0], dict):
                positions = np.array([m['position'] for m in measurements])
            else:
                positions = np.array([m[0] if isinstance(m, tuple) else m.position for m in measurements])
            ax.scatter(positions[:, 0], positions[:, 1], c=[colors[i]], 
                      s=80, alpha=0.7, edgecolor='white', linewidth=1.5,
                      label=f'Robot {robot_id} ({len(positions)} pts)')
    
    ax.set_title('Sequential Greedy - Measurement Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Kriging Believer - Trajectories
    ax = fig.add_subplot(gs[0, 1])
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    
    for i, robot in enumerate(robots_kb):
        if len(robot.trajectory) > 1:
            # Extract trajectory positions
            traj = np.array([state.position for state in robot.trajectory])
            ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], 
                   label=f'Robot {robot.id}', markersize=6, linewidth=2, alpha=0.8)
            # Start position (square)
            ax.plot(traj[0, 0], traj[0, 1], 's', color=colors[i], markersize=12,
                   edgecolor='white', linewidth=2)
            # End position (X)
            ax.plot(traj[-1, 0], traj[-1, 1], 'X', color=colors[i], markersize=12,
                   edgecolor='white', linewidth=2)
    
    total_kb_samples = sum(len(s) for s in kb_samples.values())
    ax.set_title(f'Kriging Believer - Trajectories\n({total_kb_samples} measurements)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Kriging Believer - Measurement Points
    ax = fig.add_subplot(gs[1, 1])
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
    
    for i, robot_id in enumerate(kb_samples.keys()):
        samples = kb_samples[robot_id]
        if len(samples) > 0:
            positions = np.array([s[0] for s in samples])
            ax.scatter(positions[:, 0], positions[:, 1], c=[colors[i]], 
                      s=80, alpha=0.7, edgecolor='white', linewidth=1.5,
                      label=f'Robot {robot_id} ({len(positions)} pts)')
    
    ax.set_title('Kriging Believer - Measurement Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Statistics comparison
    ax = fig.add_subplot(gs[2, :])
    
    metrics = ['Total\nMeasurements', 'Avg Path\nLength', 'Coverage\nArea']
    seq_vals = [
        seq_results['total_measurements'],
        seq_results['total_distance'] / len(robots_seq),
        len(set(tuple(m['position']) for robot_id in seq_results['robot_measurements'] 
                for m in seq_results['robot_measurements'][robot_id]))
    ]
    kb_vals = [
        total_kb_samples,
        sum(len(s) for s in kb_samples.values()) / len(robots_kb),  # Approximate
        len(set(tuple(s[0]) for samples in kb_samples.values() for s in samples))
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, seq_vals, width, label='Sequential Greedy', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, kb_vals, width, label='Kriging Believer',
                   color='coral', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Count / Value', fontsize=11, fontweight='bold')
    ax.set_title('Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Save figure
    output_path = Path(__file__).parent.parent / 'results' / 'kb_vs_sequential_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def main():
    """Run comparison experiment with exact settings from assignment_demo."""
    print("="*70)
    print("KRIGING BELIEVER VS SEQUENTIAL GREEDY COMPARISON")
    print("="*70)
    
    # Exact settings from assignment_demo
    env_name = 'townsend'
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])  # Townsend native domain
    n_init = 5
    time_limit = 100.0  # 100 seconds
    seed = 42
    
    np.random.seed(seed)
    
    print(f"\nExperiment Setup (matching assignment_demo):")
    print(f"  Environment: {env_name}")
    print(f"  Bounds: {bounds.tolist()}")
    print(f"  Physical area: 23.75m x 21.25m")
    print(f"  Time limit: {time_limit}s ({time_limit/60:.1f} minutes)")
    print(f"  Robot speed: 2.0 m/s")
    print(f"  Sensor time: 1.0s")
    print(f"  Initial samples: {n_init}")
    print(f"  GP length_scale: 0.2 (normalized space)")
    
    # Create environment and GP belief (exact same as assignment_demo)
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name=env_name,
        physical_scale=5.0,  # Each unit = 5 meters (23.75m x 21.25m area)
        observation_noise=0.05  # Low noise for Townsend's smooth regions
    )
    
    init_points = np.random.uniform(
        [bounds[0, 0], bounds[1, 0]],
        [bounds[0, 1], bounds[1, 1]],
        size=(n_init, 2)
    )
    init_values = env.evaluate(init_points)
    
    gp_belief = create_gp_belief(
        bounds,
        kernel_type='matern',
        length_scale=0.2,  # In normalized [0,1] space
        variance=1.0,
        noise=0.05,
        use_normalized_coords=True
    )
    gp_belief.update(init_points, init_values)
    
    # Create robots (exact same as assignment_demo)
    n_robots = 3
    robot_speed_ms = 2.0
    center_x = (bounds[0, 0] + bounds[0, 1]) / 2  # ~0.125
    center_y = (bounds[1, 0] + bounds[1, 1]) / 2  # ~-0.375
    robot_configs = [
        ([center_x, center_y], 100.0),  # Robot 0: start at center, 100s budget
        ([center_x, center_y], 100.0),  # Robot 1: start at center, 100s budget
        ([center_x, center_y], 100.0),  # Robot 2: start at center, 100s budget
    ]
    
    print(f"{'='*70}")
    print(f"STEP A: CANDIDATE GENERATION")
    print(f"{'='*70}")
    
    # Use same candidate generation as assignment_demo
    from src.planning.candidates.candidate_generator import CandidateGenerator
    
    generator = CandidateGenerator(
        bounds=bounds,
        quadtree_config={'max_depth': 4, 'min_cell_size': 0.8, 'variance_threshold': 0.15},
        sampling_config={'method': 'grid', 'points_per_cell': 4, 'min_spacing': 0.6}
    )
    
    # Create robots for candidate generation
    robots_temp = [
        Robot(i, np.array(pos), BudgetType.TIME, budget,
              max_speed=robot_speed_ms, environment=env)
        for i, (pos, budget) in enumerate(robot_configs)
    ]
    
    # Generate candidates using quadtree (same as assignment_demo)
    candidate_sets = generator.generate_candidates(gp_belief, robots_temp, budget_reserve=3.0)
    
    print(f"\nCandidate generation complete:")
    print(f"  Quadtree cells: {generator.quadtree.n_leaves}")
    print(f"  Max depth: {generator.quadtree.max_depth}")
    for robot_id, cand_set in candidate_sets.items():
        feasible = cand_set.get_feasible_points()
        print(f"  Robot {robot_id}: {len(feasible)}/{len(cand_set.points)} feasible candidates")

    
    # Run Sequential Greedy
    robots_seq = [
        Robot(i, np.array(pos), BudgetType.TIME, budget,
              max_speed=robot_speed_ms, environment=env)
        for i, (pos, budget) in enumerate(robot_configs)
    ]
    
    seq_results = run_sequential_greedy(env, robots_seq, gp_belief, candidate_sets, sensor_time=1.0)
    seq_final_gp = seq_results['final_belief']
    
    # Run Kriging Believer
    robots_kb = [
        Robot(i, np.array(pos), BudgetType.TIME, budget,
              max_speed=robot_speed_ms, environment=env)
        for i, (pos, budget) in enumerate(robot_configs)
    ]
    
    kb_assignments, kb_samples, kb_stats = run_kriging_believer(
        env, robots_kb, gp_belief, candidate_sets, sensor_time=1.0, time_limit=time_limit
    )
    
    # Get final GPs
    kb_final_gp = robots_kb[0].environment  # Get from assigner
    # Actually get it from the assignment object
    from src.planning.assignment.kriging_believer import KrigingBelieverAssignment
    kb_temp = KrigingBelieverAssignment(
        time_limit=time_limit,
        environment=env,
        min_time_threshold=5.0,
        sensor_time=1.0,
        verbose=False
    )
    # Reconstruct final GP from samples
    all_kb_pos = []
    all_kb_vals = []
    for samples in kb_samples.values():
        for pos, val, _ in samples:
            all_kb_pos.append(pos)
            all_kb_vals.append(val)
    
    kb_final_gp = gp_belief.copy()
    if len(all_kb_pos) > 0:
        kb_final_gp.update(np.array(all_kb_pos), np.array(all_kb_vals))
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Sequential Greedy:")
    print(f"  Total measurements: {seq_results['total_measurements']}")
    print(f"  Total distance: {seq_results['total_distance']:.2f}m")
    print(f"  Runtime: {seq_results['runtime']:.3f} seconds")
    
    print(f"\nKriging Believer:")
    print(f"  Total measurements: {kb_stats['total_samples']}")
    print(f"  Total time: {kb_stats['total_time']:.2f}s")
    print(f"  Runtime: {kb_stats['runtime']:.3f} seconds")
    
    # Calculate speedup/slowdown
    if seq_results['runtime'] > 0:
        speedup_ratio = seq_results['runtime'] / kb_stats['runtime']
        if speedup_ratio > 1:
            print(f"\n  -> Kriging Believer is {speedup_ratio:.2f}x FASTER than Sequential Greedy")
        else:
            print(f"\n  -> Sequential Greedy is {1/speedup_ratio:.2f}x FASTER than Kriging Believer")
    
    # Plot comparison
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_comparison(env, seq_results, kb_assignments, kb_samples, robots_seq, robots_kb,
                   candidate_sets, bounds, seq_final_gp, kb_final_gp)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
