"""
Gaussian Hotspot Search & Rescue Scenario Demo.

Tests Sequential Greedy planner on a realistic hotspot detection mission:
- Multiple Gaussian hotspots scattered across search area
- All robots start at depot (0,0)
- Goal: Locate and characterize hotspots quickly under budget constraint
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.environment import create_environment
from src.core.robot import Robot, BudgetType
from src.core.belief import create_gp_belief
from src.baselines.sequential_greedy_planner import SequentialGreedyIGPlanner
from src.planning.candidates.candidate_generator import CandidateGenerator


def create_hotspot_environment(bounds, n_hotspots=4, hotspot_scale=10.0, physical_scale=10.0):
    """
    Create environment with Gaussian hotspots.
    
    Args:
        bounds: Domain boundaries
        n_hotspots: Number of hotspots to generate
        hotspot_scale: Width of hotspots (in coordinate units)
        physical_scale: Meters per coordinate unit
    """
    # Generate random hotspot locations (avoid edges)
    np.random.seed(42)
    margin = 15  # Keep away from edges
    hotspot_locs = np.random.uniform(
        [bounds[0, 0] + margin, bounds[1, 0] + margin],
        [bounds[0, 1] - margin, bounds[1, 1] - margin],
        size=(n_hotspots, 2)
    )
    
    # Varying hotspot scales and amplitudes
    scales = [hotspot_scale * (0.8 + 0.4 * np.random.random()) for _ in range(n_hotspots)]
    amplitudes = [1.0] + [0.7 + 0.3 * np.random.random() for _ in range(n_hotspots - 1)]
    amplitudes = np.array(amplitudes) / np.sum(amplitudes)  # Normalize
    
    print(f"\nGenerating {n_hotspots} hotspots:")
    for i, (loc, scale, amp) in enumerate(zip(hotspot_locs, scales, amplitudes)):
        print(f"  Hotspot {i}: location={loc}, scale={scale:.1f}, weight={amp:.2f}")
    
    # Create covariance matrices (isotropic Gaussians)
    covariances = [scale**2 * np.eye(2) for scale in scales]
    
    # Create environment with Gaussian mixture
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name='gaussian_mixture',
        physical_scale=physical_scale,
        observation_noise=0.1,
        n_components=n_hotspots,
        means=hotspot_locs.tolist(),  # Pass as list to avoid numpy array comparison issue
        covs=covariances,
        weights=amplitudes.tolist()
    )
    
    return env, hotspot_locs


def calculate_metrics(env, results, hotspot_locs, bounds):
    """Calculate search & rescue metrics."""
    metrics = {}
    
    # Extract all measurement positions
    all_positions = []
    all_times = []
    for robot_id, measurements in results['robot_measurements'].items():
        for m in measurements:
            if isinstance(m, tuple):
                pos, val, timestamp = m
            else:
                pos = m['position']
                timestamp = m['timestamp']
            all_positions.append(pos)
            all_times.append(timestamp)
    
    all_positions = np.array(all_positions)
    all_times = np.array(all_times)
    
    # 1. Time to first hotspot (top 10% of field values)
    true_values = env.evaluate(all_positions)
    threshold = np.percentile(env.evaluate_grid(100)[2], 90)  # Top 10%
    hotspot_hits = true_values > threshold
    
    if np.any(hotspot_hits):
        first_hit_idx = np.where(hotspot_hits)[0][0]
        metrics['time_to_first_hotspot'] = all_times[first_hit_idx]
    else:
        metrics['time_to_first_hotspot'] = None
    
    # 2. Number of hotspots visited (within 5 units of hotspot center)
    visited_hotspots = set()
    for pos in all_positions:
        for i, hotspot_loc in enumerate(hotspot_locs):
            if np.linalg.norm(pos - hotspot_loc) < 5.0:
                visited_hotspots.add(i)
    
    metrics['hotspots_visited'] = len(visited_hotspots)
    metrics['hotspots_total'] = len(hotspot_locs)
    metrics['hotspot_recall'] = len(visited_hotspots) / len(hotspot_locs)
    
    # 3. Coverage (unique cells visited on 10x10 grid)
    cell_size = (bounds[0, 1] - bounds[0, 0]) / 10
    visited_cells = set()
    for pos in all_positions:
        cell_x = int((pos[0] - bounds[0, 0]) / cell_size)
        cell_y = int((pos[1] - bounds[1, 0]) / cell_size)
        visited_cells.add((cell_x, cell_y))
    
    metrics['coverage_cells'] = len(visited_cells)
    metrics['coverage_percent'] = len(visited_cells) / 100.0
    
    # 4. Redundant coverage (ratio of measurements to unique positions)
    unique_positions = np.unique(all_positions, axis=0)
    metrics['total_measurements'] = len(all_positions)
    metrics['unique_positions'] = len(unique_positions)
    metrics['redundancy_ratio'] = len(all_positions) / max(len(unique_positions), 1)
    
    # 5. Path efficiency
    metrics['total_distance'] = results['total_distance']
    metrics['avg_distance_per_robot'] = results['total_distance'] / len(results['robot_trajectories'])
    
    # 6. GP quality (RMSE over grid)
    if 'final_belief' in results:
        grid_res = 50
        X, Y, true_values_grid = env.evaluate_grid(resolution=grid_res)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        pred_values, _ = results['final_belief'].predict(grid_points)
        rmse = np.sqrt(np.mean((pred_values - true_values_grid.ravel())**2))
        metrics['final_rmse'] = rmse
    
    return metrics


def print_metrics(metrics, scenario_name):
    """Print metrics in organized format."""
    print(f"\n{'='*70}")
    print(f"METRICS: {scenario_name}")
    print(f"{'='*70}")
    
    print(f"\n🎯 Hotspot Detection:")
    if metrics['time_to_first_hotspot'] is not None:
        print(f"  Time to first hotspot:  {metrics['time_to_first_hotspot']:.1f}s")
    else:
        print(f"  Time to first hotspot:  NOT FOUND")
    print(f"  Hotspots visited:       {metrics['hotspots_visited']}/{metrics['hotspots_total']}")
    print(f"  Hotspot recall:         {metrics['hotspot_recall']*100:.1f}%")
    
    print(f"\n📊 Coverage:")
    print(f"  Unique cells visited:   {metrics['coverage_cells']}/100")
    print(f"  Coverage:               {metrics['coverage_percent']*100:.1f}%")
    print(f"  Total measurements:     {metrics['total_measurements']}")
    print(f"  Unique positions:       {metrics['unique_positions']}")
    print(f"  Redundancy ratio:       {metrics['redundancy_ratio']:.2f}")
    
    print(f"\n🚁 Path Efficiency:")
    print(f"  Total distance:         {metrics['total_distance']:.1f}m")
    print(f"  Avg per robot:          {metrics['avg_distance_per_robot']:.1f}m")
    
    if 'final_rmse' in metrics:
        print(f"\n🎓 GP Quality:")
        print(f"  Final RMSE:             {metrics['final_rmse']:.4f}")


def plot_results(env, results, hotspot_locs, bounds, scenario_name, metrics):
    """Plot search results with hotspots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Evaluate ground truth
    resolution = 100
    X, Y, Z = env.evaluate_grid(resolution=resolution)
    
    # Trajectory colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Ground Truth + Hotspots + Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.7)
    
    # Mark hotspots
    for i, loc in enumerate(hotspot_locs):
        ax1.scatter(loc[0], loc[1], s=300, c='red', marker='*', 
                   edgecolors='black', linewidths=2, zorder=10,
                   label=f'Hotspot {i}' if i == 0 else '')
        ax1.add_patch(plt.Circle(loc, 5.0, fill=False, edgecolor='red', 
                                linewidth=2, linestyle='--', alpha=0.6))
    
    # Plot trajectories
    for robot_id, traj in results['robot_trajectories'].items():
        if len(traj) > 0:
            traj = np.array(traj)
            color = colors[robot_id % len(colors)]
            ax1.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, 
                    alpha=0.7, label=f'Robot {robot_id}')
            ax1.scatter(traj[0, 0], traj[0, 1], s=200, c=color, marker='s',
                       edgecolors='black', linewidths=2, zorder=9)
    
    ax1.set_xlim(bounds[0, 0], bounds[0, 1])
    ax1.set_ylim(bounds[1, 0], bounds[1, 1])
    ax1.set_title('Ground Truth + Robot Trajectories', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (units)')
    ax1.set_ylabel('Y (units)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.colorbar(im1, ax=ax1, label='Field Value')
    
    # 2. Measurement Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.4)
    
    # Plot measurements
    for robot_id, measurements in results['robot_measurements'].items():
        if len(measurements) > 0:
            if isinstance(measurements[0], tuple):
                positions = np.array([m[0] for m in measurements])
            else:
                positions = np.array([m['position'] for m in measurements])
            color = colors[robot_id % len(colors)]
            ax2.scatter(positions[:, 0], positions[:, 1], s=60, c=color,
                       alpha=0.7, edgecolors='white', linewidths=1,
                       label=f'Robot {robot_id} ({len(positions)} pts)')
    
    # Mark hotspots
    for loc in hotspot_locs:
        ax2.scatter(loc[0], loc[1], s=300, c='red', marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
    
    ax2.set_xlim(bounds[0, 0], bounds[0, 1])
    ax2.set_ylim(bounds[1, 0], bounds[1, 1])
    ax2.set_title('Measurement Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (units)')
    ax2.set_ylabel('Y (units)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. GP Prediction
    ax3 = fig.add_subplot(gs[1, 0])
    if 'final_belief' in results:
        gp = results['final_belief']
        grid_points = np.c_[X.ravel(), Y.ravel()]
        gp_mean, _ = gp.predict(grid_points)
        gp_mean = gp_mean.reshape(X.shape)
        im3 = ax3.contourf(X, Y, gp_mean, levels=20, cmap='YlOrRd', alpha=0.7)
        
        # Mark hotspots
        for loc in hotspot_locs:
            ax3.scatter(loc[0], loc[1], s=300, c='red', marker='*',
                       edgecolors='black', linewidths=2, zorder=10)
        
        ax3.set_title('GP Prediction (Mean)', fontsize=12, fontweight='bold')
        fig.colorbar(im3, ax=ax3, label='Predicted Value')
    else:
        ax3.text(0.5, 0.5, 'No GP available', ha='center', va='center',
                transform=ax3.transAxes)
    
    ax3.set_xlim(bounds[0, 0], bounds[0, 1])
    ax3.set_ylim(bounds[1, 0], bounds[1, 1])
    ax3.set_xlabel('X (units)')
    ax3.set_ylabel('Y (units)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Format time to first hotspot
    time_str = f"{metrics['time_to_first_hotspot']:.1f}s" if metrics['time_to_first_hotspot'] else 'NOT FOUND'
    
    metrics_text = f"""
    MISSION METRICS
    {'─'*40}
    
    🎯 Hotspot Detection:
      • Time to first: {time_str}
      • Visited: {metrics['hotspots_visited']}/{metrics['hotspots_total']}
      • Recall: {metrics['hotspot_recall']*100:.1f}%
    
    📊 Coverage:
      • Cells: {metrics['coverage_cells']}/100 ({metrics['coverage_percent']*100:.1f}%)
      • Measurements: {metrics['total_measurements']}
      • Redundancy: {metrics['redundancy_ratio']:.2f}x
    
    🚁 Efficiency:
      • Total distance: {metrics['total_distance']:.1f}m
      • Avg per robot: {metrics['avg_distance_per_robot']:.1f}m
    """
    
    if 'final_rmse' in metrics:
        metrics_text += f"\n    🎓 GP Quality:\n      • RMSE: {metrics['final_rmse']:.4f}"
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Hotspot Search Results: {scenario_name}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(__file__).parent.parent / 'results' / f'hotspot_search_{scenario_name.lower().replace(" ", "_")}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    plt.show()


def run_scenario(scenario_name, n_robots, time_budget, n_hotspots, hotspot_scale, length_scale):
    """Run a single hotspot search scenario."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Robots: {n_robots}, Budget: {time_budget}s, Hotspots: {n_hotspots}")
    print(f"Hotspot scale: {hotspot_scale} units, GP length_scale: {length_scale}")
    
    # Setup
    bounds = np.array([[0, 100], [0, 100]])
    physical_scale = 10.0  # 10m per unit → 1km × 1km area
    
    # Create environment
    env, hotspot_locs = create_hotspot_environment(
        bounds, n_hotspots, hotspot_scale, physical_scale
    )
    
    # Initial samples
    n_init = 5
    init_points = np.random.uniform([5, 5], [95, 95], size=(n_init, 2))
    init_values = env.evaluate(init_points)
    
    # Create GP with higher variance to encourage exploration
    gp_belief = create_gp_belief(
        bounds,
        kernel_type='matern',
        length_scale=length_scale,
        variance=2.0,  # Higher variance for more uncertainty
        noise=0.1,
        use_normalized_coords=True
    )
    gp_belief.update(init_points, init_values)
    
    # Create robots (all start at depot)
    depot = np.array([0.0, 0.0])
    robots = []
    robot_speed = 5.0  # 5 m/s = 18 km/h
    
    for i in range(n_robots):
        robot = Robot(
            robot_id=i,
            initial_position=depot.copy(),
            max_speed=robot_speed,
            initial_budget=time_budget,
            budget_type=BudgetType.TIME,
            environment=env
        )
        robots.append(robot)
    
    # Generate candidates with very low variance threshold to get more candidates
    print(f"\nGenerating candidates...")
    generator = CandidateGenerator(
        bounds=bounds,
        quadtree_config={'max_depth': 7, 'min_cell_size': 5.0, 'variance_threshold': 0.01},
        sampling_config={'method': 'grid', 'points_per_cell': 5, 'min_spacing': 3.0}
    )
    candidate_sets = generator.generate_candidates(gp_belief, robots, budget_reserve=10.0)
    
    total_candidates = sum(len(cs.get_feasible_points()) for cs in candidate_sets.values())
    print(f"Generated {total_candidates} total candidates")
    
    # Run Sequential Greedy
    print(f"\n{'='*70}")
    print(f"RUNNING SEQUENTIAL GREEDY PLANNER")
    print(f"{'='*70}")
    
    # Override candidate grid
    all_candidates = set()
    for cand_set in candidate_sets.values():
        for point in cand_set.get_feasible_points():
            all_candidates.add(tuple(point))
    all_candidates = np.array([list(p) for p in all_candidates])
    
    config = {
        'candidate_resolution': 1,
        'seed': 42,
        'sensor_time': 2.0,
        'time_limit': time_budget
    }
    
    planner = SequentialGreedyIGPlanner(
        robots=robots,
        environment=env,
        gp_belief=gp_belief.copy(),
        config=config
    )
    planner.candidate_grid = all_candidates
    
    start_time = time.time()
    results = planner.execute_mission(max_iterations=1000, verbose=False)
    computation_time = time.time() - start_time
    
    print(f"\nMission Complete!")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.1f}m")
    print(f"  Computation time: {computation_time:.2f}s")
    
    # Calculate metrics
    metrics = calculate_metrics(env, results, hotspot_locs, bounds)
    metrics['computation_time'] = computation_time
    
    print_metrics(metrics, scenario_name)
    
    # Plot
    plot_results(env, results, hotspot_locs, bounds, scenario_name, metrics)
    
    return metrics


def main():
    """Run hotspot search scenarios."""
    print("="*70)
    print("GAUSSIAN HOTSPOT SEARCH & RESCUE SCENARIOS")
    print("="*70)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Easy (Broad Hotspots)',
            'n_robots': 2,
            'time_budget': 400,
            'n_hotspots': 3,
            'hotspot_scale': 15.0,
            'length_scale': 0.15
        },
        {
            'name': 'Medium (Mixed Hotspots)',
            'n_robots': 4,
            'time_budget': 400,
            'n_hotspots': 4,
            'hotspot_scale': 10.0,
            'length_scale': 0.10
        },
        {
            'name': 'Hard (Tight Hotspots)',
            'n_robots': 4,
            'time_budget': 200,
            'n_hotspots': 5,
            'hotspot_scale': 5.0,
            'length_scale': 0.05
        }
    ]
    
    # Run first scenario
    scenario = scenarios[1]  # Medium scenario
    scenario_name = scenario.pop('name')
    metrics = run_scenario(scenario_name, **scenario)
    
    print(f"\n{'='*70}")
    print(f"SCENARIO COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
