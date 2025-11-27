"""
Demo of Step A (Candidate Generation) + Step B (Kriging Believer Assignment).

This example demonstrates the complete two-step process:
1. Step A: Generate candidate sets for each robot using adaptive quadtree
2. Step B: Assign targets using kriging believer for conflict-free coordination

UNITS:
  - All distances in METERS
  - All times in SECONDS
  - Speeds in METERS/SECOND
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Any, Dict, List, Tuple

from src.core.environment import create_environment
from src.core.belief import create_gp_belief
from src.core.robot import Robot, BudgetType
from src.planning.candidates.candidate_generator import CandidateGenerator
from src.planning.assignment.kriging_believer import (
    KrigingBelieverAssignment,
    MCTSConfig
)
from src.baselines.auction_planner import AuctionVariancePlanner
from src.baselines.random_planner import RandomMultiRobotPlanner
from src.baselines.independent_greedy_planner import IndependentGreedyIGPlanner
from src.baselines.lawnmower_planner import LawnmowerPlanner
from src.baselines.sequential_greedy_planner import SequentialGreedyIGPlanner
from config.units import print_environment_info


# Visualization constants
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


def visualize_assignment_results(
    env, gp, robots, candidate_sets, assignments, samples, bounds
):
    """
    Visualize the results of Step B assignment.
    
    Shows:
    - Ground truth environment
    - GP belief after assignment
    - Robot trajectories and assigned targets
    - Individual robot assignment details
    """
    setup_plot_style()
    
    # Evaluate grids
    resolution = 100
    X_true, Y_true, true_values = env.evaluate_grid(resolution=resolution)
    
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]
    mean, std = gp.predict(points, return_std=True)
    mean = mean.reshape(X.shape)
    variance = (std ** 2).reshape(X.shape)
    
    # Create figure with 3 rows x 3 columns
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Ground truth
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_true, Y_true, true_values, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    # Add all sample points
    all_sample_positions = []
    for robot_samples in samples.values():
        for pos, val, time in robot_samples:
            all_sample_positions.append(pos)
    if len(all_sample_positions) > 0:
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
    
    # 2. GP mean after assignment
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, mean, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    if len(all_sample_positions) > 0:
        gp_sample_mean, _ = gp.predict(all_sample_positions)
        ax2.scatter(all_sample_positions[:, 0], all_sample_positions[:, 1], gp_sample_mean,
                   c='red', s=30, marker='o', edgecolors='black', linewidths=0.8, zorder=10)
    
    ax2.set_xlabel('X', labelpad=8)
    ax2.set_ylabel('Y', labelpad=8)
    ax2.set_zlabel('Value', labelpad=8)
    ax2.set_title('GP Belief (Final)', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=220)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.05)
    
    # 3. GP variance
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, variance, cmap='YlOrRd',
                              linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel('X', labelpad=8)
    ax3.set_ylabel('Y', labelpad=8)
    ax3.set_zlabel('Variance', labelpad=8)
    ax3.set_title('GP Uncertainty (Final)', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=220)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.05)
    
    # 4. All robot trajectories
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.contourf(X, Y, variance, levels=20, cmap='YlOrRd', alpha=0.35)
    
    for robot, color in zip(robots, COLORS):
        robot_id = robot.id
        robot_assignments = assignments[robot_id]
        robot_samples = samples[robot_id]
        
        # Draw robot starting position
        ax4.scatter(robot.trajectory[0].position[0], robot.trajectory[0].position[1],
                   s=ROBOT_SIZE, c=color, marker='o', edgecolors='black',
                   linewidths=ROBOT_EDGE, label=f'Robot {robot_id}', zorder=10)
        
        # Draw trajectory
        if robot_assignments:
            trajectory_points = [robot.trajectory[0].position] + robot_assignments
            trajectory_points = np.array(trajectory_points)
            ax4.plot(trajectory_points[:, 0], trajectory_points[:, 1],
                    c=color, linewidth=2, alpha=0.7, linestyle='--', zorder=5)
        
        # Draw assigned targets
        for target in robot_assignments:
            ax4.scatter(target[0], target[1], s=80, c=color, marker='X',
                       edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
    
    ax4.set_xlim(bounds[0, 0], bounds[0, 1])
    ax4.set_ylim(bounds[1, 0], bounds[1, 1])
    ax4.set_title('All Robot Trajectories', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.2)
    
    # 5-7. Individual robot details (subplots 5, 6, 7)
    for idx, (robot, color) in enumerate(zip(robots, COLORS)):
        ax = fig.add_subplot(3, 3, 5 + idx)
        ax.contourf(X, Y, variance, levels=20, cmap='YlOrRd', alpha=0.3)
        
        robot_id = robot.id
        robot_assignments = assignments[robot_id]
        robot_samples = samples[robot_id]
        cand_set = candidate_sets[robot_id]
        
        # Draw reachability circle
        initial_budget = robot.initial_budget
        circle = Circle(robot.trajectory[0].position, initial_budget,
                       fill=False, edgecolor=color, linewidth=CIRCLE_LINE,
                       linestyle='--', alpha=CIRCLE_ALPHA)
        ax.add_patch(circle)
        
        # Draw candidates
        feasible = cand_set.get_feasible_points()
        if len(feasible) > 0:
            ax.scatter(feasible[:, 0], feasible[:, 1], c='lightgray',
                      s=20, marker='.', alpha=0.4, label='Candidates')
        
        # Draw robot position
        ax.scatter(robot.trajectory[0].position[0], robot.trajectory[0].position[1],
                  s=ROBOT_SIZE, c=color, marker='o', edgecolors='black',
                  linewidths=ROBOT_EDGE, zorder=10)
        ax.annotate(f'R{robot_id}', xy=robot.trajectory[0].position,
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=ROBOT_FONT,
                   fontweight='bold', color='white', zorder=11)
        
        # Draw trajectory and targets
        if robot_assignments:
            trajectory_points = [robot.trajectory[0].position] + robot_assignments
            trajectory_points = np.array(trajectory_points)
            ax.plot(trajectory_points[:, 0], trajectory_points[:, 1],
                   c=color, linewidth=2, alpha=0.7, linestyle='--', zorder=5)
            
            for i, target in enumerate(robot_assignments):
                ax.scatter(target[0], target[1], s=80, c=color, marker='X',
                          edgecolors='darkgreen', linewidths=1.5, alpha=0.9, zorder=8)
                ax.annotate(f'{i+1}', xy=target, xytext=(0, -10),
                           textcoords='offset points', ha='center',
                           fontsize=6, color='darkgreen', fontweight='bold')
        
        ax.set_xlim(bounds[0, 0], bounds[0, 1])
        ax.set_ylim(bounds[1, 0], bounds[1, 1])
        ax.set_title(f'Robot {robot_id}: {len(robot_assignments)} targets, '
                    f'{len(robot_samples)} samples', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/assignment_demo.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("   Saved to: results/assignment_demo.png")
    plt.show()


def demo_assignment(
    env_name='townsend',
    bounds=None,
    n_init=5,
    time_limit=100,  # 100 seconds
    seed=42,
    planner_type='kriging'
):
    """
    Run complete demo of Steps A + B.
    
    Args:
        env_name: Environment function name
        bounds: Search space bounds [[x_min, x_max], [y_min, y_max]]
        n_init: Number of initial samples for GP
        time_limit: Mission time limit in seconds
        seed: Random seed
        planner_type: Type of planner to use ('kriging' or other)
    """
    np.random.seed(seed)
    env_key = env_name.lower()
    env_presets = {
        'gaussian_mixture': {
    'bounds': np.array([[0.0, 100.0], [0.0, 100.0]]),
    'physical_scale': 5.0,
    'observation_noise': 0.08,
    'env_kwargs': {'n_components': 5, 'covs': 10.0, 'weights': None},
    'gp_params': {
        'kernel_type': 'rbf',
        'length_scale': 0.06,      # in normalized coords → picks up ~6% of domain
        'variance': 2.5,
        'noise': 0.09,
        'use_normalized_coords': True
    },
    'quadtree_config': {
        'max_depth': 9,
        'min_cell_size': 0.6,
        'variance_threshold': 0.5
    },
    'sampling_config': {
        'method': 'grid',
        'points_per_cell': 9,
        'min_spacing': 0.5
    },
    'budget_reserve': 2.0
},
        'townsend': {
            'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
            'physical_scale': 20.0,
            'observation_noise': 0.05,
            'env_kwargs': {},
            'gp_params': {
                'kernel_type': 'matern',
                'length_scale': 0.2,
                'variance': 1.0,
                'noise': 0.05,
                'use_normalized_coords': True
            },
            'quadtree_config': {'max_depth': 10, 'min_cell_size': 0.8, 'variance_threshold': 0.15},
            'sampling_config': {'method': 'grid', 'points_per_cell': 8, 'min_spacing': 0.2},
            'budget_reserve': 3.0
        }
    }
    preset = env_presets.get(env_key, env_presets['gaussian_mixture'])
    if bounds is None:
        bounds = preset['bounds'].copy()
    else:
        bounds = np.array(bounds, dtype=float)
    
    print(f"\n{'='*70}")
    print(f"STEP A + B DEMONSTRATION")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"Bounds: {bounds.tolist()}")
    print(f"Time limit: {time_limit}s ({time_limit/60:.1f} minutes)")
    
    # === STEP A: CANDIDATE GENERATION ===
    print(f"\n{'='*70}")
    print(f"STEP A: CANDIDATE GENERATION")
    print(f"{'='*70}")
    
    # Create environment and GP belief
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name=env_key,
        physical_scale=preset['physical_scale'],
        observation_noise=preset['observation_noise'],
        **preset.get('env_kwargs', {})
    )
    init_points = np.random.uniform(
        [bounds[0, 0], bounds[1, 0]],
        [bounds[0, 1], bounds[1, 1]],
        size=(n_init, 2)
    )
    init_values = env.evaluate(init_points)
    # GP hyperparameters tuned per-environment (defaults come from preset table above)
    gp = create_gp_belief(bounds, **preset['gp_params'])
    gp.update(init_points, init_values)
    
    # Create robots with TIME budgets and environment link
    # Robot speed: 2 m/s = 7.2 km/h (ground robot)
    # Start all robots in the lower-left corner to stress long traversals
    robot_speed_ms = 5.0
    start_x = bounds[0, 0]
    start_y = bounds[1, 0]
    robot_configs = [
        ([start_x, start_y], 100.0),  # Robot 0: start at southwest corner, 100s budget
        ([start_x, start_y], 100.0),  # Robot 1: start at southwest corner, 100s budget
        ([start_x, start_y], 100.0),  # Robot 2: start at southwest corner, 100s budget
    ]
    
    robots = [
        Robot(i, np.array(pos), BudgetType.TIME, budget,
              max_speed=robot_speed_ms, environment=env)
        for i, (pos, budget) in enumerate(robot_configs)
    ]
    
    print(f"\nRobots initialized:")
    for robot in robots:
        print(f"  Robot {robot.id}: position={robot.position}, budget={robot.remaining_budget}")
    
    # Generate candidates with environment-specific refinement heuristics
    generator = CandidateGenerator(
        bounds=bounds,
        quadtree_config=preset['quadtree_config'],
        sampling_config=preset['sampling_config']
    )
    
    candidate_sets = generator.generate_candidates(
        gp,
        robots,
        budget_reserve=preset.get('budget_reserve', 2.0)
    )
    per_robot_candidates: Dict[int, np.ndarray] = {}
    
    print(f"\nCandidate generation complete:")
    print(f"  Quadtree cells: {generator.quadtree.n_leaves}")
    print(f"  Max depth: {generator.quadtree.max_depth}")
    
    for robot_id, cand_set in candidate_sets.items():
        feasible = cand_set.get_feasible_points()
        print(f"  Robot {robot_id}: {len(feasible)}/{len(cand_set.points)} feasible candidates")
        per_robot_candidates[robot_id] = np.array(feasible) if len(feasible) > 0 else np.empty((0, 2))
    
    planner_key = planner_type.lower()
    valid_types = {'random', 'auction', 'independent', 'lawnmower', 'sequential', 'kriging'}
    if planner_key not in valid_types:
        raise ValueError(f"planner_type must be one of {sorted(valid_types)}")
    label_map = {
        'random': 'RANDOM BASELINE',
        'auction': 'AUCTION-BASED',
        'independent': 'INDEPENDENT GREEDY',
        'lawnmower': 'LAWNMOWER COVERAGE',
        'sequential': 'SEQUENTIAL GREEDY',
        'kriging': 'KRIGING BELIEVER (MCTS)'
    }
    step_b_label = label_map[planner_key]
    print(f"{'='*70}")
    print(f"STEP B: {step_b_label} ASSIGNMENT")
    print(f"{'='*70}")
    
    assignments: Dict[int, List[np.ndarray]] = {}
    samples: Dict[int, List[Tuple[np.ndarray, float, float]]] = {}
    stats: Dict[str, Any]
    final_gp = gp.copy()
    
    if planner_key == 'kriging':
        print("  Using Kriging Believer with MCTS-backed acquisition and periodic candidate refresh")
        mcts_config = MCTSConfig(
            iterations=600,
            exploration_constant=1.2,
            max_depth=10,
            simulation_depth=5,
            discount_factor=0.9,
            pw_alpha=0.7,
            pw_constant=1.5,
            time_limit=1,
            verbose=False
        )
        kb_assignment = KrigingBelieverAssignment(
            time_limit=time_limit,
            environment=env,
            min_time_threshold=3.0,
            sensor_time=1.0,
            verbose=True,
            use_mcts_acquisition=False,
            mcts_config=mcts_config,
            mcts_candidate_limit=500,
            candidate_refresh_interval=500,
            candidate_budget_reserve=2.0
        )
        env_sampler = lambda pos: env.evaluate(pos.reshape(1, -1))[0]
        assignments, samples = kb_assignment.assign_targets(
            robots=robots,
            candidate_sets=candidate_sets,
            gp_belief=gp.copy(),
            environment_sampler=env_sampler,
            candidate_generator=generator
        )
        final_gp = kb_assignment.get_final_gp()
        stats = kb_assignment.get_statistics()
    else:
        # Shared configuration for BaseMultiRobotPlanner settings
        planner_config: Dict[str, Any] = {
            'sensor_time': 1.0,
            'time_limit': time_limit,
            'min_time_threshold': 3.0,
            'seed': seed,
            'verbose': True,
        }
        if planner_key == 'auction':
            planner_config.update({
                'grid_resolution': 50,
                'num_candidates': 35,
                'pool_factor': 2.0,
                'variance_threshold_frac': 0.2,
                'use_variance_only': True,
                'normalize_utility': True,
                'normalize_cost': True,
                'distance_units': 'meters',
                'min_cost_meters': 1.0,
            })
            planner = AuctionVariancePlanner(
                robots=robots,
                environment=env,
                gp_belief=gp,
                config=planner_config
            )
        elif planner_key == 'random':
            planner_config.update({
                'step_size': None,  # None => sample anywhere within bounds
                'max_attempts': 200,
            })
            planner = RandomMultiRobotPlanner(
                robots=robots,
                environment=env,
                gp_belief=gp,
                config=planner_config
            )
        elif planner_key == 'independent':
            planner_config.update({
                'candidate_resolution': 30,
            })
            planner = IndependentGreedyIGPlanner(
                robots=robots,
                environment=env,
                gp_belief=gp,
                config=planner_config
            )
        elif planner_key == 'lawnmower':
            planner_config.update({
                'stripe_width': 15.0,  # meters
                'orientation': 'vertical',
                'waypoint_spacing': 12.0,
            })
            planner = LawnmowerPlanner(
                robots=robots,
                environment=env,
                gp_belief=gp,
                config=planner_config
            )
        else:  # sequential greedy
            planner_config.update({
                'candidate_resolution': 30,
            })
            planner = SequentialGreedyIGPlanner(
                robots=robots,
                environment=env,
                gp_belief=gp,
                config=planner_config
            )
            seq_candidates = {
                rid: points for rid, points in per_robot_candidates.items() if len(points) > 0
            }
            if seq_candidates:
                planner.set_robot_candidates(seq_candidates)
        
        # Auction planner benefits from reusing Step-A candidates; random planner ignores them
        if isinstance(planner, AuctionVariancePlanner):
            all_candidate_points = set()
            for points in per_robot_candidates.values():
                for point in points:
                    all_candidate_points.add(tuple(point))
            if all_candidate_points:
                planner.eval_grid = np.array(sorted(all_candidate_points))
                planner.num_candidates = min(planner.num_candidates, len(all_candidate_points))
        
        # Run baseline assignment
        results = planner.execute_mission(max_iterations=1000, verbose=True)
        
        # Convert planner outputs to assignment_demo-friendly format
        for robot in robots:
            rid = robot.id
            trajectory = results['robot_trajectories'][rid]
            assignments[rid] = [np.array(pos) for pos in trajectory[1:]] if len(trajectory) > 1 else []
            measurements: List[Tuple[np.ndarray, float, float]] = []
            for pos, value, timestamp in robot.measurements:
                measurements.append((pos.copy(), value, timestamp))
            samples[rid] = measurements
        
        final_gp = planner.gp_belief
        stats = {
            'total_time': planner.simulation_clock,
            'total_samples': results['total_measurements'],
            'robot_stats': {
                robot.id: {
                    'samples_collected': len(samples[robot.id]),
                    'targets_assigned': len(assignments[robot.id]),
                    'final_budget': robot.remaining_budget,
                    'budget_used': robot.initial_budget - robot.remaining_budget,
                }
                for robot in robots
            }
        }
    
    print(f"\n{'='*70}")
    print(f"ASSIGNMENT STATISTICS")
    print(f"{'='*70}")
    print(f"Total mission time: {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} minutes)")
    print(f"Total samples collected: {stats['total_samples']}")
    print(f"\nPer-robot statistics:")
    for robot_id, robot_stats in stats['robot_stats'].items():
        print(f"  Robot {robot_id}:")
        print(f"    Targets assigned: {robot_stats['targets_assigned']}")
        print(f"    Samples collected: {robot_stats['samples_collected']}")
        print(f"    Budget used: {robot_stats['budget_used']:.1f} / {robots[robot_id].initial_budget:.1f}")
    
    # Visualize results
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    visualize_assignment_results(
        env, final_gp, robots, candidate_sets, assignments, samples, bounds
    )
    
    print(f"\n{'='*70}")
    print(f"DEMO COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    demo_assignment(planner_type='kriging')
 