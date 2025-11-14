"""
Full System Demo: Complete IPP Algorithm with Receding Horizon

This demo demonstrates the complete Informative Path Planning algorithm
integrating all steps (A-F) in a receding horizon control loop:

Steps per cycle:
  A) Candidate Generation - Quadtree adaptive refinement
  B) Assignment - Kriging Believer target allocation
  C) MCTS Planning - Tree search for optimal paths
  D) Segment Selection - Extract immediate actions
  E) Execution - Execute and collect measurements
  F) GP Update - Update belief with observations

The system runs until robots exhaust their budgets or a stopping condition is met.

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
from copy import deepcopy
from typing import List, Dict, Tuple

from src.core.environment import create_environment
from src.core.robot import Robot, BudgetType
from src.core.belief import create_gp_belief
from config.units import print_environment_info
from src.planning import (
    CandidateGenerator,
    KrigingBelieverAssignment,
    MCTSPlanner,
    MCTSConfig
)
from src.planning.candidates.candidate_generator import CandidateSet
from config.units import print_environment_info


class RecedingHorizonIPP:
    """
    Receding Horizon IPP Controller.
    
    Coordinates all planning steps (A-F) in a loop until robots
    complete their missions.
    """
    
    def __init__(
        self,
        env,
        robots: List[Robot],
        gp_belief,
        bounds: np.ndarray,
        candidate_config: dict = None,
        assignment_config: dict = None,
        mcts_config: MCTSConfig = None,
        sensor_time: float = 1.0,
        execution_step: int = 1,
        verbose: bool = True
    ):
        """
        Initialize the receding horizon controller.
        
        Args:
            env: Environment for ground truth measurements
            robots: List of Robot objects
            gp_belief: Initial GP belief
            bounds: Environment bounds
            candidate_config: Config for candidate generation
            assignment_config: Config for assignment planner
            mcts_config: Config for MCTS planner
            sensor_time: Time to take one measurement
            execution_step: Number of waypoints to execute before replanning
            verbose: Print progress information
        """
        self.env = env
        self.robots = robots
        self.gp_belief = gp_belief
        self.bounds = bounds
        self.sensor_time = sensor_time
        self.execution_step = execution_step
        self.verbose = verbose
        
        # Default configs
        self.candidate_config = candidate_config or {
            'quadtree_config': {
                'max_depth': 8,
                'min_cell_size': 2.0,
                'variance_threshold': 0.01
            },
            'sampling_config': {
                'method': 'grid',
                'points_per_cell': 4,
                'min_spacing': 7.0
            }
        }
        
        self.assignment_config = assignment_config or {
            'time_limit': 1000.0,  # Will be updated per cycle
            'min_time_threshold': 10.0,
            'sensor_time': sensor_time,
            'verbose': False
        }
        
        self.mcts_config = mcts_config or MCTSConfig(
            iterations=500,
            exploration_constant=1.414,
            max_depth=10,
            discount_factor=0.95,
            simulation_depth=3,
            use_progressive_widening=True,
            pw_alpha=0.5,
            pw_constant=1.0,
            verbose=False
        )
        
        # Initialize planners
        self.candidate_generator = CandidateGenerator(
            bounds=bounds,
            **self.candidate_config
        )
        
        # Use kriging believer for assignment (simpler greedy approach for receding horizon)
        self.use_kriging_believer = True
        self.mcts_planner = MCTSPlanner(config=self.mcts_config)
        
        # Tracking
        self.cycle_count = 0
        self.samples_per_robot = {robot.id: [] for robot in robots}
        self.paths_per_robot = {robot.id: [robot.position.copy()] for robot in robots}
        self.history = {
            'cycles': [],
            'candidates_generated': [],
            'assignments': [],
            'gp_variance': [],
            'samples_collected': 0
        }
        
    def run(self, max_cycles: int = 100) -> Dict:
        """
        Run the complete receding horizon IPP loop.
        
        Args:
            max_cycles: Maximum number of planning cycles
            
        Returns:
            Dictionary with execution results and statistics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STARTING RECEDING HORIZON IPP")
            print(f"{'='*70}")
            print(f"Robots: {len(self.robots)}")
            print(f"Initial GP samples: {len(self.gp_belief.X_train)}")
            print(f"Execution step: {self.execution_step} waypoints")
            print(f"Max cycles: {max_cycles}")
        
        while self.cycle_count < max_cycles:
            # Check if any robot has budget remaining
            active_robots = [r for r in self.robots if r.remaining_budget > 0]
            if len(active_robots) == 0:
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"ALL ROBOTS EXHAUSTED - MISSION COMPLETE")
                    print(f"{'='*70}")
                break
            
            self.cycle_count += 1
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"CYCLE {self.cycle_count}")
                print(f"{'='*70}")
                print(f"Active robots: {len(active_robots)}/{len(self.robots)}")
            
            # === STEP A: CANDIDATE GENERATION ===
            if self.verbose:
                print(f"\nStep A: Generating candidates...")
            
            candidate_sets_dict = self.candidate_generator.generate_candidates(
                self.gp_belief,
                active_robots
            )
            
            total_candidates = sum(len(cs.get_feasible_points()) for cs in candidate_sets_dict.values())
            self.history['candidates_generated'].append(total_candidates)
            
            if self.verbose:
                print(f"  Generated {total_candidates} total candidates")
                for robot_id, cs in candidate_sets_dict.items():
                    print(f"    Robot {robot_id}: {len(cs.get_feasible_points())} candidates")
            
            # Check if any robot has candidates
            if total_candidates == 0:
                if self.verbose:
                    print(f"  No candidates available - ending mission")
                break
            
            # === STEP B: ASSIGNMENT ===
            if self.verbose:
                print(f"\nStep B: Assigning targets...")
            
            # Simple greedy assignment with kriging believer
            assignments = self._greedy_assignment(active_robots, candidate_sets_dict)
            
            self.history['assignments'].append(deepcopy(assignments))
            
            if self.verbose:
                for robot_id, targets in assignments.items():
                    print(f"    Robot {robot_id}: {len(targets)} targets assigned")
            
            # === STEP C: MCTS PLANNING ===
            if self.verbose:
                print(f"\nStep C: MCTS planning...")
            
            planned_paths = {}
            for robot in active_robots:
                if len(assignments[robot.id]) == 0:
                    planned_paths[robot.id] = []
                    continue
                
                # Create candidate set from assigned targets
                assigned_candidates = CandidateSet(
                    robot_id=robot.id,
                    points=np.array(assignments[robot.id])
                )
                
                # Plan with MCTS
                path = self.mcts_planner.plan(
                    robot=robot,
                    candidates=assigned_candidates,
                    gp_belief=self.gp_belief,
                    sensor_time=self.sensor_time
                )
                
                planned_paths[robot.id] = path
                
                if self.verbose:
                    print(f"    Robot {robot.id}: planned {len(path)} waypoints")
            
            # === STEP D: SEGMENT SELECTION ===
            if self.verbose:
                print(f"\nStep D: Selecting execution segments...")
            
            execution_segments = {}
            for robot_id, path in planned_paths.items():
                # Extract first N waypoints (execution_step)
                segment = path[:self.execution_step] if len(path) > 0 else []
                execution_segments[robot_id] = segment
                
                if self.verbose and len(segment) > 0:
                    print(f"    Robot {robot_id}: executing {len(segment)} waypoint(s)")
            
            # === STEP E: EXECUTION ===
            if self.verbose:
                print(f"\nStep E: Executing segments...")
            
            cycle_samples = []
            for robot in active_robots:
                segment = execution_segments[robot.id]
                if len(segment) == 0:
                    continue
                
                for waypoint in segment:
                    # Check if robot has enough budget
                    distance = np.linalg.norm(waypoint - robot.position)
                    time_needed = distance / robot.max_speed + self.sensor_time
                    
                    if time_needed > robot.remaining_budget:
                        if self.verbose:
                            print(f"    Robot {robot.id}: insufficient budget, skipping waypoint")
                        break
                    
                    # Calculate travel time and new timestamp
                    travel_time = distance / robot.max_speed
                    new_timestamp = robot.state.timestamp + travel_time + self.sensor_time
                    
                    # Execute: move and measure
                    value = self.env.evaluate(waypoint.reshape(1, -1))[0]
                    
                    # Update robot
                    robot.move_to(waypoint, timestamp=new_timestamp, update_budget=True)
                    
                    # Record
                    sample = (waypoint.copy(), value, robot.state.timestamp)
                    self.samples_per_robot[robot.id].append(sample)
                    self.paths_per_robot[robot.id].append(waypoint.copy())
                    cycle_samples.append(sample)
                    
                    if self.verbose:
                        print(f"    Robot {robot.id}: sampled at {waypoint}, value={value:.4f}, budget={robot.remaining_budget:.1f}s")
            
            # === STEP F: GP UPDATE ===
            if self.verbose:
                print(f"\nStep F: Updating GP belief...")
            
            if len(cycle_samples) > 0:
                sample_positions = np.array([s[0] for s in cycle_samples])
                sample_values = np.array([s[1] for s in cycle_samples])
                
                self.gp_belief.update(sample_positions, sample_values)
                self.history['samples_collected'] += len(cycle_samples)
                
                if self.verbose:
                    print(f"    Updated GP with {len(cycle_samples)} new samples")
                    print(f"    Total samples: {len(self.gp_belief.X_train)}")
            
            # Track GP variance
            grid_points = self._get_grid_points(resolution=50)
            _, std = self.gp_belief.predict(grid_points, return_std=True)
            mean_variance = np.mean(std ** 2)
            self.history['gp_variance'].append(mean_variance)
            
            if self.verbose:
                print(f"    Mean GP variance: {mean_variance:.4f}")
        
        # Compile results
        results = self._compile_results()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"MISSION COMPLETE")
            print(f"{'='*70}")
            print(f"Total cycles: {self.cycle_count}")
            print(f"Total samples: {self.history['samples_collected']}")
            print(f"Final GP variance: {self.history['gp_variance'][-1]:.4f}")
        
        return results
    
    def _greedy_assignment(
        self,
        robots: List[Robot],
        candidate_sets_dict: Dict[int, 'CandidateSet']
    ) -> Dict[int, List[np.ndarray]]:
        """
        Simple greedy assignment with kriging believer.
        
        Each robot greedily selects top-k candidates by acquisition function.
        Uses a temporary believer GP to avoid conflicts.
        """
        assignments = {robot.id: [] for robot in robots}
        believer_gp = deepcopy(self.gp_belief)
        
        # Assign targets to each robot
        for robot in robots:
            if robot.id not in candidate_sets_dict:
                continue
                
            candidate_set = candidate_sets_dict[robot.id]
            feasible = candidate_set.get_feasible_points()
            
            if len(feasible) == 0:
                continue
            
            # Calculate acquisition values (use variance as proxy for information gain)
            _, std = believer_gp.predict(feasible, return_std=True)
            acquisition_values = std ** 2  # Variance
            
            # Penalize by distance
            distances = np.linalg.norm(feasible - robot.position, axis=1)
            acquisition_values = acquisition_values - 0.001 * distances
            
            # Select top candidates (up to budget)
            sorted_indices = np.argsort(acquisition_values)[::-1]
            
            selected_targets = []
            remaining_budget = robot.remaining_budget
            current_pos = robot.position.copy()
            
            for idx in sorted_indices:
                target = feasible[idx]
                
                # Check if reachable
                distance = np.linalg.norm(target - current_pos)
                time_needed = distance / robot.max_speed + self.sensor_time
                
                if time_needed <= remaining_budget:
                    selected_targets.append(target)
                    
                    # Update believer GP (kriging believer)
                    predicted_value, _ = believer_gp.predict(target.reshape(1, -1))
                    believer_gp.update(target.reshape(1, -1), predicted_value)
                    
                    # Update for next iteration
                    remaining_budget -= time_needed
                    current_pos = target.copy()
                    
                    # Limit to reasonable number
                    if len(selected_targets) >= 10:
                        break
            
            assignments[robot.id] = selected_targets
        
        return assignments
    
    def _get_grid_points(self, resolution: int = 50) -> np.ndarray:
        """Generate grid points for variance evaluation."""
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], resolution)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], resolution)
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.ravel(), Y.ravel()])
    
    def _compile_results(self) -> Dict:
        """Compile execution results."""
        return {
            'cycles': self.cycle_count,
            'samples_per_robot': self.samples_per_robot,
            'paths_per_robot': self.paths_per_robot,
            'total_samples': self.history['samples_collected'],
            'gp_variance_history': self.history['gp_variance'],
            'final_gp_belief': self.gp_belief,
            'robots': self.robots,
            'history': self.history
        }


def visualize_full_system_results(
    env,
    results: Dict,
    bounds: np.ndarray,
    save_path: str = 'results/full_system_demo.png'
):
    """
    Visualize complete system execution results.
    
    Shows:
    - Ground truth with all robot paths
    - GP belief evolution
    - Sample locations and values
    - Variance reduction over time
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })
    
    fig = plt.figure(figsize=(20, 13))
    
    # Get data
    gp = results['final_gp_belief']
    robots = results['robots']
    paths = results['paths_per_robot']
    samples = results['samples_per_robot']
    
    # Grid for predictions
    resolution = 100
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    true_values = env.evaluate(grid_points).reshape(X.shape)
    gp_mean, gp_std = gp.predict(grid_points, return_std=True)
    gp_mean = gp_mean.reshape(X.shape)
    gp_variance = (gp_std ** 2).reshape(X.shape)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    
    # === ROW 1: 3D Surfaces ===
    
    # 1. Ground Truth with Paths
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, true_values, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
    
    # Add robot paths
    for robot_id, path in paths.items():
        if len(path) > 1:
            path_array = np.array(path)
            path_values = env.evaluate(path_array)
            ax1.plot(path_array[:, 0], path_array[:, 1], path_values,
                    c=colors[robot_id], linewidth=3, alpha=0.8, label=f'Robot {robot_id}')
    
    ax1.set_xlabel('X', labelpad=8)
    ax1.set_ylabel('Y', labelpad=8)
    ax1.set_zlabel('Value', labelpad=8)
    ax1.set_title('Ground Truth + Robot Paths', fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=220)
    ax1.legend()
    
    # 2. GP Mean
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, gp_mean, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    
    # Add sample points
    all_samples = []
    for robot_samples in samples.values():
        all_samples.extend(robot_samples)
    
    if len(all_samples) > 0:
        sample_pos = np.array([s[0] for s in all_samples])
        gp_sample_mean, _ = gp.predict(sample_pos)
        ax2.scatter(sample_pos[:, 0], sample_pos[:, 1], gp_sample_mean,
                   c='red', s=30, marker='o', edgecolors='darkred', linewidths=1, zorder=10)
    
    ax2.set_xlabel('X', labelpad=8)
    ax2.set_ylabel('Y', labelpad=8)
    ax2.set_zlabel('Value', labelpad=8)
    ax2.set_title('GP Mean (Final)', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=220)
    
    # 3. GP Variance
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot_surface(X, Y, gp_variance, cmap='YlOrRd', alpha=0.9, linewidth=0, antialiased=True)
    ax3.set_xlabel('X', labelpad=8)
    ax3.set_ylabel('Y', labelpad=8)
    ax3.set_zlabel('Variance', labelpad=8)
    ax3.set_title('GP Uncertainty (Final)', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=220)
    
    # === ROW 2: 2D Analysis ===
    
    # 4. All Robot Trajectories
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.contourf(X, Y, gp_variance, levels=20, cmap='YlOrRd', alpha=0.4)
    
    for robot_id, path in paths.items():
        if len(path) > 1:
            path_array = np.array(path)
            ax4.plot(path_array[:, 0], path_array[:, 1], c=colors[robot_id],
                    linewidth=2, alpha=0.8, label=f'Robot {robot_id}')
            ax4.scatter(path_array[0, 0], path_array[0, 1], s=150,
                       c=colors[robot_id], marker='o', edgecolors='black', linewidths=2, zorder=10)
            
            # Sample points for this robot
            robot_samples = samples[robot_id]
            if len(robot_samples) > 0:
                sample_pos = np.array([s[0] for s in robot_samples])
                ax4.scatter(sample_pos[:, 0], sample_pos[:, 1], s=50,
                           c=colors[robot_id], marker='X', edgecolors='white', linewidths=1.5, alpha=0.9)
    
    ax4.set_xlim(bounds[0, 0], bounds[0, 1])
    ax4.set_ylim(bounds[1, 0], bounds[1, 1])
    ax4.set_title('All Robot Trajectories', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Variance Reduction Over Time
    ax5 = fig.add_subplot(2, 3, 5)
    variance_history = results['gp_variance_history']
    cycles = np.arange(1, len(variance_history) + 1)
    ax5.plot(cycles, variance_history, 'b-', linewidth=2, marker='o', markersize=6)
    ax5.set_xlabel('Cycle')
    ax5.set_ylabel('Mean GP Variance')
    ax5.set_title('Uncertainty Reduction Over Time', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = "MISSION STATISTICS\n" + "="*40 + "\n\n"
    stats_text += f"Total Cycles:     {results['cycles']}\n"
    stats_text += f"Total Samples:    {results['total_samples']}\n"
    stats_text += f"Total Distance:   {sum(len(p)-1 for p in paths.values()) * 10:.1f} m (approx)\n\n"
    
    stats_text += "PER-ROBOT RESULTS\n" + "-"*40 + "\n"
    for robot in robots:
        robot_samples = samples[robot.id]
        robot_path_length = len(paths[robot.id]) - 1
        budget_used = robot.initial_budget - robot.remaining_budget
        
        stats_text += f"\nRobot {robot.id}:\n"
        stats_text += f"  Samples:        {len(robot_samples)}\n"
        stats_text += f"  Path length:    {robot_path_length} segments\n"
        stats_text += f"  Budget used:    {budget_used:.1f}s / {robot.initial_budget:.1f}s\n"
        stats_text += f"  Budget remain:  {robot.remaining_budget:.1f}s\n"
        
        if len(robot_samples) > 0:
            sample_vals = [s[1] for s in robot_samples]
            stats_text += f"  Value range:    [{min(sample_vals):.3f}, {max(sample_vals):.3f}]\n"
    
    stats_text += "\n" + "GP BELIEF\n" + "-"*40 + "\n"
    stats_text += f"Training samples: {len(gp.X_train)}\n"
    stats_text += f"Initial variance: {variance_history[0]:.4f}\n"
    stats_text += f"Final variance:   {variance_history[-1]:.4f}\n"
    stats_text += f"Variance reduced: {(1 - variance_history[-1]/variance_history[0])*100:.1f}%\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   Saved to: {save_path}")
    plt.show()


def demo_full_system(
    env_name: str = 'peaks',
    n_robots: int = 3,
    n_init: int = 5,
    robot_budget: float = 300.0,
    execution_step: int = 1,
    max_cycles: int = 50,
    seed: int = 42
):
    """
    Run complete full-system demo.
    
    Args:
        env_name: Environment function name
        n_robots: Number of robots
        n_init: Initial GP samples
        robot_budget: Budget per robot (in SECONDS for TIME budget)
        execution_step: Waypoints to execute before replanning
        max_cycles: Maximum planning cycles
        seed: Random seed
    """
    np.random.seed(seed)
    
    print(f"{'='*70}")
    print(f"FULL SYSTEM IPP DEMO")
    print(f"{'='*70}\n")
    
    # Setup - medium search area: 1km × 1km
    bounds = np.array([[0, 100], [0, 100]])  # Coordinate bounds
    physical_scale = 10.0  # Each coordinate unit = 10 meters
    robot_speed_ms = 5.0  # 5 m/s = 18 km/h (fast ground robot)
    
    print(f"Configuration:")
    print(f"  Environment: {env_name}")
    print(f"  Robots: {n_robots}")
    print(f"  Execution step: {execution_step} waypoint(s)")
    print(f"  Max cycles: {max_cycles}")
    
    # Create environment with physical scale
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name=env_name,
        observation_noise=0.1,
        physical_scale=physical_scale,
        seed=seed
    )
    
    # Print environment information
    print_environment_info(
        env_name=env_name,
        bounds=bounds,
        physical_scale=physical_scale,
        robot_speed=robot_speed_ms,
        robot_budget=robot_budget,
        budget_type='time'
    )
    
    # Initial GP
    print(f"Initializing GP with {n_init} random samples...")
    init_points = np.random.uniform(
        [bounds[0, 0], bounds[1, 0]],
        [bounds[0, 1], bounds[1, 1]],
        size=(n_init, 2)
    )
    init_values = env.evaluate(init_points)
    gp = create_gp_belief(
        bounds=bounds,
        kernel_type='matern',
        length_scale=15.0,
        variance=1.0,
        noise=0.1
    )
    gp.update(init_points, init_values)
    
    # Create robots with physical parameters
    print(f"\nCreating {n_robots} robots...")
    robots = []
    for i in range(n_robots):
        # Random starting position in coordinate system
        start_pos = np.random.uniform([20, 20], [80, 80])
        
        robot = Robot(
            robot_id=i,
            initial_position=start_pos,
            budget_type=BudgetType.TIME,
            initial_budget=robot_budget,  # seconds
            max_speed=robot_speed_ms,     # m/s
            environment=env               # Link to environment for scaling
        )
        robots.append(robot)
        
        # Convert position to physical coordinates for display
        phys_pos = start_pos * physical_scale
        max_distance = robot_budget * robot_speed_ms
        print(f"  Robot {i}: position ({phys_pos[0]:.0f}m, {phys_pos[1]:.0f}m), "
              f"budget {robot_budget:.0f}s, max distance {max_distance:.0f}m")
    
    # Create controller
    print(f"\nInitializing Receding Horizon Controller...")
    controller = RecedingHorizonIPP(
        env=env,
        robots=robots,
        gp_belief=gp,
        bounds=bounds,
        sensor_time=1.0,
        execution_step=execution_step,
        verbose=True
    )
    
    # Run mission
    results = controller.run(max_cycles=max_cycles)
    
    # Visualize
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATION")
    print(f"{'='*70}")
    visualize_full_system_results(env, results, bounds)
    
    print(f"\n{'='*70}")
    print(f"DEMO COMPLETE")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    # Run with default settings
    # Robot budget: 300 seconds @ 5 m/s = 1500m max distance in 1km×1km area
    results = demo_full_system(
        env_name='peaks',
        n_robots=3,
        n_init=5,
        robot_budget=300.0,  # 300 seconds (5 minutes)
        execution_step=1,
        max_cycles=50,
        seed=42
    )
