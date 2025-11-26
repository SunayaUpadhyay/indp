"""
Experiment Runner for IPP Baseline Comparisons.

Runs all baseline planners across different scenarios and collects comprehensive metrics.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import create_environment
from src.core.belief import create_gp_belief

from src.baselines import (
    RandomMultiRobotPlanner,
    LawnmowerPlanner,
    SequentialGreedyIGPlanner,
    IndependentGreedyIGPlanner,
    AuctionVariancePlanner
)

from metrics import (
    compute_rmse,
    compute_integrated_variance,
    compute_coverage_metrics,
    compute_overlap_metrics,
    compute_time_to_first_hotspot,
    compute_hotspot_recall,
    compute_probability_mass_covered,
    compute_redundant_hotspot_coverage
)

from experimental_config import *


class ExperimentRunner:
    """
    Manages execution of baseline planner experiments.
    """
    
    def __init__(
        self,
        output_dir: str = 'results/experiments',
        use_drones: bool = False
    ):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory for saving results
            use_drones: If True, use drone parameters instead of ground robots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get robot configuration
        self.robot_config = get_robot_config(use_drones=use_drones)
        
        # Generate evaluation grid for metrics
        self.eval_grid = self._generate_eval_grid()
        
        # Results storage
        self.results = []
    
    def _generate_eval_grid(self) -> np.ndarray:
        """Generate grid for RMSE and variance evaluation."""
        x = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], GRID_RESOLUTION)
        y = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], GRID_RESOLUTION)
        X, Y = np.meshgrid(x, y)
        return np.c_[X.ravel(), Y.ravel()]
    
    def create_planner(
        self,
        planner_name: str,
        robots: List[Robot],
        environment: Any,
        gp_belief: Optional[Any] = None
    ):
        """
        Create planner instance by name.
        
        Args:
            planner_name: Name of planner
            robots: List of robots
            environment: Environment instance
            gp_belief: GP belief (required for some planners)
            
        Returns:
            Planner instance
        """
        if planner_name == 'Random':
            return RandomMultiRobotPlanner(
                robots=robots,
                environment=environment,
                gp_belief=gp_belief,
                config=RANDOM_PLANNER_CONFIG
            )
        
        elif planner_name == 'Lawnmower':
            return LawnmowerPlanner(
                robots=robots,
                environment=environment,
                gp_belief=gp_belief,
                config=LAWNMOWER_CONFIG
            )
        
        elif planner_name == 'SequentialGreedy':
            if gp_belief is None:
                raise ValueError("SequentialGreedy requires GP belief!")
            return SequentialGreedyIGPlanner(
                robots=robots,
                environment=environment,
                gp_belief=gp_belief,
                config=SEQUENTIAL_GREEDY_CONFIG
            )
        
        elif planner_name == 'IndependentGreedy':
            if gp_belief is None:
                raise ValueError("IndependentGreedy requires GP belief!")
            return IndependentGreedyIGPlanner(
                robots=robots,
                environment=environment,
                gp_belief=gp_belief,
                config=INDEPENDENT_GREEDY_CONFIG
            )
        
        elif planner_name == 'Auction':
            if gp_belief is None:
                raise ValueError("Auction requires GP belief!")
            return AuctionVariancePlanner(
                robots=robots,
                environment=environment,
                gp_belief=gp_belief,
                config=AUCTION_CONFIG
            )
        
        else:
            raise ValueError(f"Unknown planner: {planner_name}")
    
    def run_single_experiment(
        self,
        planner_name: str,
        scenario_type: str,
        env_function: str,
        num_robots: int,
        budget: float,
        env_seed: int,
        run_seed: int,
        max_iterations: int = 200,
        **env_kwargs
    ) -> Dict[str, Any]:
        """
        Run a single experiment configuration.
        
        Args:
            planner_name: Name of planner to test
            scenario_type: Type of scenario (gaussian_hotspot, smooth_mapping, etc.)
            env_function: Environment function name
            num_robots: Number of robots
            budget: Time budget per robot
            env_seed: Random seed for environment
            run_seed: Random seed for this run
            max_iterations: Max planning iterations
            **env_kwargs: Additional environment parameters
            
        Returns:
            Dictionary with all metrics and metadata
        """
        print(f"\n{'='*70}")
        print(f"Running: {planner_name} | {scenario_type} | {env_function}")
        print(f"  Robots: {num_robots} | Budget: {budget}s | Seeds: env={env_seed}, run={run_seed}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Set random seed
        np.random.seed(run_seed)
        
        # Create environment
        env = create_environment(
            bounds=BOUNDS,
            env_type='synthetic',
            function_name=env_function,
            observation_noise=OBSERVATION_NOISE,
            seed=env_seed,
            physical_scale=PHYSICAL_SCALE,
            **env_kwargs
        )
        
        # Create GP belief with initial samples
        gp = create_gp_belief(
            bounds=BOUNDS,
            kernel_type='matern',
            length_scale=15.0,
            variance=1.0,
            noise=OBSERVATION_NOISE
        )
        
        # Add a few initial random samples
        n_init = 5
        init_points = np.random.uniform(
            [BOUNDS[0, 0], BOUNDS[1, 0]],
            [BOUNDS[0, 1], BOUNDS[1, 1]],
            (n_init, 2)
        )
        init_values = env.evaluate(init_points)
        gp.update(init_points, init_values)
        
        # Create robots
        robots = []
        for i in range(num_robots):
            robot = Robot(
                robot_id=i,
                initial_position=START_POSITION.copy(),
                budget_type=BUDGET_TYPE,
                initial_budget=budget,
                max_speed=self.robot_config['max_speed'],
                sensor_range=self.robot_config['sensor_range'],
                environment=env
            )
            robots.append(robot)
        
        # Create planner
        planner = self.create_planner(planner_name, robots, env, gp)
        
        # Execute mission
        mission_results = planner.execute_mission(
            max_iterations=max_iterations,
            verbose=False
        )
        
        # Collect all visited positions
        all_positions = []
        robot_positions_dict = {}
        
        for robot in robots:
            measurements = mission_results['robot_measurements'][robot.id]
            if measurements:
                positions = np.array([m[0] for m in measurements])
                all_positions.extend(positions)
                robot_positions_dict[robot.id] = positions
        
        all_positions = np.array(all_positions) if all_positions else np.array([]).reshape(0, 2)
        
        # Compute metrics
        final_rmse = compute_rmse(env, planner.gp_belief, self.eval_grid)
        final_variance = compute_integrated_variance(planner.gp_belief, self.eval_grid)
        
        coverage_metrics = compute_coverage_metrics(all_positions, BOUNDS)
        overlap_metrics = compute_overlap_metrics(robot_positions_dict)
        
        # SAR-specific metrics (if applicable)
        sar_metrics = {}
        if scenario_type == 'gaussian_hotspot':
            # Time to first hotspot (for each robot, take minimum)
            times_to_hotspot = []
            for robot in robots:
                trajectory = mission_results['robot_trajectories'][robot.id]
                if len(trajectory) > 0:
                    t = compute_time_to_first_hotspot(
                        list(trajectory), env, self.eval_grid, HOTSPOT_PERCENTILE
                    )
                    if t is not None:
                        times_to_hotspot.append(t)
            
            sar_metrics['time_to_first_hotspot'] = min(times_to_hotspot) if times_to_hotspot else None
            sar_metrics['mean_time_to_hotspot'] = np.mean(times_to_hotspot) if times_to_hotspot else None
            
            # Hotspot recall
            grid_shape = (GRID_RESOLUTION, GRID_RESOLUTION)
            recall_metrics = compute_hotspot_recall(
                all_positions, env, self.eval_grid, grid_shape,
                k_hotspots=NUM_HOTSPOTS_TO_TRACK
            )
            sar_metrics.update(recall_metrics)
            
            # Probability mass covered
            prob_mass = compute_probability_mass_covered(
                all_positions, env, self.eval_grid
            )
            sar_metrics['prob_mass_covered'] = prob_mass
            
            # Redundant coverage
            redundancy_metrics = compute_redundant_hotspot_coverage(
                robot_positions_dict, env, self.eval_grid, grid_shape,
                k_hotspots=NUM_HOTSPOTS_TO_TRACK
            )
            sar_metrics.update(redundancy_metrics)
        
        total_time = time.time() - start_time
        
        # Compile results
        result = {
            # Metadata
            'experiment_id': f"{planner_name}_{scenario_type}_{num_robots}r_{budget}s_{run_seed}",
            'timestamp': datetime.now().isoformat(),
            'scenario_type': scenario_type,
            'env_function': env_function,
            'env_seed': env_seed,
            'bounds': BOUNDS.tolist(),
            'noise_std': OBSERVATION_NOISE,
            
            # Robot configuration
            'num_robots': num_robots,
            'budget_type': 'TIME',
            'initial_budget': budget,
            'robot_max_speed': self.robot_config['max_speed'],
            'robot_sensor_range': self.robot_config['sensor_range'],
            'start_position': START_POSITION.tolist(),
            
            # Planner info
            'planner_name': planner_name,
            'planner_params': planner.get_planner_info(),
            'run_seed': run_seed,
            
            # Primary metrics
            'rmse_final': float(final_rmse),
            'integrated_variance_final': float(final_variance),
            'total_planning_time': float(total_time),
            'mean_planning_time_per_event': float(total_time / mission_results['stats']['events_processed']) if mission_results['stats']['events_processed'] > 0 else 0.0,
            
            # Coverage metrics
            'total_distance_per_robot': [
                float(mission_results['stats']['total_distance'][i]) for i in range(num_robots)
            ],
            'coverage_fraction': float(coverage_metrics['coverage_fraction']),
            'unique_cells_visited': int(coverage_metrics['unique_cells_visited']),
            'overlap_fraction': float(overlap_metrics['overlap_fraction']),
            
            # Mission stats
            'total_measurements': int(mission_results['total_measurements']),
            'total_distance': float(mission_results['total_distance']),
            'events_processed': int(mission_results['stats']['events_processed']),
            
            # SAR metrics
            **sar_metrics,
            
            # Per-robot stats
            'per_robot_measurements': [
                len(mission_results['robot_measurements'][i]) for i in range(num_robots)
            ],
            'per_robot_budget_used': [
                float(robots[i].initial_budget - robots[i].remaining_budget) for i in range(num_robots)
            ],
            
            # Trajectory data for visualization
            'robot_trajectories': {
                i: robot_positions_dict[i].tolist() if i in robot_positions_dict else []
                for i in range(num_robots)
            }
        }
        
        print(f"\n  RMSE: {final_rmse:.4f}")
        print(f"  Coverage: {coverage_metrics['coverage_fraction']:.2%}")
        print(f"  Total measurements: {mission_results['total_measurements']}")
        print(f"  Planning time: {total_time:.2f}s")
        
        if scenario_type == 'gaussian_hotspot':
            print(f"  Hotspot recall: {sar_metrics.get('hotspot_recall', 0):.2%}")
            print(f"  Prob mass covered: {sar_metrics.get('prob_mass_covered', 0):.2%}")
        
        return result
    
    def save_result(self, result: Dict[str, Any]):
        """Save result to JSON file."""
        filename = f"{result['experiment_id']}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  Saved: {filepath}")
    
    def run_scenario_suite(
        self,
        scenario_type: str,
        env_function: str,
        planners: List[str],
        num_robots_list: List[int],
        budgets: List[float],
        num_repetitions: int = 5,
        env_kwargs: Optional[Dict] = None,
        max_iterations: int = 200
    ):
        """
        Run a full suite of experiments for a scenario.
        
        Args:
            scenario_type: Type of scenario
            env_function: Environment function name
            planners: List of planner names to test
            num_robots_list: List of robot counts to test
            budgets: List of budgets to test
            num_repetitions: Number of repetitions per configuration
            env_kwargs: Additional environment parameters
            max_iterations: Max planning iterations
        """
        if env_kwargs is None:
            env_kwargs = {}
        
        total_experiments = (
            len(planners) * len(num_robots_list) * len(budgets) * num_repetitions
        )
        
        print(f"\n{'='*70}")
        print(f"SCENARIO SUITE: {scenario_type} ({env_function})")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Planners: {planners}")
        print(f"  Robots: {num_robots_list}")
        print(f"  Budgets: {budgets}")
        print(f"  Repetitions: {num_repetitions}")
        print(f"{'='*70}\n")
        
        experiment_count = 0
        
        for planner_name in planners:
            for num_robots in num_robots_list:
                for budget in budgets:
                    for rep in range(num_repetitions):
                        experiment_count += 1
                        
                        print(f"\nExperiment {experiment_count}/{total_experiments}")
                        
                        env_seed = SEED_START + rep
                        run_seed = SEED_START + 1000 + rep
                        
                        try:
                            result = self.run_single_experiment(
                                planner_name=planner_name,
                                scenario_type=scenario_type,
                                env_function=env_function,
                                num_robots=num_robots,
                                budget=budget,
                                env_seed=env_seed,
                                run_seed=run_seed,
                                max_iterations=max_iterations,
                                **env_kwargs
                            )
                            
                            self.save_result(result)
                            self.results.append(result)
                            
                        except Exception as e:
                            print(f"\n  ✗ ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
        
        print(f"\n{'='*70}")
        print(f"SUITE COMPLETE: {experiment_count} experiments run")
        print(f"{'='*70}\n")
