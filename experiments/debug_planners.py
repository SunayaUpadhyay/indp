"""
Debug script to trace planner behavior step-by-step.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import create_environment
from src.core.belief import create_gp_belief
from src.baselines import AuctionVariancePlanner, SequentialGreedyIGPlanner
from experimental_config import *


def debug_auction_planner():
    """Debug Auction planner step by step."""
    print("="*70)
    print("DEBUGGING AUCTION PLANNER")
    print("="*70)
    
    # Create environment
    env = create_environment(
        bounds=BOUNDS,
        env_type='synthetic',
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0
    )
    
    # Create GP
    gp = create_gp_belief(
        bounds=BOUNDS,
        kernel_type='matern',
        length_scale=15.0,
        variance=1.0,
        noise=0.1
    )
    
    # Add initial samples
    init_points = np.array([[25, 25], [75, 75]])
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    # Create 2 robots
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=150.0,
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create planner
    planner = AuctionVariancePlanner(
        robots=robots,
        environment=env,
        gp_belief=gp,
        config={
            'num_candidates': 50,
            'replan_interval': 5,
            'grid_resolution': 50
        }
    )
    
    # Run first 10 iterations with detailed logging
    for iteration in range(10):
        print(f"\n--- Iteration {iteration} ---")
        print(f"steps_since_auction: {planner.steps_since_auction}")
        
        # Show robot positions and budgets
        for robot in robots:
            print(f"  Robot {robot.id}: pos={robot.position}, budget={robot.remaining_budget:.1f}s")
        
        # Show current assignments
        print(f"  Assignments: {planner.assignments}")
        
        # Plan step
        waypoints = planner.plan_step()
        print(f"  Waypoints returned: {waypoints}")
        
        if not waypoints:
            print("  WARNING: No waypoints returned!")
            break
        
        # Execute movements
        for robot_id, waypoint in waypoints.items():
            robot = robots[robot_id]
            if robot.is_active and robot.can_reach(waypoint):
                distance = robot.move_to(waypoint, timestamp=iteration, update_budget=True)
                observation = env.observe(waypoint.reshape(1, -1))[0]
                robot.add_measurement(waypoint, observation, iteration)
                gp.update(waypoint.reshape(1, -1), np.array([observation]))
                print(f"  Robot {robot_id} moved {distance:.2f}m to {waypoint}, budget now {robot.remaining_budget:.1f}s")
            else:
                print(f"  Robot {robot_id} CANNOT reach waypoint {waypoint}")


def debug_sequential_greedy():
    """Debug SequentialGreedy planner step by step."""
    print("\n\n")
    print("="*70)
    print("DEBUGGING SEQUENTIAL GREEDY PLANNER")
    print("="*70)
    
    # Create environment
    env = create_environment(
        bounds=BOUNDS,
        env_type='synthetic',
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0
    )
    
    # Create GP
    gp = create_gp_belief(
        bounds=BOUNDS,
        kernel_type='matern',
        length_scale=15.0,
        variance=1.0,
        noise=0.1
    )
    
    # Add initial samples
    init_points = np.array([[25, 25], [75, 75]])
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    # Create 2 robots
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=150.0,
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create planner
    planner = SequentialGreedyIGPlanner(
        robots=robots,
        environment=env,
        gp_belief=gp,
        config={
            'candidate_resolution': 10,
            'seed': 42
        }
    )
    
    # Run first 10 iterations with detailed logging
    for iteration in range(10):
        print(f"\n--- Iteration {iteration} ---")
        
        # Show robot positions and budgets
        for robot in robots:
            print(f"  Robot {robot.id}: pos={robot.position}, budget={robot.remaining_budget:.1f}s")
        
        # Plan step
        waypoints = planner.plan_step()
        print(f"  Waypoints returned: {waypoints}")
        
        if not waypoints:
            print("  WARNING: No waypoints returned!")
            break
        
        # Execute movements
        for robot_id, waypoint in waypoints.items():
            robot = robots[robot_id]
            if robot.is_active and robot.can_reach(waypoint):
                old_pos = robot.position.copy()
                distance = robot.move_to(waypoint, timestamp=iteration, update_budget=True)
                observation = env.observe(waypoint.reshape(1, -1))[0]
                robot.add_measurement(waypoint, observation, iteration)
                gp.update(waypoint.reshape(1, -1), np.array([observation]))
                print(f"  Robot {robot_id} moved from {old_pos} to {robot.position}, "
                      f"distance={distance:.2f}m, budget now {robot.remaining_budget:.1f}s")
            else:
                print(f"  Robot {robot_id} CANNOT reach waypoint {waypoint}")


if __name__ == '__main__':
    debug_auction_planner()
    debug_sequential_greedy()
