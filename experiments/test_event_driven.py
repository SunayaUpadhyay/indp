"""Quick test of event-driven execution."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import create_environment
from src.core.belief import create_gp_belief
from src.baselines import RandomMultiRobotPlanner
from experimental_config import *


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

# Create 2 robots
robots = []
for i in range(2):
    robot = Robot(
        robot_id=i,
        initial_position=np.array([0.0, 0.0]),
        budget_type=BudgetType.TIME,
        initial_budget=30.0,  # Short test
        max_speed=2.0,
        sensor_range=5.0,
        environment=env
    )
    robots.append(robot)

# Create planner
planner = RandomMultiRobotPlanner(
    robots=robots,
    environment=env,
    gp_belief=gp,
    config={
        'step_size': 15.0,
        'seed': 42,
        'sensor_time': 5.0  # Add sensor time config
    }
)

print("Testing Random Planner with Event-Driven Execution")
print("="*70)

# Execute mission with verbose output
results = planner.execute_mission(max_iterations=100, verbose=True)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total measurements: {results['total_measurements']}")
print(f"Events processed: {results['stats']['events_processed']}")
print(f"Robot 0: {results['stats']['measurements_taken'][0]} measurements")
print(f"Robot 1: {results['stats']['measurements_taken'][1]} measurements")
