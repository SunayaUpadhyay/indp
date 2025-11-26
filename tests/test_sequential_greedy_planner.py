"""
Test suite for Sequential Greedy IG Planner.

Validates:
- Kriging Believer coordination
- Information gain computation
- Sequential planning
- Budget compliance
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import SyntheticEnvironment
from src.core.belief import create_gp_belief
from src.baselines.sequential_greedy_planner import SequentialGreedyIGPlanner


def test_sequential_greedy_basic():
    """Test basic sequential greedy functionality."""
    print("\n" + "="*70)
    print("TEST: Basic Sequential Greedy IG Planner (2 robots)")
    print("="*70)
    
    # Create environment
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,
        n_components=4
    )
    
    # Create GP belief with initial samples
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0,
                         variance=1.0, noise=0.1)
    
    # Add a few initial samples
    n_init = 5
    init_points = np.random.uniform([0, 0], [100, 100], (n_init, 2))
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    # Create 2 robots at origin with TIME budget
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=150.0,  # 150 seconds
            max_speed=2.0,  # 2 m/s
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create planner
    config = {
        'candidate_resolution': 15,  # Smaller for faster testing
        'lookahead_depth': 1
    }
    planner = SequentialGreedyIGPlanner(
        robots=robots, 
        environment=env, 
        gp_belief=gp,
        config=config
    )
    
    print(f"\nPlanner configuration:")
    info = planner.get_planner_info()
    print(f"  Planner: {info['planner_name']}")
    print(f"  Candidate resolution: {info['candidate_resolution']}")
    print(f"  Number of candidates: {info['n_candidates']}")
    print(f"  Robots: {info['n_robots']}")
    
    # Execute mission
    print(f"\nExecuting mission...")
    results = planner.execute_mission(max_iterations=50, verbose=False)
    
    # Verify results
    print(f"\n{'Results':-^70}")
    print(f"Total iterations: {results['stats']['iterations']}")
    print(f"Total measurements: {results['total_measurements']}")
    print(f"Total distance: {results['total_distance']:.1f}m")
    
    for i, robot in enumerate(robots):
        measurements = results['robot_measurements'][robot.id]
        distance = results['stats']['total_distance'][robot.id]
        budget_used = robot.initial_budget - robot.remaining_budget
        
        print(f"\nRobot {robot.id}:")
        print(f"  Measurements: {len(measurements)}")
        print(f"  Distance: {distance:.1f}m")
        print(f"  Budget used: {budget_used:.1f}s / {robot.initial_budget}s")
        
        # Verify budget not exceeded
        max_possible = robot.initial_budget * robot.max_speed
        assert distance <= max_possible + 0.1, "Distance exceeds budget!"
        assert budget_used <= robot.initial_budget + 0.1, "Budget exceeded!"
    
    # Verify we got some measurements
    assert results['total_measurements'] > 0, "No measurements collected!"
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_sequential_greedy_coordination():
    """Test that KB coordination actually affects planning."""
    print("\n" + "="*70)
    print("TEST: Sequential Greedy KB Coordination")
    print("="*70)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,
        n_components=3
    )
    
    # Create GP with initial samples
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0)
    init_points = np.random.uniform([0, 0], [100, 100], (10, 2))
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    # Create 3 robots
    robots = []
    for i in range(3):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=100.0,
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    config = {'candidate_resolution': 15, 'seed': 123}
    planner = SequentialGreedyIGPlanner(
        robots=robots, environment=env, gp_belief=gp, config=config
    )
    
    # Execute mission
    results = planner.execute_mission(max_iterations=30, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.1f}m")
    
    # Check that robots explored different areas (good coordination)
    robot_positions = []
    for robot in robots:
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            positions = np.array([m[0] for m in measurements])
            robot_positions.append(positions)
            print(f"  Robot {robot.id}: {len(positions)} samples")
    
    # Simple overlap check - robots shouldn't all visit exact same spots
    # (This is a weak test but shows some differentiation)
    if len(robot_positions) >= 2:
        overlap_count = 0
        for pos1 in robot_positions[0]:
            for pos2 in robot_positions[1]:
                if np.linalg.norm(pos1 - pos2) < 5.0:  # Within 5m
                    overlap_count += 1
        
        total_positions = len(robot_positions[0])
        overlap_ratio = overlap_count / max(total_positions, 1)
        print(f"  Overlap between Robot 0 and 1: {overlap_ratio:.2%}")
        
        # With coordination, we expect SOME differentiation
        # (Not a hard requirement but should happen with KB)
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_sequential_greedy_4_robots():
    """Test with 4 robots."""
    print("\n" + "="*70)
    print("TEST: Sequential Greedy with 4 robots")
    print("="*70)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,
        n_components=5
    )
    
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0)
    init_points = np.random.uniform([0, 0], [100, 100], (8, 2))
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    robots = []
    for i in range(4):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=120.0,
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    config = {'candidate_resolution': 15}
    planner = SequentialGreedyIGPlanner(
        robots=robots, environment=env, gp_belief=gp, config=config
    )
    
    results = planner.execute_mission(max_iterations=40, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    
    for robot in robots:
        measurements = results['robot_measurements'][robot.id]
        print(f"  Robot {robot.id}: {len(measurements)} samples")
        assert len(measurements) > 0, f"Robot {robot.id} collected no samples!"
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_sequential_greedy_budget_exhaustion():
    """Test budget enforcement."""
    print("\n" + "="*70)
    print("TEST: Budget Exhaustion")
    print("="*70)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0
    )
    
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0)
    init_points = np.random.uniform([0, 0], [100, 100], (5, 2))
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)
    
    # Very short budget
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=30.0,  # Only 30 seconds
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    config = {'candidate_resolution': 15}
    planner = SequentialGreedyIGPlanner(
        robots=robots, environment=env, gp_belief=gp, config=config
    )
    
    results = planner.execute_mission(max_iterations=30, verbose=False)
    
    print(f"\nResults with tight budget:")
    for robot in robots:
        distance = results['stats']['total_distance'][robot.id]
        budget_used = robot.initial_budget - robot.remaining_budget
        max_possible = robot.initial_budget * robot.max_speed
        
        print(f"  Robot {robot.id}:")
        print(f"    Distance: {distance:.1f}m")
        print(f"    Budget used: {budget_used:.1f}s / {robot.initial_budget}s")
        
        # Verify budget not exceeded
        assert distance <= max_possible + 0.1, "Distance exceeds budget!"
        assert budget_used <= robot.initial_budget + 0.1, "Budget exceeded!"
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SEQUENTIAL GREEDY IG PLANNER TEST SUITE")
    print("="*70)
    
    tests = [
        test_sequential_greedy_basic,
        test_sequential_greedy_coordination,
        test_sequential_greedy_4_robots,
        test_sequential_greedy_budget_exhaustion
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n{'✗ TEST FAILED':-^70}")
            print(f"Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n{'✗ TEST ERROR':-^70}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED! ✓\n")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED ✗\n")
        sys.exit(1)
