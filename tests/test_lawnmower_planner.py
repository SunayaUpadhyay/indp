"""
Test suite for Lawnmower (Coverage) Planner.

Validates:
- Vertical strip partitioning
- Lawnmower path generation
- Budget compliance
- Coverage patterns
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import SyntheticEnvironment
from src.baselines.lawnmower_planner import LawnmowerPlanner


def test_lawnmower_basic():
    """Test basic lawnmower functionality with 2 robots."""
    print("\n" + "="*70)
    print("TEST: Basic Lawnmower Planner (2 robots)")
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
    
    # Create 2 robots at origin with TIME budget
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=300.0,  # 300 seconds
            max_speed=2.0,  # 2 m/s
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create planner
    config = {
        'stripe_width': 10.0,
        'orientation': 'vertical'
    }
    planner = LawnmowerPlanner(robots=robots, environment=env, config=config)
    
    print(f"\nPlanner configuration:")
    info = planner.get_planner_info()
    print(f"  Orientation: {info['orientation']}")
    print(f"  Stripe width: {info['stripe_width']}m")
    print(f"  Robots: {info['n_robots']}")
    print(f"  Waypoints per robot: {info['total_waypoints']}")
    
    # Execute mission
    print(f"\nExecuting mission...")
    results = planner.execute_mission(max_iterations=200, verbose=False)
    
    # Verify results
    print(f"\n{'Results':-^70}")
    print(f"Total iterations: {results['stats']['iterations']}")
    print(f"Total measurements: {results['total_measurements']}")
    print(f"Total distance: {results['total_distance']:.1f}m")
    
    # Check that robots covered different strips
    for i, robot in enumerate(robots):
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            positions = np.array([m[0] for m in measurements])
            x_coords = positions[:, 0]
            
            print(f"\nRobot {robot.id}:")
            print(f"  Measurements: {len(measurements)}")
            print(f"  X range: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
            print(f"  Distance: {results['stats']['total_distance'][robot.id]:.1f}m")
            print(f"  Budget used: {robot.initial_budget - robot.remaining_budget:.1f}s")
            
            # Verify robot stayed in its strip
            expected_x_min = i * 50.0
            expected_x_max = (i + 1) * 50.0
            assert x_coords.min() >= expected_x_min - 1.0, \
                f"Robot {i} x_min {x_coords.min()} below strip boundary {expected_x_min}"
            assert x_coords.max() <= expected_x_max + 1.0, \
                f"Robot {i} x_max {x_coords.max()} above strip boundary {expected_x_max}"
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_lawnmower_4_robots():
    """Test lawnmower with 4 robots."""
    print("\n" + "="*70)
    print("TEST: Lawnmower Planner with 4 robots")
    print("="*70)
    
    # Create environment
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,
        n_components=5
    )
    
    # Create 4 robots at origin
    robots = []
    for i in range(4):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=200.0,  # 200 seconds
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    # Create planner
    config = {'stripe_width': 10.0, 'orientation': 'vertical'}
    planner = LawnmowerPlanner(robots=robots, environment=env, config=config)
    
    # Execute mission
    results = planner.execute_mission(max_iterations=200, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.1f}m")
    
    # Verify strip separation
    print(f"\nStrip verification:")
    for i, robot in enumerate(robots):
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            positions = np.array([m[0] for m in measurements])
            x_coords = positions[:, 0]
            
            expected_x_center = (i + 0.5) * 25.0  # Each strip is 25m wide (100/4)
            actual_x_center = (x_coords.min() + x_coords.max()) / 2
            
            print(f"  Robot {i}: X center {actual_x_center:.1f}m (expected ~{expected_x_center:.1f}m)")
            
            # Check strip boundaries (with some tolerance)
            expected_x_min = i * 25.0
            expected_x_max = (i + 1) * 25.0
            assert x_coords.min() >= expected_x_min - 2.0
            assert x_coords.max() <= expected_x_max + 2.0
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_lawnmower_horizontal():
    """Test horizontal orientation."""
    print("\n" + "="*70)
    print("TEST: Horizontal Lawnmower")
    print("="*70)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0
    )
    
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=200.0,
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    config = {'stripe_width': 10.0, 'orientation': 'horizontal'}
    planner = LawnmowerPlanner(robots=robots, environment=env, config=config)
    
    results = planner.execute_mission(max_iterations=150, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    
    # Verify horizontal strips (partitioned in Y)
    for i, robot in enumerate(robots):
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            positions = np.array([m[0] for m in measurements])
            y_coords = positions[:, 1]
            
            expected_y_min = i * 50.0
            expected_y_max = (i + 1) * 50.0
            
            print(f"  Robot {i}: Y range [{y_coords.min():.1f}, {y_coords.max():.1f}]")
            
            # Check horizontal strip boundaries
            assert y_coords.min() >= expected_y_min - 2.0
            assert y_coords.max() <= expected_y_max + 2.0
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


def test_lawnmower_budget_exhaustion():
    """Test that planner respects budget limits."""
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
    
    # Short budget
    robots = []
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=50.0,  # Only 50 seconds
            max_speed=2.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    config = {'stripe_width': 10.0, 'orientation': 'vertical'}
    planner = LawnmowerPlanner(robots=robots, environment=env, config=config)
    
    results = planner.execute_mission(max_iterations=100, verbose=False)
    
    print(f"\nResults with tight budget:")
    for robot in robots:
        distance = results['stats']['total_distance'][robot.id]
        budget_used = robot.initial_budget - robot.remaining_budget
        max_possible = robot.initial_budget * robot.max_speed
        
        print(f"  Robot {robot.id}:")
        print(f"    Distance: {distance:.1f}m")
        print(f"    Budget used: {budget_used:.1f}s / {robot.initial_budget}s")
        print(f"    Max possible: {max_possible:.1f}m")
        
        # Verify budget not exceeded
        assert distance <= max_possible + 0.1, "Distance exceeds budget!"
        assert budget_used <= robot.initial_budget + 0.1, "Budget exceeded!"
    
    print(f"\n{'✓ Test passed!':-^70}")
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LAWNMOWER PLANNER TEST SUITE")
    print("="*70)
    
    tests = [
        test_lawnmower_basic,
        test_lawnmower_4_robots,
        test_lawnmower_horizontal,
        test_lawnmower_budget_exhaustion
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
