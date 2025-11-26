"""
Test script for Random Multi-Robot Planner baseline.

Verifies:
1. Robots stay within environment bounds
2. Budget constraints are respected
3. Multiple robots can operate independently
4. Basic statistics are tracked correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.robot import Robot, BudgetType
from src.core.environment import SyntheticEnvironment
from src.baselines.random_planner import RandomMultiRobotPlanner


def test_random_planner_basic():
    """Test basic functionality of random planner."""
    print("=" * 60)
    print("Test 1: Basic Random Planner Functionality")
    print("=" * 60)
    
    # Create simple gaussian mixture environment
    # Bounds in coordinate units, with physical_scale to convert to meters
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,  # 1 coordinate unit = 1 meter
        n_components=4,
        covs=10.0
    )
    
    print(f"\nEnvironment: gaussian_mixture, bounds {bounds.tolist()}")
    print(f"  Physical scale: 1.0 (1 coord unit = 1 meter)")
    print(f"  Components: 4, covariance: 10.0")
    
    # Create 2 robots with TIME budget, starting at SAME position
    robots = []
    start_pos = np.array([0.0, 0.0])  # All robots start at origin
    for i in range(2):
        robot = Robot(
            robot_id=i,
            initial_position=start_pos,
            budget_type=BudgetType.TIME,  # Use TIME budget
            initial_budget=40.0,  # 40 seconds
            max_speed=5.0,  # 5 m/s
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
        print(f"\nRobot {i}: start={start_pos}, budget=40s, max_speed=5.0m/s")
    
    # Create random planner
    config = {
        'step_size': 20.0,  # 20 meter steps
        'max_attempts': 100,
        'seed': 123
    }
    
    planner = RandomMultiRobotPlanner(
        robots=robots,
        environment=env,
        config=config
    )
    
    print(f"\nPlanner config: step_size=20m, max_attempts=100")
    
    # Execute mission
    print("\n" + "-" * 60)
    print("Executing mission...")
    print("-" * 60)
    
    results = planner.execute_mission(max_iterations=100, verbose=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Mission Complete - Results")
    print("=" * 60)
    
    print(f"\nTotal iterations: {results['stats']['iterations']}")
    print(f"Total measurements: {results['total_measurements']}")
    print(f"Total distance traveled: {results['total_distance']:.2f}m")
    
    for robot in robots:
        budget_used = robot.initial_budget - robot.remaining_budget
        # Calculate equivalent distance for time budget
        max_possible_distance = robot.initial_budget * robot.max_speed
        
        print(f"\nRobot {robot.id}:")
        print(f"  Measurements: {results['stats']['measurements_taken'][robot.id]}")
        print(f"  Distance: {results['stats']['total_distance'][robot.id]:.2f}m")
        print(f"  Time budget: {robot.initial_budget:.2f}s (max distance: {max_possible_distance:.2f}m)")
        print(f"  Remaining budget: {robot.remaining_budget:.2f}s")
        print(f"  Budget used: {budget_used:.2f}s")
        
        # Verify constraints
        trajectory = results['robot_trajectories'][robot.id]
        
        # Check bounds
        within_bounds = np.all(
            (trajectory >= bounds[:, 0]) & (trajectory <= bounds[:, 1])
        )
        print(f"  All waypoints within bounds: {within_bounds}")
        
        # Check budget
        budget_respected = robot.remaining_budget >= -0.01  # Allow small numerical error
        print(f"  Budget respected: {budget_respected}")
        
        assert within_bounds, f"Robot {robot.id} went out of bounds!"
        assert budget_respected, f"Robot {robot.id} exceeded budget!"
    
    print("\n✓ All constraint checks passed!")
    
    return results, robots, env


def test_random_planner_visualization():
    """Test and visualize random planner with multiple robots starting from corners."""
    print("\n" + "=" * 60)
    print("Test 2: Visualization Test - Corner Starts")
    print("=" * 60)
    
    # Create environment
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0,
        n_components=4,
        means='random',
        covs=8.0
    )
    
    # Create 4 robots in different corners (for coverage comparison)
    robot_starts = [
        [10, 10],
        [90, 10],
        [10, 90],
        [90, 90]
    ]
    
    robots = []
    for i, start_pos in enumerate(robot_starts):
        robot = Robot(
            robot_id=i,
            initial_position=np.array(start_pos, dtype=float),
            budget_type=BudgetType.TIME,
            initial_budget=30.0,  # 30 seconds
            max_speed=5.0,  # 5 m/s
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
    
    print(f"\nCreated {len(robots)} robots at corners (for comparison with coverage baselines)")
    print("Max possible distance per robot: 150m (30s × 5m/s)")
    
    # Run planner
    planner = RandomMultiRobotPlanner(
        robots=robots,
        environment=env,
        config={'step_size': 15.0, 'seed': 456}
    )
    
    results = planner.execute_mission(max_iterations=100, verbose=False)
    
    print(f"Mission complete: {results['total_measurements']} measurements")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Environment with trajectories
    X, Y, Z = env.evaluate_grid(resolution=100)
    
    im = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax1, label='Field Value')
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, robot in enumerate(robots):
        trajectory = results['robot_trajectories'][robot.id]
        
        # Plot trajectory
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 
                color=colors[i], alpha=0.7, linewidth=2,
                label=f'Robot {robot.id}')
        
        # Plot start and end
        ax1.plot(trajectory[0, 0], trajectory[0, 1], 
                'o', color=colors[i], markersize=10, 
                markeredgecolor='black', markeredgewidth=2)
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 
                's', color=colors[i], markersize=10,
                markeredgecolor='black', markeredgewidth=2)
        
        # Plot measurements
        measurements = results['robot_measurements'][robot.id]
        if measurements:
            meas_pos = np.array([m[0] for m in measurements])
            ax1.scatter(meas_pos[:, 0], meas_pos[:, 1], 
                       c=colors[i], s=30, alpha=0.5, marker='x')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Random Planner: Trajectories on Gaussian Mixture')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bounds[0])
    ax1.set_ylim(bounds[1])
    
    # Plot 2: Statistics
    robot_ids = [r.id for r in robots]
    measurements = [results['stats']['measurements_taken'][i] for i in robot_ids]
    distances = [results['stats']['total_distance'][i] for i in robot_ids]
    
    x = np.arange(len(robot_ids))
    width = 0.35
    
    ax2_measurements = ax2
    ax2_distance = ax2.twinx()
    
    bars1 = ax2_measurements.bar(x - width/2, measurements, width, 
                                  label='Measurements', color='steelblue', alpha=0.7)
    bars2 = ax2_distance.bar(x + width/2, distances, width,
                              label='Distance (m)', color='coral', alpha=0.7)
    
    ax2_measurements.set_xlabel('Robot ID')
    ax2_measurements.set_ylabel('Number of Measurements', color='steelblue')
    ax2_measurements.tick_params(axis='y', labelcolor='steelblue')
    ax2_distance.set_ylabel('Distance Traveled (m)', color='coral')
    ax2_distance.tick_params(axis='y', labelcolor='coral')
    
    ax2_measurements.set_xticks(x)
    ax2_measurements.set_xticklabels([f'Robot {i}' for i in robot_ids])
    ax2_measurements.set_title('Per-Robot Statistics')
    ax2_measurements.grid(True, alpha=0.3, axis='y')
    
    # Add legends
    lines1, labels1 = ax2_measurements.get_legend_handles_labels()
    lines2, labels2 = ax2_distance.get_legend_handles_labels()
    ax2_measurements.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_random_planner.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return results


def test_budget_exhaustion():
    """Test that robots properly stop when budget exhausted."""
    print("\n" + "=" * 60)
    print("Test 3: Budget Exhaustion - Origin Start")
    print("=" * 60)
    
    bounds = np.array([[0, 50], [0, 50]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        seed=999,
        physical_scale=1.0,
        n_components=2
    )
    
    # Create robot with small budget, starting at origin
    robot = Robot(
        robot_id=0,
        initial_position=np.array([0.0, 0.0]),
        budget_type=BudgetType.TIME,
        initial_budget=6.0,  # Only 6 seconds (max 30m at 5m/s)
        max_speed=5.0,
        environment=env
    )
    
    max_possible_distance = robot.initial_budget * robot.max_speed
    print(f"\nRobot with small time budget: {robot.initial_budget}s")
    print(f"Max possible distance: {max_possible_distance}m")
    
    planner = RandomMultiRobotPlanner(
        robots=[robot],
        environment=env,
        config={'step_size': 10.0, 'seed': 789}
    )
    
    results = planner.execute_mission(max_iterations=100, verbose=False)
    
    actual_time_used = robot.initial_budget - robot.remaining_budget
    expected_max_distance = robot.initial_budget * robot.max_speed
    
    print(f"\nResults:")
    print(f"  Iterations: {results['stats']['iterations']}")
    print(f"  Measurements: {results['total_measurements']}")
    print(f"  Distance traveled: {results['total_distance']:.2f}m")
    print(f"  Time used: {actual_time_used:.2f}s / {robot.initial_budget:.2f}s")
    print(f"  Time remaining: {robot.remaining_budget:.2f}s")
    print(f"  Robot stopped: {not robot.is_active}")
    
    # Verify robot stopped due to budget
    # Distance should be <= time * speed
    assert results['total_distance'] <= expected_max_distance + 1.0, \
        f"Robot exceeded time budget! Distance {results['total_distance']:.2f}m > max {expected_max_distance:.2f}m"
    assert not robot.is_active or robot.remaining_budget < 1.0, \
        "Robot should have stopped!"
    
    print("\n✓ Budget exhaustion test passed!")


def test_random_starts():
    """Test with random starting positions."""
    print("\n" + "=" * 60)
    print("Test 4: Random Starting Positions")
    print("=" * 60)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=100,
        physical_scale=1.0,
        n_components=6,
        covs=12.0
    )
    
    # Create robots with random starting positions
    np.random.seed(200)
    n_robots = 4
    robots = []
    
    print(f"\nCreating {n_robots} robots with random starts:")
    for i in range(n_robots):
        # Random position within bounds
        start_x = np.random.uniform(bounds[0, 0] + 10, bounds[0, 1] - 10)
        start_y = np.random.uniform(bounds[1, 0] + 10, bounds[1, 1] - 10)
        start_pos = np.array([start_x, start_y])
        
        robot = Robot(
            robot_id=i,
            initial_position=start_pos,
            budget_type=BudgetType.TIME,
            initial_budget=35.0,  # 35 seconds
            max_speed=5.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
        print(f"  Robot {i}: start=({start_x:.1f}, {start_y:.1f})")
    
    planner = RandomMultiRobotPlanner(
        robots=robots,
        environment=env,
        config={'step_size': 18.0, 'seed': 300}
    )
    
    results = planner.execute_mission(max_iterations=100, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.2f}m")
    print(f"  Avg per robot: {results['total_distance']/n_robots:.2f}m")
    
    print("\n✓ Random starts test passed!")


def test_origin_starts():
    """Test with all robots starting from origin (baseline for experiments)."""
    print("\n" + "=" * 60)
    print("Test 5: All Robots Start at Origin (Baseline)")
    print("=" * 60)
    
    bounds = np.array([[0, 100], [0, 100]])
    env = SyntheticEnvironment(
        bounds=bounds,
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=500,
        physical_scale=1.0,
        n_components=5,
        covs=15.0
    )
    
    # All robots start at (0, 0)
    n_robots = 4
    start_pos = np.array([0.0, 0.0])
    robots = []
    
    print(f"\nCreating {n_robots} robots all starting at origin {start_pos}:")
    for i in range(n_robots):
        robot = Robot(
            robot_id=i,
            initial_position=start_pos.copy(),
            budget_type=BudgetType.TIME,
            initial_budget=30.0,  # 30 seconds
            max_speed=5.0,
            sensor_range=5.0,
            environment=env
        )
        robots.append(robot)
        print(f"  Robot {i}: budget=30s (max distance: 150m)")
    
    planner = RandomMultiRobotPlanner(
        robots=robots,
        environment=env,
        config={'step_size': 20.0, 'seed': 600}
    )
    
    results = planner.execute_mission(max_iterations=100, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Total distance: {results['total_distance']:.2f}m")
    
    # Check that robots dispersed from origin
    distances_from_origin = []
    for robot in robots:
        final_pos = robot.position
        distance_from_origin = np.linalg.norm(final_pos)
        distances_from_origin.append(distance_from_origin)
        print(f"  Robot {robot.id}: final distance from origin = {distance_from_origin:.1f}m")
    
    # At least some robots should have moved away (random walk can be unlucky)
    avg_distance = np.mean(distances_from_origin)
    print(f"  Average distance from origin: {avg_distance:.1f}m")
    assert avg_distance > 10.0, "On average, robots should have dispersed from origin"
    
    print("\n✓ Origin starts test passed!")


if __name__ == "__main__":
    print("Testing Random Multi-Robot Planner")
    print("=" * 60)
    
    # Run tests
    try:
        # Test 1: Basic functionality
        results1, robots1, env1 = test_random_planner_basic()
        
        # Test 2: Visualization (corner starts)
        results2 = test_random_planner_visualization()
        
        # Test 3: Budget exhaustion
        test_budget_exhaustion()
        
        # Test 4: Random starting positions
        test_random_starts()
        
        # Test 5: Origin starts (baseline for experiments)
        test_origin_starts()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nKey findings:")
        print("  • Time-based budgets working correctly")
        print("  • Distance = time × speed constraint respected")
        print("  • Tested: origin starts, corner starts, random starts")
        print("  • Ready for baseline experiments")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
