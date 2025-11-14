"""
Test script to verify coordinate normalization and physical scaling work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.environment import create_environment
from src.core.robot import Robot, BudgetType
from src.core.belief import create_gp_belief


def test_environment_scaling():
    """Test environment coordinate to meter conversion."""
    print("="*70)
    print("TEST 1: Environment Coordinate Scaling")
    print("="*70)
    
    # Create environment with physical scale
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
    physical_scale = 100.0  # Each coordinate unit = 100 meters
    
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name='townsend',
        physical_scale=physical_scale,
        use_normalized_coords=False  # Test without normalization first
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  Bounds: {bounds.tolist()}")
    print(f"  Physical scale: {physical_scale} m/unit")
    print(f"  Coord range: {env.coord_range}")
    print(f"  Physical size: {env.physical_size} meters")
    print(f"  Physical area: {env.physical_area:.0f} m² ({env.physical_area/1e6:.3f} km²)")
    
    # Test distance conversion
    distance_coords = 2.0
    distance_meters = env.coord_to_meters(distance_coords)
    print(f"\n  Distance conversion:")
    print(f"    {distance_coords} coord units = {distance_meters} meters")
    assert distance_meters == 200.0, f"Expected 200m, got {distance_meters}m"
    
    print("\n✅ Environment scaling test PASSED")


def test_gp_normalization():
    """Test GP coordinate normalization."""
    print("\n" + "="*70)
    print("TEST 2: GP Coordinate Normalization")
    print("="*70)
    
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
    
    # Create GP with normalization enabled
    gp = create_gp_belief(
        bounds=bounds,
        kernel_type='matern',
        length_scale=0.1,  # In normalized space
        use_normalized_coords=True
    )
    
    print(f"\nGP Configuration:")
    print(f"  Bounds (original): {gp.bounds.tolist()}")
    print(f"  Bounds (internal): {gp.internal_bounds.tolist()}")
    print(f"  Use normalized: {gp.use_normalized_coords}")
    print(f"  Coord range: {gp.coord_range}")
    
    # Test coordinate conversion
    X_original = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 1.0]])
    X_normalized = gp.to_internal(X_original)
    X_back = gp.from_internal(X_normalized)
    
    print(f"\n  Coordinate conversion test:")
    print(f"    Original:   {X_original}")
    print(f"    Normalized: {X_normalized}")
    print(f"    Back:       {X_back}")
    
    # Verify round-trip conversion
    assert np.allclose(X_original, X_back), "Round-trip conversion failed!"
    
    # Verify normalized coordinates are in [0, 1]
    assert np.all(X_normalized >= 0) and np.all(X_normalized <= 1), \
        "Normalized coordinates outside [0,1]!"
    
    print("\n✅ GP normalization test PASSED")


def test_gp_training_with_normalization():
    """Test GP training with normalized coordinates."""
    print("\n" + "="*70)
    print("TEST 3: GP Training with Normalization")
    print("="*70)
    
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
    
    # Create GP
    gp = create_gp_belief(
        bounds=bounds,
        kernel_type='matern',
        length_scale=0.1,
        use_normalized_coords=True
    )
    
    # Train with original coordinates
    X_train_original = np.array([
        [-2.0, -2.0],
        [0.0, 0.0],
        [2.0, 1.0]
    ])
    y_train = np.array([0.5, 1.0, 0.3])
    
    print(f"\nTraining GP with original coordinates:")
    print(f"  X_train (original): {X_train_original}")
    print(f"  y_train: {y_train}")
    
    gp.update(X_train_original, y_train)
    
    # Check internal storage
    info = gp.get_training_info()
    print(f"\nGP Training Info:")
    print(f"  N samples: {info['n_train']}")
    print(f"  Is fitted: {info['is_fitted']}")
    print(f"  X_train range (internal): [{info['X_train_min']}, {info['X_train_max']}]")
    print(f"  y_train: mean={info['y_train_mean']:.3f}, std={info['y_train_std']:.3f}")
    
    # Verify internal coordinates are normalized
    assert np.all(info['X_train_min'] >= 0) and np.all(info['X_train_max'] <= 1), \
        "Training data not properly normalized!"
    
    # Test prediction with original coordinates
    X_test_original = np.array([[0.0, 0.0], [1.0, 0.5]])
    mean, std = gp.predict(X_test_original, return_std=True)
    
    print(f"\nPrediction at original coordinates:")
    print(f"  X_test (original): {X_test_original}")
    print(f"  Predicted mean: {mean}")
    print(f"  Predicted std: {std}")
    
    # Sanity checks
    assert len(mean) == len(X_test_original), "Mean prediction length mismatch!"
    assert len(std) == len(X_test_original), "Std prediction length mismatch!"
    assert np.all(std > 0), "Standard deviation must be positive!"
    
    print("\n✅ GP training test PASSED")


def test_robot_budget_with_scaling():
    """Test robot budget consumption with coordinate scaling."""
    print("\n" + "="*70)
    print("TEST 4: Robot Budget with Coordinate Scaling")
    print("="*70)
    
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
    physical_scale = 100.0
    
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name='townsend',
        physical_scale=physical_scale,
        use_normalized_coords=False
    )
    
    # Create robot
    robot = Robot(
        robot_id=0,
        initial_position=np.array([-2.0, -2.0]),
        max_speed=5.0,  # 5 m/s
        initial_budget=1000.0,  # 1000 meters
        budget_type=BudgetType.DISTANCE,
        environment=env
    )
    
    print(f"\nRobot Configuration:")
    print(f"  Initial position: {robot.position}")
    print(f"  Max speed: {robot.max_speed} m/s")
    print(f"  Initial budget: {robot.initial_budget} meters")
    print(f"  Budget type: {robot.budget_type.value}")
    
    # Move robot
    target = np.array([0.0, 0.0])
    distance_traveled = robot.move_to(target, timestamp=1.0)
    
    # Calculate expected values
    distance_coords = np.linalg.norm(target - np.array([-2.0, -2.0]))
    expected_distance_meters = distance_coords * physical_scale
    
    print(f"\nMovement Test:")
    print(f"  From: [-2.0, -2.0]")
    print(f"  To: [0.0, 0.0]")
    print(f"  Distance (coords): {distance_coords:.3f} units")
    print(f"  Expected (meters): {expected_distance_meters:.1f} m")
    print(f"  Actual (meters): {distance_traveled:.1f} m")
    print(f"  Budget consumed: {robot.initial_budget - robot.remaining_budget:.1f} m")
    print(f"  Budget remaining: {robot.remaining_budget:.1f} m")
    
    # Verify budget consumption
    assert np.isclose(distance_traveled, expected_distance_meters, rtol=0.01), \
        f"Distance mismatch! Expected {expected_distance_meters}, got {distance_traveled}"
    
    assert np.isclose(robot.initial_budget - robot.remaining_budget, expected_distance_meters, rtol=0.01), \
        "Budget consumption doesn't match distance traveled!"
    
    print("\n✅ Robot budget test PASSED")


def test_end_to_end_consistency():
    """Test that environment, GP, and robot all use consistent coordinate systems."""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Consistency")
    print("="*70)
    
    bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
    physical_scale = 100.0
    
    # Create environment
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name='townsend',
        physical_scale=physical_scale,
        use_normalized_coords=False
    )
    
    # Create GP
    gp = create_gp_belief(
        bounds=bounds,
        kernel_type='matern',
        length_scale=0.1,
        use_normalized_coords=True
    )
    
    # Create robot
    robot = Robot(
        robot_id=0,
        initial_position=np.array([-2.0, -2.0]),
        max_speed=5.0,
        initial_budget=600.0,  # seconds
        budget_type=BudgetType.TIME,
        environment=env
    )
    
    print(f"\nSystem Configuration:")
    print(f"  Environment: physical_scale={physical_scale}, norm={env.use_normalized_coords}")
    print(f"  GP: use_normalized={gp.use_normalized_coords}")
    print(f"  Robot: speed={robot.max_speed}m/s, budget={robot.initial_budget}s")
    
    # Simulate observation and GP update
    obs_location = robot.position.copy()
    obs_value = env.evaluate(obs_location.reshape(1, -1))[0]
    
    print(f"\nObservation:")
    print(f"  Location (original): {obs_location}")
    print(f"  Value: {obs_value:.4f}")
    
    # Update GP (pass original coordinates)
    gp.update(obs_location.reshape(1, -1), np.array([obs_value]))
    
    # Check GP internals
    info = gp.get_training_info()
    print(f"\nGP After Update:")
    print(f"  N samples: {info['n_train']}")
    print(f"  X_train range (internal): [{info['X_train_min']}, {info['X_train_max']}]")
    
    # Make prediction at same location
    mean, std = gp.predict(obs_location.reshape(1, -1), return_std=True)
    print(f"\nPrediction at observation location:")
    print(f"  Mean: {mean[0]:.4f} (should be close to {obs_value:.4f})")
    print(f"  Std: {std[0]:.4f} (should be small)")
    
    # Verify prediction is close to observation
    assert np.abs(mean[0] - obs_value) < 0.1, \
        f"GP prediction {mean[0]} far from observation {obs_value}!"
    
    # Move robot and check budget
    target = np.array([0.0, 0.0])
    distance_traveled = robot.move_to(target, timestamp=1.0)
    time_consumed = distance_traveled / robot.max_speed
    
    print(f"\nRobot Movement:")
    print(f"  Distance: {distance_traveled:.1f} m")
    print(f"  Time: {time_consumed:.1f} s")
    print(f"  Budget remaining: {robot.remaining_budget:.1f} s")
    
    assert robot.remaining_budget > 0, "Robot ran out of budget!"
    
    print("\n✅ End-to-end consistency test PASSED")


if __name__ == '__main__':
    try:
        test_environment_scaling()
        test_gp_normalization()
        test_gp_training_with_normalization()
        test_robot_budget_with_scaling()
        test_end_to_end_consistency()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nCoordinate scaling system is working correctly:")
        print("  ✅ Environment physical scaling")
        print("  ✅ GP coordinate normalization")
        print("  ✅ GP training/prediction")
        print("  ✅ Robot budget consumption")
        print("  ✅ End-to-end consistency")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
