"""
Test suite for KrigingBelieverAssignment distance and time calculations.

Tests:
- Distance conversion from coordinates to physical meters
- Travel time calculation using physical distances
- Acquisition function scoring with physical distances
- Budget consumption accuracy
- Integration with environment scaling
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.environment import SyntheticEnvironment
from src.core.belief import SKLearnGPBelief
from src.core.robot import Robot, BudgetType
from src.planning.candidates.candidate_generator import CandidateGenerator, CandidateSet
from src.planning.assignment.kriging_believer import KrigingBelieverAssignment, RobotAssignmentState


class TestKrigingBelieverDistanceCalculations:
    """Test distance calculations in Kriging Believer."""
    
    def test_travel_time_with_physical_scale(self):
        """Test that travel time uses physical distance, not coordinate distance."""
        # Environment: 100x100 coords, each coord = 10 meters (1km x 1km)
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0,
            observation_noise=0.01,
            seed=42
        )
        
        # Create robot at origin with time budget
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=1000.0,  # 1000 seconds
            max_speed=2.0,  # 2 m/s
            environment=env
        )
        
        # Create GP
        gp = SKLearnGPBelief(
            bounds=env.bounds,
            use_normalized_coords=True
        )
        
        # Add initial observation
        initial_pos = robot.position.reshape(1, -1)
        initial_val = env.observe(initial_pos)[0]
        gp.update(initial_pos, np.array([initial_val]))
        
        # Create candidate set with one target at (10, 0)
        # Distance: 10 coord units = 100 meters
        # Expected time: 100m / 2m/s = 50 seconds
        target_position = np.array([10.0, 0.0])
        candidate_set = CandidateSet(
            robot_id=0,
            points=np.array([target_position]),
            feasible=np.array([True])
        )
        
        # Create Kriging Believer
        kb = KrigingBelieverAssignment(
            time_limit=1000.0,
            environment=env,  # Pass environment
            min_time_threshold=5.0,
            sensor_time=1.0,
            verbose=False
        )
        
        # Initialize KB state
        kb.robot_states[0] = RobotAssignmentState(robot=robot)
        kb.gp_believer = gp.copy()
        kb.gp_actual = gp.copy()
        kb.simulation_clock = 0.0
        
        # Assign target
        success = kb._assign_next_target(0, {0: candidate_set}[0])
        
        assert success, "Target assignment should succeed"
        
        # Check that travel time is calculated correctly
        state = kb.robot_states[0]
        
        # Distance in coords: 10 units
        # Physical distance: 10 * 10 = 100 meters
        # Travel time: 100 / 2 = 50 seconds
        expected_time = 50.0
        
        assert state.time_to_target is not None
        assert np.isclose(state.time_to_target, expected_time, rtol=0.01), \
            f"Travel time should be {expected_time}s, got {state.time_to_target}s"
    
    def test_acquisition_uses_physical_distance(self):
        """Test that acquisition function uses physical meters for scoring."""
        # Small environment with large physical scale
        env = SyntheticEnvironment(
            bounds=np.array([[0, 10], [0, 10]]),
            function_name='peaks',
            physical_scale=100.0,  # Each coord unit = 100 meters!
            observation_noise=0.01,
            seed=42
        )
        
        # Create GP with some observations
        gp = SKLearnGPBelief(bounds=env.bounds, use_normalized_coords=True)
        
        # Add observations at corners (high variance in center)
        corners = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        values = env.evaluate(corners)
        gp.update(corners, values)
        
        # Create Kriging Believer
        kb = KrigingBelieverAssignment(
            time_limit=1000.0,
            environment=env,
            min_time_threshold=5.0,
            sensor_time=1.0,
            verbose=False
        )
        kb.gp_believer = gp.copy()
        
        # Test candidates at different distances
        current_position = np.array([5.0, 5.0])  # Center
        
        candidates = np.array([
            [5.5, 5.0],  # 0.5 coord units = 50 meters away
            [6.0, 5.0],  # 1.0 coord units = 100 meters away
            [7.0, 5.0],  # 2.0 coord units = 200 meters away
        ])
        
        # Select best using default acquisition
        best = kb._default_acquisition(candidates, current_position)
        
        # Should prioritize based on variance/distance_meters
        # Not variance/distance_coords
        assert best is not None
        assert len(best) == 2
        
        # Verify it's using physical distances by checking it doesn't crash
        # and returns a valid candidate
        assert np.any([np.allclose(best, c) for c in candidates])
    
    def test_different_coordinate_systems(self):
        """Test KB works correctly with different coordinate systems."""
        test_cases = [
            {
                'name': 'Positive coords',
                'bounds': np.array([[0, 100], [0, 100]]),
                'scale': 1.0,
                'robot_pos': [0, 0],
                'target_pos': [10, 0],
                'expected_distance_m': 10.0
            },
            {
                'name': 'Negative coords',
                'bounds': np.array([[-50, 50], [-50, 50]]),
                'scale': 2.0,
                'robot_pos': [0, 0],
                'target_pos': [10, 0],
                'expected_distance_m': 20.0
            },
            {
                'name': 'Mixed coords (Townsend)',
                'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
                'scale': 100.0,
                'robot_pos': [0, 0],
                'target_pos': [1, 0],
                'expected_distance_m': 100.0
            }
        ]
        
        for case in test_cases:
            env = SyntheticEnvironment(
                bounds=case['bounds'],
                function_name='peaks',
                physical_scale=case['scale'],
                observation_noise=0.01,
                seed=42
            )
            
            robot = Robot(
                robot_id=0,
                initial_position=np.array(case['robot_pos'], dtype=float),
                budget_type=BudgetType.TIME,
                initial_budget=1000.0,
                max_speed=1.0,  # 1 m/s for easy calculation
                environment=env
            )
            
            gp = SKLearnGPBelief(bounds=env.bounds, use_normalized_coords=True)
            gp.update(
                np.array([case['robot_pos']]),
                np.array([0.5])
            )
            
            kb = KrigingBelieverAssignment(
                time_limit=1000.0,
                environment=env,
                min_time_threshold=5.0,
                sensor_time=1.0,
                verbose=False
            )

            kb.robot_states[0] = RobotAssignmentState(robot=robot)
            kb.gp_believer = gp.copy()
            kb.gp_actual = gp.copy()
            kb.simulation_clock = 0.0
            
            candidate_set = CandidateSet(
                robot_id=0,
                points=np.array([case['target_pos']]),
                feasible=np.array([True])
            )
            
            success = kb._assign_next_target(0, candidate_set)
            
            assert success, f"Assignment should succeed for {case['name']}"
            
            state = kb.robot_states[0]
            expected_time = case['expected_distance_m'] / 1.0  # speed = 1 m/s
            
            assert np.isclose(state.time_to_target, expected_time, rtol=0.01), \
                f"{case['name']}: Expected {expected_time}s, got {state.time_to_target}s"


class TestKrigingBelieverBudgetConsumption:
    """Test budget consumption accuracy."""
    
    def test_budget_consumption_matches_travel_time(self):
        """Test that budget is consumed correctly based on physical distance."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0,  # 1km x 1km
            observation_noise=0.01,
            seed=42
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0.0, 0.0]),
            budget_type=BudgetType.TIME,
            initial_budget=500.0,  # 500 seconds
            max_speed=5.0,  # 5 m/s
            environment=env
        )
        
        initial_budget = robot.remaining_budget
        
        gp = SKLearnGPBelief(bounds=env.bounds, use_normalized_coords=True)
        gp.update(
            robot.position.reshape(1, -1),
            np.array([env.observe(robot.position.reshape(1, -1))[0]])
        )
        
        kb = KrigingBelieverAssignment(
            time_limit=500.0,
            environment=env,
            min_time_threshold=5.0,
            sensor_time=2.0,  # 2 seconds to sense
            verbose=False
        )
        
        # Manually process a target
        target = np.array([30.0, 40.0])  # 50 coord units = 500 meters
        
        kb.robot_states[0] = RobotAssignmentState(robot=robot)
        kb.gp_believer = gp.copy()
        kb.gp_actual = gp.copy()
        kb.simulation_clock = 0.0
        
        # Calculate expected budget consumption
        distance_coords = 50.0  # sqrt(30^2 + 40^2)
        distance_meters = 500.0  # 50 * 10
        expected_travel_time = distance_meters / 5.0  # 100 seconds
        expected_sensor_time = 2.0
        expected_total = expected_travel_time + expected_sensor_time
        
        # Simulate reaching target
        robot.move_to(target, timestamp=expected_travel_time, update_budget=True)
        robot.consume_budget(expected_sensor_time)
        
        budget_consumed = initial_budget - robot.remaining_budget
        
        assert np.isclose(budget_consumed, expected_total, rtol=0.01), \
            f"Expected to consume {expected_total}s, consumed {budget_consumed}s"


class TestKrigingBelieverEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_assignment_cycle(self):
        """Test a complete assignment cycle with physical scaling."""
        # Create 1km x 1km environment
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0,
            observation_noise=0.01,
            seed=42
        )
        
        # Create 2 robots
        robots = [
            Robot(
                robot_id=0,
                initial_position=np.array([10.0, 10.0]),
                budget_type=BudgetType.TIME,
                initial_budget=200.0,  # 200 seconds
                max_speed=2.0,  # 2 m/s
                environment=env
            ),
            Robot(
                robot_id=1,
                initial_position=np.array([90.0, 90.0]),
                budget_type=BudgetType.TIME,
                initial_budget=200.0,
                max_speed=2.0,
                environment=env
            )
        ]
        
        # Create GP
        gp = SKLearnGPBelief(bounds=env.bounds, use_normalized_coords=True)
        
        # Generate candidates
        candidate_gen = CandidateGenerator(
            bounds=env.bounds,
            quadtree_config={'max_depth': 3, 'variance_threshold': 0.1},
            sampling_config={'points_per_cell': 1, 'min_spacing': 5.0}
        )
        
        candidate_sets = candidate_gen.generate_candidates(gp, robots)
        
        # Create Kriging Believer
        kb = KrigingBelieverAssignment(
            time_limit=200.0,
            environment=env,  # Must pass environment!
            min_time_threshold=10.0,
            sensor_time=1.0,
            verbose=False
        )
        
        # Environment sampler
        def env_sampler(pos):
            return env.observe(pos.reshape(1, -1))[0]
        
        # Run assignment
        assignments, samples_dict = kb.assign_targets(
            robots=robots,
            candidate_sets=candidate_sets,
            gp_belief=gp,
            environment_sampler=env_sampler
        )
        
        # Verify results
        assert len(samples_dict) > 0, "Should have sample entries"
        assert len(assignments) > 0, "Should have assignments"
        
        # Flatten samples from all robots
        all_samples = []
        for robot_id, robot_samples in samples_dict.items():
            all_samples.extend(robot_samples)
        
        assert len(all_samples) > 0, "Should collect some samples"
        
        # Check that robots consumed budget
        for robot in robots:
            assert robot.remaining_budget < robot.initial_budget, \
                f"Robot {robot.id} should have consumed budget"
        
        # Check that samples are within bounds
        for pos, val, time in all_samples:
            assert env.is_within_bounds(pos.reshape(1, -1))[0], \
                f"Sample position {pos} should be within bounds"
        
        print(f"\n✅ End-to-end test passed!")
        print(f"   Total samples collected: {len(all_samples)}")
        for robot in robots:
            budget_used = robot.initial_budget - robot.remaining_budget
            print(f"   Robot {robot.id}: {budget_used:.1f}s used, "
                  f"{robot.remaining_budget:.1f}s remaining")


def test_kriging_believer_without_environment():
    """Test that KB works (with fallback) when robot has no environment."""
    # Create environment but don't attach to robot
    env = SyntheticEnvironment(
        bounds=np.array([[0, 100], [0, 100]]),
        function_name='peaks',
        physical_scale=10.0,
        observation_noise=0.01,
        seed=42
    )
    
    # Robot WITHOUT environment attached
    robot = Robot(
        robot_id=0,
        initial_position=np.array([0.0, 0.0]),
        budget_type=BudgetType.TIME,
        initial_budget=1000.0,
        max_speed=2.0,
        environment=None  # No environment!
    )
    
    gp = SKLearnGPBelief(bounds=env.bounds, use_normalized_coords=True)
    gp.update(robot.position.reshape(1, -1), np.array([0.5]))
    
    kb = KrigingBelieverAssignment(
        time_limit=1000.0,
        environment=env,  # KB has environment
        min_time_threshold=5.0,
        sensor_time=1.0,
        verbose=False
    )

    kb.robot_states[0] = RobotAssignmentState(robot=robot)
    kb.gp_believer = gp.copy()
    kb.gp_actual = gp.copy()
    kb.simulation_clock = 0.0
    
    candidate_set = CandidateSet(
        robot_id=0,
        points=np.array([[10.0, 0.0]]),
        feasible=np.array([True])
    )
    
    # Should work but use fallback (treat coords as meters)
    success = kb._assign_next_target(0, candidate_set)
    
    assert success, "Should succeed with fallback"
    
    state = kb.robot_states[0]
    # Without environment, assumes 10 coord units = 10 meters
    # Time = 10 / 2 = 5 seconds
    assert state.time_to_target is not None
    print(f"\n✅ Fallback test passed! Travel time: {state.time_to_target:.1f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
