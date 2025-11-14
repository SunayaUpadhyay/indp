"""
Test suite for Robot physical distance calculations and budget management.

Tests:
- Physical distance conversion with environment
- Budget consumption (DISTANCE, TIME, ENERGY)
- can_reach() with physical distances
- get_total_distance_traveled() with physical units
- Different coordinate systems
- Robot behavior with and without environment
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.robot import Robot, BudgetType
from core.environment import SyntheticEnvironment


class TestRobotWithoutEnvironment:
    """Test robot behavior when no environment is attached (assumes coords = meters)."""
    
    def test_move_without_environment(self):
        """Test move_to() without environment assumes coordinates are meters."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=100.0,
            max_speed=1.0
        )
        
        # Move 10 coordinate units
        distance = robot.move_to([10, 0], timestamp=0.0)
        
        # Should consume 10 meters (assumes coord = meters)
        assert np.allclose(distance, 10.0)
        assert np.allclose(robot.remaining_budget, 90.0)
    
    def test_can_reach_without_environment(self):
        """Test can_reach() without environment."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=50.0,
            max_speed=1.0
        )
        
        # Point 30 units away
        target = np.array([30, 0])
        
        # Should be reachable (30 < 50)
        assert robot.can_reach(target)
        
        # Point 60 units away
        target_far = np.array([60, 0])
        
        # Should not be reachable (60 > 50)
        assert not robot.can_reach(target_far)


class TestRobotWithEnvironmentDistanceBudget:
    """Test robot with environment and DISTANCE budget type."""
    
    def test_move_with_physical_scale(self):
        """Test that move_to() uses physical distance from environment."""
        # Environment: each coord unit = 10 meters
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=1000.0,  # 1000 meters
            max_speed=1.0,
            environment=env
        )
        
        # Move 10 coordinate units (= 100 meters)
        distance = robot.move_to([10, 0], timestamp=0.0)
        
        assert np.allclose(distance, 100.0)  # 10 * 10 = 100 meters
        assert np.allclose(robot.remaining_budget, 900.0)  # 1000 - 100
    
    def test_move_diagonal_with_scale(self):
        """Test diagonal movement with physical scale."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=5.0  # Each coord unit = 5 meters
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=500.0,
            max_speed=1.0,
            environment=env
        )
        
        # Move to (3, 4) - Pythagorean triple
        distance = robot.move_to([3, 4], timestamp=0.0)
        
        # Distance: sqrt(3^2 + 4^2) = 5 coord units = 25 meters
        expected_distance = 5.0 * 5.0  # 25 meters
        assert np.allclose(distance, expected_distance)
        assert np.allclose(robot.remaining_budget, 475.0)
    
    def test_can_reach_with_physical_scale(self):
        """Test can_reach() uses physical distance."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=500.0,  # 500 meters
            max_speed=1.0,
            environment=env
        )
        
        # Target 40 coord units away = 400 meters
        target = np.array([40, 0])
        assert robot.can_reach(target)  # 400 < 500
        
        # Target 60 coord units away = 600 meters
        target_far = np.array([60, 0])
        assert not robot.can_reach(target_far)  # 600 > 500


class TestRobotWithEnvironmentTimeBudget:
    """Test robot with environment and TIME budget type."""
    
    def test_time_budget_consumption(self):
        """Test that time budget uses distance / speed."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0  # Each coord unit = 10 meters
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.TIME,
            initial_budget=1000.0,  # 1000 seconds
            max_speed=2.0,  # 2 m/s
            environment=env
        )
        
        # Move 10 coord units = 100 meters
        # Time = distance / speed = 100 / 2 = 50 seconds
        robot.move_to([10, 0], timestamp=0.0)
        
        expected_remaining = 1000.0 - 50.0
        assert np.allclose(robot.remaining_budget, expected_remaining)
    
    def test_can_reach_with_time_budget(self):
        """Test can_reach() with time budget."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.TIME,
            initial_budget=100.0,  # 100 seconds
            max_speed=5.0,  # 5 m/s
            environment=env
        )
        
        # Target 40 coord units = 400 meters
        # Time needed = 400 / 5 = 80 seconds
        target = np.array([40, 0])
        assert robot.can_reach(target)  # 80 < 100
        
        # Target 60 coord units = 600 meters
        # Time needed = 600 / 5 = 120 seconds
        target_far = np.array([60, 0])
        assert not robot.can_reach(target_far)  # 120 > 100


class TestRobotWithEnvironmentEnergyBudget:
    """Test robot with environment and ENERGY budget type."""
    
    def test_energy_budget_consumption(self):
        """Test that energy budget uses distance^2."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.ENERGY,
            initial_budget=10000.0,
            max_speed=1.0,
            environment=env
        )
        
        # Move 10 coord units = 100 meters
        # Energy = distance^2 = 100^2 = 10000
        robot.move_to([10, 0], timestamp=0.0)
        
        expected_remaining = 10000.0 - 10000.0
        assert np.allclose(robot.remaining_budget, expected_remaining)
    
    def test_can_reach_with_energy_budget(self):
        """Test can_reach() with energy budget."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=1.0  # Simplified: 1 coord = 1 meter
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.ENERGY,
            initial_budget=2500.0,
            max_speed=1.0,
            environment=env
        )
        
        # Target 40 units = 40 meters
        # Energy needed = 40^2 = 1600
        target = np.array([40, 0])
        assert robot.can_reach(target)  # 1600 < 2500
        
        # Target 60 units = 60 meters
        # Energy needed = 60^2 = 3600
        target_far = np.array([60, 0])
        assert not robot.can_reach(target_far)  # 3600 > 2500


class TestRobotTotalDistance:
    """Test get_total_distance_traveled() with physical scaling."""
    
    def test_total_distance_with_scale(self):
        """Test that total distance uses physical meters."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=5000.0,
            max_speed=1.0,
            environment=env
        )
        
        # Move in a path: (0,0) -> (10,0) -> (10,10) -> (0,10)
        robot.move_to([10, 0], timestamp=0.0)
        robot.move_to([10, 10], timestamp=1.0)
        robot.move_to([0, 10], timestamp=2.0)
        
        # Total: 10 + 10 + 10 = 30 coord units = 300 meters
        total_distance = robot.get_total_distance_traveled()
        assert np.allclose(total_distance, 300.0)
    
    def test_total_distance_without_environment(self):
        """Test total distance without environment (assumes coords = meters)."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=500.0,
            max_speed=1.0
        )
        
        # Move in path
        robot.move_to([10, 0], timestamp=0.0)
        robot.move_to([10, 10], timestamp=1.0)
        
        # Total: 10 + 10 = 20 coord units (treated as meters)
        total_distance = robot.get_total_distance_traveled()
        assert np.allclose(total_distance, 20.0)


class TestRobotDifferentCoordinateSystems:
    """Test robot with different coordinate systems."""
    
    def test_negative_coordinate_system(self):
        """Test robot in negative coordinate environment."""
        env = SyntheticEnvironment(
            bounds=np.array([[-50, 50], [-50, 50]]),
            function_name='sphere',
            physical_scale=2.0  # Each coord unit = 2 meters
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([-40, -40]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=200.0,
            max_speed=1.0,
            environment=env
        )
        
        # Move to origin (40*sqrt(2) coord units away)
        # Distance: sqrt(40^2 + 40^2) = 56.57 coord units = 113.14 meters
        distance = robot.move_to([0, 0], timestamp=0.0)
        
        expected_distance = np.sqrt(40**2 + 40**2) * 2.0
        assert np.allclose(distance, expected_distance)
        assert np.allclose(robot.remaining_budget, 200.0 - expected_distance)
    
    def test_mixed_coordinate_system(self):
        """Test robot in mixed positive/negative coordinate environment."""
        env = SyntheticEnvironment(
            bounds=np.array([[-2.25, 2.5], [-2.5, 1.75]]),
            function_name='townsend',
            physical_scale=100.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([-2.25, -2.5]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=1000.0,
            max_speed=1.0,
            environment=env
        )
        
        # Move to center
        center = np.array([0.125, -0.375])
        distance = robot.move_to(center, timestamp=0.0)
        
        # Calculate expected
        coord_distance = np.linalg.norm(center - np.array([-2.25, -2.5]))
        expected_distance = coord_distance * 100.0
        
        assert np.allclose(distance, expected_distance)


class TestRobotBudgetConsistency:
    """Test that budget consumption is consistent across methods."""
    
    def test_move_and_can_reach_consistency(self):
        """Test that move_to() and can_reach() use same distance calculation."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=1000.0,
            max_speed=1.0,
            environment=env
        )
        
        target = np.array([30, 40])  # 50 coord units away
        
        # Check if reachable
        can_reach = robot.can_reach(target)
        assert can_reach  # 500 meters < 1000 budget
        
        # Move there
        distance = robot.move_to(target, timestamp=0.0)
        
        # Should consume exactly the predicted amount
        # Distance = sqrt(30^2 + 40^2) = 50 coord units = 500 meters
        assert np.allclose(distance, 500.0)
        assert np.allclose(robot.remaining_budget, 500.0)
    
    def test_budget_reserve_in_can_reach(self):
        """Test budget_reserve parameter in can_reach()."""
        env = SyntheticEnvironment(
            bounds=np.array([[0, 100], [0, 100]]),
            function_name='peaks',
            physical_scale=10.0
        )
        
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=500.0,
            max_speed=1.0,
            environment=env
        )
        
        # Target exactly 50 coord units away = 500 meters
        target = np.array([50, 0])
        
        # Without reserve: should be reachable (barely)
        assert robot.can_reach(target, budget_reserve=0.0)
        
        # With 50m reserve: should not be reachable
        assert not robot.can_reach(target, budget_reserve=50.0)


class TestRobotTrajectoryTracking:
    """Test trajectory tracking."""
    
    def test_trajectory_records_positions(self):
        """Test that trajectory records all positions."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=1000.0,
            max_speed=1.0
        )
        
        # Initial position
        assert len(robot.trajectory) == 1
        
        # Move three times
        robot.move_to([10, 0], timestamp=0.0)
        robot.move_to([10, 10], timestamp=1.0)
        robot.move_to([0, 10], timestamp=2.0)
        
        # Should have 4 positions (initial + 3 moves)
        assert len(robot.trajectory) == 4
        
        positions = robot.get_trajectory_positions()
        assert positions.shape == (4, 2)
        assert np.allclose(positions[0], [0, 0])
        assert np.allclose(positions[3], [0, 10])


class TestRobotReset:
    """Test robot reset functionality."""
    
    def test_reset_restores_budget(self):
        """Test that reset restores initial budget."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=100.0,
            max_speed=1.0
        )
        
        # Move and consume budget
        robot.move_to([50, 0], timestamp=0.0)
        assert robot.remaining_budget < 100.0
        
        # Reset
        robot.reset()
        
        # Budget should be restored
        assert np.allclose(robot.remaining_budget, 100.0)
    
    def test_reset_clears_trajectory(self):
        """Test that reset clears trajectory."""
        robot = Robot(
            robot_id=0,
            initial_position=np.array([0, 0]),
            budget_type=BudgetType.DISTANCE,
            initial_budget=100.0,
            max_speed=1.0
        )
        
        # Move multiple times
        robot.move_to([10, 0], timestamp=0.0)
        robot.move_to([10, 10], timestamp=1.0)
        assert len(robot.trajectory) > 1
        
        # Reset
        robot.reset()
        
        # Should only have initial position
        assert len(robot.trajectory) == 1


def test_robot_environment_integration():
    """Test complete robot-environment integration."""
    # Create environment: 1km x 1km
    env = SyntheticEnvironment(
        bounds=np.array([[0, 100], [0, 100]]),
        function_name='peaks',
        physical_scale=10.0,
        observation_noise=0.01
    )
    
    # Create robot with 5km budget
    robot = Robot(
        robot_id=0,
        initial_position=np.array([10, 10]),
        budget_type=BudgetType.DISTANCE,
        initial_budget=5000.0,  # 5000 meters = 5 km
        max_speed=2.0,  # 2 m/s
        environment=env
    )
    
    # Plan a path
    waypoints = [
        [30, 10],
        [30, 30],
        [50, 30],
        [50, 50]
    ]
    
    for wp in waypoints:
        # Check if reachable
        if robot.can_reach(wp):
            # Move there
            distance = robot.move_to(wp, timestamp=0.0)
            print(f"Moved to {wp}, distance: {distance:.2f}m, budget: {robot.remaining_budget:.2f}m")
        else:
            print(f"Cannot reach {wp}")
            break
    
    # Check total distance
    total = robot.get_total_distance_traveled()
    print(f"Total distance: {total:.2f}m")
    
    # Should have consumed less than 5km
    assert robot.remaining_budget > 0
    assert total < 5000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
