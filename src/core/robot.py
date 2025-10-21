"""
Robot state and dynamics representation.

This module defines the Robot class which encapsulates:
- Current position and state
- Budget tracking (time/energy/distance)
- Kinematic constraints
- Trajectory history
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class BudgetType(Enum):
    """Type of budget constraint."""
    TIME = "time"
    ENERGY = "energy"
    DISTANCE = "distance"


@dataclass
class RobotState:
    """
    Snapshot of robot state at a given time.
    
    Attributes:
        position: Current [x, y] position
        velocity: Current [vx, vy] velocity (optional)
        heading: Current heading angle in radians (optional)
        timestamp: Time of this state
    """
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    heading: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Ensure position is a numpy array."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if self.velocity is not None and not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)


class Robot:
    """
    Robot agent with state, budget, and trajectory tracking.
    
    This class manages:
    - Robot identification
    - Current state (position, velocity, etc.)
    - Budget tracking and consumption
    - Trajectory history
    - Kinematic constraints
    
    Design rationale:
    - Immutable ID for multi-robot coordination
    - Flexible budget types for different mission scenarios
    - Trajectory history for analysis and visualization
    - Extensible for different robot types via inheritance
    """
    
    def __init__(
        self,
        robot_id: int,
        initial_position: np.ndarray,
        budget_type: BudgetType = BudgetType.DISTANCE,
        initial_budget: float = 100.0,
        max_speed: float = 1.0,
        sensor_range: float = 5.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a robot.
        
        Args:
            robot_id: Unique identifier for this robot
            initial_position: Starting [x, y] position
            budget_type: Type of budget constraint
            initial_budget: Initial budget amount
            max_speed: Maximum velocity magnitude
            sensor_range: Sensing radius around robot
            config: Additional configuration parameters
        """
        self.id = robot_id
        self.budget_type = budget_type
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.max_speed = max_speed
        self.sensor_range = sensor_range
        self.config = config or {}
        
        # Initialize state
        self.state = RobotState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(2),
            heading=0.0,
            timestamp=0.0
        )
        
        # Trajectory history: list of RobotState objects
        self.trajectory: List[RobotState] = [self.state]
        
        # Measurements collected: list of (position, value, timestamp)
        self.measurements: List[Tuple[np.ndarray, float, float]] = []
        
    @property
    def position(self) -> np.ndarray:
        """Current position."""
        return self.state.position
    
    @property
    def is_active(self) -> bool:
        """Check if robot still has budget remaining."""
        return self.remaining_budget > 0
    
    def consume_budget(self, amount: float) -> None:
        """
        Consume budget.
        
        Args:
            amount: Budget amount to consume
        """
        self.remaining_budget = max(0, self.remaining_budget - amount)
    
    def move_to(
        self,
        target_position: np.ndarray,
        timestamp: float,
        update_budget: bool = True
    ) -> float:
        """
        Move robot to a target position.
        
        Args:
            target_position: Target [x, y] position
            timestamp: Time of this move
            update_budget: Whether to consume budget based on distance
            
        Returns:
            Distance traveled (budget consumed if applicable)
        """
        target_position = np.array(target_position, dtype=float)
        distance = np.linalg.norm(target_position - self.position)
        
        # Update state
        velocity = (target_position - self.position) / (distance + 1e-10) * self.max_speed
        heading = np.arctan2(velocity[1], velocity[0])
        
        self.state = RobotState(
            position=target_position.copy(),
            velocity=velocity,
            heading=heading,
            timestamp=timestamp
        )
        
        # Record in trajectory
        self.trajectory.append(self.state)
        
        # Update budget
        if update_budget:
            if self.budget_type == BudgetType.DISTANCE:
                self.consume_budget(distance)
            elif self.budget_type == BudgetType.TIME:
                time_cost = distance / self.max_speed
                self.consume_budget(time_cost)
            elif self.budget_type == BudgetType.ENERGY:
                # Simple energy model: proportional to distance squared
                energy_cost = distance ** 2
                self.consume_budget(energy_cost)
        
        return distance
    
    def add_measurement(
        self,
        position: np.ndarray,
        value: float,
        timestamp: float
    ) -> None:
        """
        Record a measurement taken by this robot.
        
        Args:
            position: Position where measurement was taken
            value: Measured value
            timestamp: Time of measurement
        """
        self.measurements.append((np.array(position), value, timestamp))
    
    def can_reach(
        self,
        target_position: np.ndarray,
        budget_reserve: float = 0.0
    ) -> bool:
        """
        Check if robot can reach a target position within its budget.
        
        Args:
            target_position: Target [x, y] position
            budget_reserve: Budget to keep in reserve (safety margin)
            
        Returns:
            True if target is reachable within budget
        """
        target_position = np.array(target_position, dtype=float)
        distance = np.linalg.norm(target_position - self.position)
        
        if self.budget_type == BudgetType.DISTANCE:
            required_budget = distance
        elif self.budget_type == BudgetType.TIME:
            required_budget = distance / self.max_speed
        elif self.budget_type == BudgetType.ENERGY:
            required_budget = distance ** 2
        else:
            required_budget = distance
        
        return (self.remaining_budget - budget_reserve) >= required_budget
    
    def get_trajectory_positions(self) -> np.ndarray:
        """
        Get all positions in trajectory as array.
        
        Returns:
            Array of shape (n_points, 2) with trajectory positions
        """
        return np.array([state.position for state in self.trajectory])
    
    def get_total_distance_traveled(self) -> float:
        """
        Calculate total distance traveled by robot.
        
        Returns:
            Total distance in trajectory
        """
        positions = self.get_trajectory_positions()
        if len(positions) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.sum(distances)
    
    def reset(self, initial_position: Optional[np.ndarray] = None) -> None:
        """
        Reset robot to initial state.
        
        Args:
            initial_position: New initial position (uses original if None)
        """
        if initial_position is None:
            initial_position = self.trajectory[0].position
        
        self.remaining_budget = self.initial_budget
        self.state = RobotState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(2),
            heading=0.0,
            timestamp=0.0
        )
        self.trajectory = [self.state]
        self.measurements = []
    
    def __repr__(self) -> str:
        return (f"Robot(id={self.id}, pos={self.position}, "
                f"budget={self.remaining_budget:.2f}/{self.initial_budget:.2f})")
