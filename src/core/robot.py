"""
Robot state and dynamics representation.

This module defines the Robot class which encapsulates:
- Current position and state
- Budget tracking (time/energy/distance)
- Kinematic constraints
- Trajectory history
"""

import numpy as np
import warnings
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .environment import Environment


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
        environment: Optional['Environment'] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a robot.
        
        Args:
            robot_id: Unique identifier for this robot
            initial_position: Starting [x, y] position
            budget_type: Type of budget constraint
            initial_budget: Initial budget amount (in meters if DISTANCE, seconds if TIME)
            max_speed: Maximum velocity magnitude (in m/s)
            sensor_range: Sensing radius around robot (in meters)
            environment: Environment instance for coordinate conversion
            config: Additional configuration parameters
        """
        self.id = robot_id
        self.budget_type = budget_type
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.max_speed = max_speed
        self.sensor_range = sensor_range
        self.environment = environment
        self.config = config or {}
        
        # Validate parameters
        self._validate_parameters()
        
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
    
    def _validate_parameters(self) -> None:
        """Validate robot parameters for physical realism."""
        # Speed checks (assuming meters/second)
        if self.max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {self.max_speed}")
        
        if self.max_speed > 50:  # 180 km/h seems unrealistic for most robots
            warnings.warn(
                f"Unusually high max_speed: {self.max_speed} m/s ({self.max_speed*3.6:.1f} km/h). "
                f"Did you mean km/h? Convert with: speed_ms = speed_kmh / 3.6",
                UserWarning
            )
        
        # Budget checks
        if self.initial_budget <= 0:
            raise ValueError(f"initial_budget must be positive, got {self.initial_budget}")
        
        if self.budget_type == BudgetType.DISTANCE:
            if self.initial_budget > 100000:  # 100 km
                warnings.warn(
                    f"Very large distance budget: {self.initial_budget}m ({self.initial_budget/1000:.1f} km)",
                    UserWarning
                )
        elif self.budget_type == BudgetType.TIME:
            if self.initial_budget > 86400:  # 24 hours
                warnings.warn(
                    f"Very large time budget: {self.initial_budget}s ({self.initial_budget/3600:.1f} hours)",
                    UserWarning
                )
        
        # Sensor range check
        if self.sensor_range <= 0:
            warnings.warn(f"sensor_range is {self.sensor_range}, should be positive", UserWarning)
        
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
            Distance traveled in coordinate units (physical meters if environment attached)
        """
        target_position = np.array(target_position, dtype=float)
        
        # Calculate distance in coordinate units
        distance_coords = np.linalg.norm(target_position - self.position)
        
        # Convert to physical distance if environment is available
        if self.environment is not None:
            distance_physical = self.environment.coord_to_meters(distance_coords)
        else:
            # Assume coordinates are already in meters
            distance_physical = distance_coords
        
        # Update state
        velocity = (target_position - self.position) / (distance_coords + 1e-10) * self.max_speed
        heading = np.arctan2(velocity[1], velocity[0])
        
        self.state = RobotState(
            position=target_position.copy(),
            velocity=velocity,
            heading=heading,
            timestamp=timestamp
        )
        
        # Record in trajectory
        self.trajectory.append(self.state)
        
        # Update budget using PHYSICAL distance
        if update_budget:
            if self.budget_type == BudgetType.DISTANCE:
                self.consume_budget(distance_physical)
            elif self.budget_type == BudgetType.TIME:
                # Time = distance / speed (both in physical units)
                time_cost = distance_physical / self.max_speed
                self.consume_budget(time_cost)
            elif self.budget_type == BudgetType.ENERGY:
                # Simple energy model: proportional to distance squared
                energy_cost = distance_physical ** 2
                self.consume_budget(energy_cost)
        
        return distance_physical
    
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
        
        # Calculate distance in coordinate units
        distance_coords = np.linalg.norm(target_position - self.position)
        
        # Convert to physical distance if environment is available
        if self.environment is not None:
            distance_physical = self.environment.coord_to_meters(distance_coords)
        else:
            # Assume coordinates are already in meters
            distance_physical = distance_coords
        
        # Calculate required budget using PHYSICAL distance
        if self.budget_type == BudgetType.DISTANCE:
            required_budget = distance_physical
        elif self.budget_type == BudgetType.TIME:
            required_budget = distance_physical / self.max_speed
        elif self.budget_type == BudgetType.ENERGY:
            required_budget = distance_physical ** 2
        else:
            required_budget = distance_physical
        
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
        Calculate total distance traveled by robot in physical meters.
        
        Returns:
            Total distance in meters (if environment attached) or coordinate units
        """
        positions = self.get_trajectory_positions()
        if len(positions) < 2:
            return 0.0
        
        # Calculate distances in coordinate units
        distances_coords = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance_coords = np.sum(distances_coords)
        
        # Convert to physical distance if environment is available
        if self.environment is not None:
            return self.environment.coord_to_meters(total_distance_coords)
        else:
            return total_distance_coords
    
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
