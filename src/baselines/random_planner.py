"""
Random Multi-Robot Planner - Sanity Check Baseline.

Each robot performs a constrained random walk:
- Picks random valid positions within the environment bounds
- Respects budget constraints
- No coordination between robots (independent random walks)

This serves as the lower bound: any intelligent planner should beat this.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings

from .base_planner import BaseMultiRobotPlanner
from ..core.robot import Robot
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


class RandomMultiRobotPlanner(BaseMultiRobotPlanner):
    """
    Random walk baseline for multi-robot systems.
    
    Strategy:
    - Each robot independently selects random waypoints
    - Validates waypoints are within bounds
    - Respects budget constraints (won't select unreachable points)
    - Option for step size constraints (local vs global random walk)
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: Optional[GaussianProcessBelief] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize random planner.
        
        Args:
            robots: List of Robot instances
            environment: Environment to explore
            gp_belief: Optional shared GP belief (not used by random planner)
            config: Configuration with optional keys:
                - step_size: Maximum step size in meters (None = no limit)
                - max_attempts: Max attempts to find valid waypoint (default: 100)
                - seed: Random seed for reproducibility
        """
        super().__init__(robots, environment, gp_belief, config)
        
        self.step_size = config.get('step_size', None)
        self.max_attempts = config.get('max_attempts', 100)
        
        # Initialize random number generator
        seed = config.get('seed', None)
        self.rng = np.random.RandomState(seed)
    
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Plan random waypoint for each idle robot.
        
        Returns:
            Dictionary mapping robot_id -> next_position
        """
        waypoints = {}
        
        for robot in self.robots:
            # Only plan for idle robots
            if robot.id not in self.idle_robots:
                continue
            
            if not robot.is_active:
                continue
            
            # Generate random waypoint
            waypoint = self._generate_random_waypoint(robot)
            
            if waypoint is not None:
                waypoints[robot.id] = waypoint
        
        return waypoints
    
    def _generate_random_waypoint(self, robot: Robot) -> Optional[np.ndarray]:
        """
        Generate a valid random waypoint for a robot.
        
        Tries multiple times to find a waypoint that is:
        1. Within environment bounds
        2. Reachable within robot's remaining budget
        3. Within step_size if configured
        
        Args:
            robot: Robot to generate waypoint for
            
        Returns:
            Valid waypoint as [x, y] or None if no valid waypoint found
        """
        bounds = self.environment.bounds
        
        for attempt in range(self.max_attempts):
            if self.step_size is not None:
                # Local random walk: sample within step_size radius
                waypoint = self._sample_local_waypoint(robot, bounds)
            else:
                # Global random walk: sample anywhere in environment
                waypoint = self._sample_global_waypoint(bounds)
            
            # Validate waypoint
            if self._is_valid_waypoint(robot, waypoint):
                return waypoint
        
        # Could not find valid waypoint after max_attempts
        warnings.warn(
            f"Robot {robot.id} could not find valid random waypoint after "
            f"{self.max_attempts} attempts. Remaining budget: {robot.remaining_budget:.2f}",
            UserWarning
        )
        return None
    
    def _sample_global_waypoint(self, bounds: np.ndarray) -> np.ndarray:
        """
        Sample random point anywhere in environment.
        
        Args:
            bounds: Environment bounds [[x_min, x_max], [y_min, y_max]]
            
        Returns:
            Random point [x, y]
        """
        x = self.rng.uniform(bounds[0, 0], bounds[0, 1])
        y = self.rng.uniform(bounds[1, 0], bounds[1, 1])
        return np.array([x, y])
    
    def _sample_local_waypoint(self, robot: Robot, bounds: np.ndarray) -> np.ndarray:
        """
        Sample random point within step_size of robot's current position.
        
        Args:
            robot: Robot instance
            bounds: Environment bounds
            
        Returns:
            Random point [x, y] near robot
        """
        # Convert step size to coordinate units if needed
        if self.environment.physical_scale != 1.0:
            step_size_coords = self.environment.meters_to_coord(self.step_size)
        else:
            step_size_coords = self.step_size
        
        # Sample random angle and distance
        angle = self.rng.uniform(0, 2 * np.pi)
        distance = self.rng.uniform(0, step_size_coords)
        
        # Calculate waypoint
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        waypoint = robot.position + np.array([dx, dy])
        
        # Clip to bounds
        waypoint[0] = np.clip(waypoint[0], bounds[0, 0], bounds[0, 1])
        waypoint[1] = np.clip(waypoint[1], bounds[1, 0], bounds[1, 1])
        
        return waypoint
    
    def _is_valid_waypoint(self, robot: Robot, waypoint: np.ndarray) -> bool:
        """
        Check if waypoint is valid for robot.
        
        Args:
            robot: Robot instance
            waypoint: Candidate waypoint [x, y]
            
        Returns:
            True if waypoint is valid
        """
        # Check if within bounds
        if not self.environment.is_within_bounds(waypoint.reshape(1, -1))[0]:
            return False
        
        # Check if reachable within budget
        if not robot.can_reach(waypoint, budget_reserve=0.0):
            return False
        
        return True
    
    def get_planner_info(self) -> Dict[str, Any]:
        """Get information about planner configuration."""
        return {
            'planner_type': 'RandomMultiRobot',
            'step_size': self.step_size,
            'max_attempts': self.max_attempts,
            'num_robots': len(self.robots),
            'seed': self.config.get('seed', None),
        }
