"""
Lawnmower (Coverage) Multi-Robot Planner - Geometric Coverage Baseline.

Partitions the environment into vertical strips (one per robot) and follows
a back-and-forth lawnmower pattern within each strip.

This is a non-informative baseline that provides systematic geometric coverage
without using any GP belief or information gain.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings

from .base_planner import BaseMultiRobotPlanner
from ..core.robot import Robot
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


class LawnmowerPlanner(BaseMultiRobotPlanner):
    """
    Lawnmower coverage baseline for multi-robot systems.
    
    Strategy:
    1. Partition map into N vertical strips (one per robot)
    2. Each robot follows a predetermined lawnmower path in its strip:
       - Start from depot, move to strip entry
       - Perform back-and-forth sweeps in y direction
       - Shift x by stripe_width between sweeps
    3. Continue until budget exhausted
    
    No coordination or information gain - pure geometric coverage.
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: Optional[GaussianProcessBelief] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize lawnmower planner.
        
        Args:
            robots: List of Robot instances
            environment: Environment to explore
            gp_belief: Optional GP belief (not used by lawnmower)
            config: Configuration with optional keys:
                - stripe_width: Distance between parallel sweeps (meters)
                - orientation: 'vertical' or 'horizontal' strips
        """
        super().__init__(robots, environment, gp_belief, config)
        
        self.stripe_width = config.get('stripe_width', 10.0)
        self.orientation = config.get('orientation', 'vertical')
        self.waypoint_spacing = config.get('waypoint_spacing', 10.0)  # Spacing along sweeps
        
        # Precompute lawnmower paths for each robot
        self.robot_paths = {}
        self._generate_lawnmower_paths()
        
        # Track current waypoint index for each robot
        self.waypoint_indices = {robot.id: 0 for robot in robots}
    
    def _generate_lawnmower_paths(self):
        """Generate complete lawnmower path for each robot."""
        bounds = self.environment.bounds
        n_robots = len(self.robots)
        
        if self.orientation == 'vertical':
            # Partition into vertical strips
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            strip_width = (x_max - x_min) / n_robots
            
            for i, robot in enumerate(self.robots):
                # Strip boundaries for this robot
                strip_x_min = x_min + i * strip_width
                strip_x_max = x_min + (i + 1) * strip_width
                
                path = self._generate_vertical_lawnmower(
                    strip_x_min, strip_x_max, y_min, y_max
                )
                self.robot_paths[robot.id] = path
                
        else:  # horizontal
            # Partition into horizontal strips
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            strip_height = (y_max - y_min) / n_robots
            
            for i, robot in enumerate(self.robots):
                # Strip boundaries for this robot
                strip_y_min = y_min + i * strip_height
                strip_y_max = y_min + (i + 1) * strip_height
                
                path = self._generate_horizontal_lawnmower(
                    x_min, x_max, strip_y_min, strip_y_max
                )
                self.robot_paths[robot.id] = path
    
    def _generate_vertical_lawnmower(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> List[np.ndarray]:
        """
        Generate vertical lawnmower path (vertical strips, horizontal sweeps).
        
        Pattern:
        1. Move to strip entry at (x_min, y_min) - robot starts at depot
        2. Sweep up: y_min → y_max at x_min
        3. Shift right: x_min → x_min + stripe_width
        4. Sweep down: y_max → y_min at x_min + stripe_width
        5. Repeat until strip covered
        
        Args:
            x_min, x_max: Strip x boundaries
            y_min, y_max: Strip y boundaries
            
        Returns:
            List of waypoints as numpy arrays
        """
        path = []
        
        # First waypoint: strip entry (bottom-left corner of assigned strip)
        # Robot will travel from depot (0,0) to here
        path.append(np.array([x_min, y_min]))
        
        # Generate back-and-forth sweeps within strip
        x_current = x_min
        going_up = True
        
        while x_current <= x_max:
            if going_up:
                # Sweep up in y - add intermediate waypoints
                y_current = y_min
                while y_current < y_max:
                    y_next = min(y_current + self.waypoint_spacing, y_max)
                    path.append(np.array([x_current, y_next]))
                    y_current = y_next
                going_up = False
            else:
                # Sweep down in y - add intermediate waypoints
                y_current = y_max
                while y_current > y_min:
                    y_next = max(y_current - self.waypoint_spacing, y_min)
                    path.append(np.array([x_current, y_next]))
                    y_current = y_next
                going_up = True
            
            # Move to next sweep line
            x_current += self.stripe_width
            
            # Add horizontal transition to next sweep (if within strip)
            if x_current <= x_max:
                path.append(np.array([x_current, y_max if not going_up else y_min]))
        
        return path
    
    def _generate_horizontal_lawnmower(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> List[np.ndarray]:
        """
        Generate horizontal lawnmower path (horizontal strips, vertical sweeps).
        
        Similar to vertical but sweeps in x direction.
        """
        path = []
        
        # First waypoint: strip entry (bottom-left corner of assigned strip)
        path.append(np.array([x_min, y_min]))
        
        # Generate back-and-forth sweeps within strip
        y_current = y_min
        going_right = True
        
        while y_current <= y_max:
            if going_right:
                # Sweep right in x - add intermediate waypoints
                x_current = x_min
                while x_current < x_max:
                    x_next = min(x_current + self.waypoint_spacing, x_max)
                    path.append(np.array([x_next, y_current]))
                    x_current = x_next
                going_right = False
            else:
                # Sweep left in x - add intermediate waypoints
                x_current = x_max
                while x_current > x_min:
                    x_next = max(x_current - self.waypoint_spacing, x_min)
                    path.append(np.array([x_next, y_current]))
                    x_current = x_next
                going_right = True
            
            # Move to next sweep line
            y_current += self.stripe_width
            
            # Add vertical transition to next sweep (if within strip)
            if y_current <= y_max:
                path.append(np.array([x_max if not going_right else x_min, y_current]))
        
        return path
    
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Get next waypoint from precomputed path for each idle robot.
        
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
            
            # Get next waypoint from precomputed path
            path = self.robot_paths[robot.id]
            idx = self.waypoint_indices[robot.id]
            
            # Check if path complete
            if idx >= len(path):
                # Path exhausted, robot stays at current position
                continue
            
            next_waypoint = path[idx]
            
            # Check if robot can reach this waypoint
            if robot.can_reach(next_waypoint):
                waypoints[robot.id] = next_waypoint
                # Advance to next waypoint for next step
                self.waypoint_indices[robot.id] += 1
            else:
                # Can't reach next waypoint - budget exhausted
                # Robot will become inactive after this step
                pass
        
        return waypoints
    
    def reset(self):
        """Reset planner state for new mission."""
        super().reset()
        self.waypoint_indices = {robot.id: 0 for robot in self.robots}
        # Regenerate paths in case robot positions changed
        self._generate_lawnmower_paths()
    
    def get_planner_info(self) -> Dict[str, Any]:
        """Return information about planner configuration."""
        info = {
            'planner_name': 'LawnmowerPlanner',
            'orientation': self.orientation,
            'stripe_width': self.stripe_width,
            'n_robots': len(self.robots),
            'total_waypoints': {
                robot_id: len(path) 
                for robot_id, path in self.robot_paths.items()
            }
        }
        return info
