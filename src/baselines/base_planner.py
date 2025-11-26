"""
Base class for multi-robot planners.

Defines the common interface for all baseline and advanced multi-robot planners.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import heapq
from dataclasses import dataclass

from ..core.robot import Robot, BudgetType, RobotState
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


@dataclass
class RobotEvent:
    """Event for robot reaching a waypoint."""
    time: float
    robot_id: int
    position: np.ndarray
    
    def __lt__(self, other):
        """Priority queue comparison (earlier times first)."""
        return self.time < other.time


class BaseMultiRobotPlanner(ABC):
    """
    Abstract base class for multi-robot planners.
    
    Provides common interface and utilities for:
    - Single-step planning (get next waypoints for all robots)
    - Full mission execution
    - Statistics tracking
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: Optional[GaussianProcessBelief] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-robot planner.
        
        Args:
            robots: List of Robot instances
            environment: Environment to explore
            gp_belief: Shared GP belief (if using informative planning)
            config: Planner-specific configuration
        """
        self.robots = robots
        self.environment = environment
        self.gp_belief = gp_belief
        self.config = config or {}
        
        # Event-driven execution state
        self.simulation_clock = 0.0
        self.event_queue: List[RobotEvent] = []
        self.sensor_time = config.get('sensor_time', 5.0)  # Time to take measurement (seconds)
        self.min_time_threshold = config.get('min_time_threshold', 0.0)
        
        # Track which robots are currently idle (waiting for next waypoint)
        self.idle_robots = set(robot.id for robot in robots)
        
        # Statistics tracking
        self.stats = {
            'total_distance': {robot.id: 0.0 for robot in robots},
            'measurements_taken': {robot.id: 0 for robot in robots},
            'events_processed': 0,
        }
    
    @abstractmethod
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Plan next waypoint for each idle robot.
        
        Called when robots are idle and ready for new waypoints.
        Use self.idle_robots to check which robots need planning.
        
        Returns:
            Dictionary mapping robot_id -> next_position (x, y)
            Only includes idle robots that should receive new waypoints
        """
        pass
    
    def execute_mission(
        self,
        max_iterations: int = 1000,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute full mission using event-driven parallel simulation.
        
        All robots move simultaneously. Events are processed in time order
        to simulate realistic parallel execution.
        
        Args:
            max_iterations: Maximum number of events to process (safety limit)
            verbose: Print progress information
            
        Returns:
            Dictionary with mission statistics and results
        """
        # Initialize: all robots start idle at t=0, take initial measurements
        self.simulation_clock = 0.0
        self.event_queue = []
        self.idle_robots = set(robot.id for robot in self.robots)
        
        # Take initial measurements at starting positions
        for robot in self.robots:
            if robot.is_active:
                self._take_measurement(robot, robot.position, verbose)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STARTING EVENT-DRIVEN MISSION SIMULATION")
            print(f"{'='*70}")
            print(f"Robots: {len(self.robots)}")
            print(f"Initial budget per robot: {self.robots[0].initial_budget:.1f}s")
            print(f"Sensor time: {self.sensor_time:.1f}s")
            print(f"\nInitial measurements taken at depot positions")
        
        # Plan initial waypoints for all idle robots
        self._plan_for_idle_robots(verbose)
        
        # Main event loop: process events in time order
        events_processed = 0
        while self.event_queue and events_processed < max_iterations:
            # Get next event (earliest time)
            event = heapq.heappop(self.event_queue)
            
            # Update simulation clock
            self.simulation_clock = event.time
            
            # Process event: robot reaches waypoint
            robot = self._get_robot(event.robot_id)
            
            if not robot.is_active:
                # Robot ran out of budget during travel
                continue
            
            # Move robot to position
            old_position = robot.position.copy()
            
            # Create new state and append to trajectory
            new_state = RobotState(
                position=event.position.copy(),
                velocity=np.zeros(2),
                heading=0.0,
                timestamp=event.time
            )
            robot.state = new_state
            robot.trajectory.append(new_state)
            
            # Calculate distance traveled
            distance_coords = np.linalg.norm(event.position - old_position)
            if robot.environment is not None:
                distance_meters = robot.environment.coord_to_meters(distance_coords)
            else:
                distance_meters = distance_coords
            
            self.stats['total_distance'][robot.id] += distance_meters
            
            # Take measurement at new position
            self._take_measurement(robot, event.position, verbose)
            
            # Mark robot as idle (ready for next waypoint)
            self.idle_robots.add(robot.id)
            
            if verbose:
                print(f"  Robot {robot.id} now IDLE, ready for next waypoint")
                print(f"  Budget remaining: {robot.remaining_budget:.1f}s")
            
            # Plan next waypoint for this robot (and any others that are idle)
            self._plan_for_idle_robots(verbose)
            
            events_processed += 1
            self.stats['events_processed'] = events_processed
            
            if verbose and events_processed % 10 == 0:
                active = sum(1 for r in self.robots if r.is_active)
                total_measurements = sum(self.stats['measurements_taken'].values())
                print(f"\n[Status] Time: {self.simulation_clock:.1f}s, "
                      f"Active robots: {active}, "
                      f"Total measurements: {total_measurements}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"MISSION COMPLETE")
            print(f"{'='*70}")
            print(f"Final time: {self.simulation_clock:.1f}s")
            print(f"Events processed: {events_processed}")
            print(f"Total measurements: {sum(self.stats['measurements_taken'].values())}")
        
        self.stats['iterations'] = events_processed
        
        # Compile final statistics
        return self._compile_results()
    
    def _plan_for_idle_robots(self, verbose: bool = False) -> None:
        """Plan next waypoints for all idle robots."""
        if not self.idle_robots:
            return
        
        # Check if mission time limit reached (if configured)
        time_limit = self.config.get('time_limit', None)
        if time_limit is not None and self.simulation_clock >= time_limit:
            if verbose:
                print(f"\n[Time {self.simulation_clock:.1f}s] Mission time limit reached ({time_limit}s)")
                print(f"  Stopping mission execution")
            return
        
        # Get waypoints for all idle robots from planner
        waypoints = self.plan_step()
        
        if not waypoints:
            return
        
        # Schedule events for each waypoint
        for robot_id, waypoint in waypoints.items():
            if robot_id not in self.idle_robots:
                continue  # Robot not idle, skip
            
            robot = self._get_robot(robot_id)

            if robot.remaining_budget <= self.min_time_threshold:
                # Robot does not have enough remaining time to justify travel + measurement
                self.idle_robots.discard(robot_id)
                if verbose:
                    print(f"\n[Time {self.simulation_clock:.1f}s] Robot {robot_id} - Minimum threshold reached")
                    print(f"  Remaining budget: {robot.remaining_budget:.1f}s (threshold {self.min_time_threshold:.1f}s)")
                    print(f"  Robot is DONE exploring")
                continue
            
            if not robot.is_active or not robot.can_reach(waypoint):
                continue
            
            # Calculate travel time
            distance_coords = np.linalg.norm(waypoint - robot.position)
            if robot.environment is not None:
                distance_meters = robot.environment.coord_to_meters(distance_coords)
            else:
                distance_meters = distance_coords
            
            travel_time = distance_meters / robot.max_speed
            
            # Check if robot has enough budget for travel + measurement
            if robot.budget_type == BudgetType.TIME:
                required_budget = travel_time + self.sensor_time
            elif robot.budget_type == BudgetType.DISTANCE:
                required_budget = distance_meters + (self.sensor_time * robot.max_speed)
            else:
                required_budget = travel_time + self.sensor_time
            
            if robot.remaining_budget < required_budget:
                # Not enough budget, robot is done
                if verbose:
                    print(f"\n[Time {self.simulation_clock:.1f}s] Robot {robot_id} - Insufficient budget")
                    print(f"  Required: {required_budget:.1f}s, Remaining: {robot.remaining_budget:.1f}s")
                    print(f"  Robot is DONE exploring")
                continue
            
            # Consume travel time from budget
            if robot.budget_type == BudgetType.TIME:
                robot.consume_budget(travel_time)
            elif robot.budget_type == BudgetType.DISTANCE:
                robot.consume_budget(distance_meters)
            
            # Schedule arrival event
            arrival_time = self.simulation_clock + travel_time
            heapq.heappush(
                self.event_queue,
                RobotEvent(arrival_time, robot_id, waypoint.copy())
            )
            
            # Robot is no longer idle
            self.idle_robots.discard(robot_id)
            
            if verbose:
                print(f"\n[Time {self.simulation_clock:.1f}s] Robot {robot_id} - NEW WAYPOINT")
                print(f"  Current: {robot.position}")
                print(f"  Target: {waypoint}")
                print(f"  Distance: {distance_meters:.2f}m")
                print(f"  Travel time: {travel_time:.1f}s")
                print(f"  Arrival: {arrival_time:.1f}s")
                print(f"  Budget after travel: {robot.remaining_budget:.1f}s")
    
    def _take_measurement(self, robot: Robot, position: np.ndarray, verbose: bool = False) -> None:
        """Take measurement at position and update belief."""
        # Get measurement from environment
        observation = self.environment.observe(position.reshape(1, -1))[0]
        
        # Add to robot's measurement history
        robot.add_measurement(position, observation, self.simulation_clock)
        
        # Consume sensor time from budget
        if robot.budget_type == BudgetType.TIME:
            robot.consume_budget(self.sensor_time)
        elif robot.budget_type == BudgetType.DISTANCE:
            # For distance budget, sensor time consumes "distance" based on speed
            sensor_distance = self.sensor_time * robot.max_speed
            robot.consume_budget(sensor_distance)
        
        # Update shared GP belief
        if self.gp_belief is not None:
            self.gp_belief.update(
                position.reshape(1, -1),
                np.array([observation])
            )
        
        # Update stats
        self.stats['measurements_taken'][robot.id] += 1
        
        if verbose:
            print(f"\n[Time {self.simulation_clock:.1f}s] Robot {robot.id} - MEASUREMENT")
            print(f"  Position: {position}")
            print(f"  Value: {observation:.3f}")
            print(f"  Sensor time: {self.sensor_time:.1f}s")
            print(f"  Total measurements: {self.stats['measurements_taken'][robot.id]}")
    
    def _has_active_robots(self) -> bool:
        """Check if any robot still has budget."""
        return any(robot.is_active for robot in self.robots)
    
    def _get_robot(self, robot_id: int) -> Robot:
        """Get robot by ID."""
        for robot in self.robots:
            if robot.id == robot_id:
                return robot
        raise ValueError(f"Robot {robot_id} not found")
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final mission results."""
        results = {
            'stats': self.stats.copy(),
            'robot_trajectories': {
                robot.id: robot.get_trajectory_positions()
                for robot in self.robots
            },
            'robot_measurements': {
                robot.id: robot.measurements
                for robot in self.robots
            },
            'total_measurements': sum(self.stats['measurements_taken'].values()),
            'total_distance': sum(self.stats['total_distance'].values()),
        }
        
        # Add GP belief if available
        if self.gp_belief is not None:
            results['final_belief'] = self.gp_belief
        
        return results
    
    def reset(self):
        """Reset planner and all robots to initial state."""
        for robot in self.robots:
            robot.reset()
        
        self.simulation_clock = 0.0
        self.event_queue = []
        self.idle_robots = set(robot.id for robot in self.robots)
        
        self.stats = {
            'total_distance': {robot.id: 0.0 for robot in self.robots},
            'measurements_taken': {robot.id: 0 for robot in self.robots},
            'events_processed': 0,
        }
