"""
Sequential Greedy Information Gain Planner.

Implements multi-robot greedy planning via Sequential Allocation:
1. Plan a full path for robot 1 greedily using IG/cost
2. Apply Kriging Believer (KB) virtual updates to GP for those planned measurements
3. Plan a path for robot 2 on the residual information
4. Continue for all robots

This is a canonical multi-robot information-driven baseline that explicitly
coordinates robots through virtual measurements.

The planner maintains a persistent "believer" GP that accumulates virtual updates
for all planned waypoints. When robots reach waypoints and take real measurements,
the virtual observations are replaced with actual data.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set

from .base_planner import BaseMultiRobotPlanner
from ..core.robot import Robot
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


class SequentialGreedyIGPlanner(BaseMultiRobotPlanner):
    """
    Sequential greedy planner with Kriging Believer coordination.
    
    Strategy:
    1. For each robot in sequence:
       a. Greedily select next waypoint with max IG/cost
       b. Move robot and collect measurement
       c. Apply KB update to shared belief
    2. Robots coordinate through KB virtual observations
    3. Later robots plan on residual uncertainty
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: GaussianProcessBelief,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sequential greedy planner.
        
        Args:
            robots: List of Robot instances
            environment: Environment to explore
            gp_belief: Shared GP belief (REQUIRED for information gain)
            config: Configuration with optional keys:
                - candidate_resolution: Grid resolution for candidates
                - seed: RNG seed for deterministic tie-breaking
        """
        if gp_belief is None:
            raise ValueError("SequentialGreedyIGPlanner requires a GP belief!")

        super().__init__(robots, environment, gp_belief, config)
        cfg = self.config

        self.candidate_resolution = cfg.get('candidate_resolution', 20)
        self.rng = np.random.RandomState(cfg.get('seed', None))

        # Persistent Kriging Believer GP that accumulates virtual updates
        self.gp_believer = None
        self.planned_waypoints: Dict[int, np.ndarray] = {}
        self.reserved_targets: Dict[Tuple[float, float], int] = {}
        self.completed_targets: Set[Tuple[float, float]] = set()
        self.robot_candidate_sets: Dict[int, np.ndarray] = {}

        self._generate_candidate_grid()
    
    def set_robot_candidates(self, candidates: Dict[int, np.ndarray]) -> None:
        """Optionally supply per-robot candidate sets (matches assignment Step A)."""
        self.robot_candidate_sets = {
            rid: np.array(points, copy=True)
            for rid, points in candidates.items()
            if points is not None and len(points) > 0
        }

    def _generate_candidate_grid(self):
        """Generate grid of candidate positions for evaluation."""
        bounds = self.environment.bounds
        
        x = np.linspace(bounds[0, 0], bounds[0, 1], self.candidate_resolution)
        y = np.linspace(bounds[1, 0], bounds[1, 1], self.candidate_resolution)
        
        X, Y = np.meshgrid(x, y)
        self.candidate_grid = np.c_[X.ravel(), Y.ravel()]
    
    def _compute_variance(
        self,
        candidate: np.ndarray,
        belief: GaussianProcessBelief
    ) -> float:
        """Compute IG of a candidate under the provided belief (using variance as proxy)."""
        candidate_2d = candidate.reshape(1, -1)
        # Use variance as proxy for information gain (much faster than full variance reduction)
        _, std = belief.predict(candidate_2d, return_std=True)
        return std[0] ** 2  # Return variance
    
    def _select_greedy_waypoint(
        self,
        robot: Robot,
        belief: GaussianProcessBelief,
        blocked_positions: Set[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """Select the best reachable candidate for a robot under belief."""
        best_score = -np.inf
        best_candidate: Optional[np.ndarray] = None
        
        candidates = self.robot_candidate_sets.get(robot.id, self.candidate_grid)
        if candidates is None or len(candidates) == 0:
            return None

        # Get visited positions from robot's trajectory
        visited_positions = {tuple(state.position) for state in robot.trajectory}

        for candidate in candidates:
            # Skip if already visited
            if tuple(candidate) in visited_positions:
                continue
            candidate_key = tuple(candidate)

            # Skip if another robot already reserved or completed this target
            if candidate_key in blocked_positions:
                continue
                
            if not robot.can_reach(candidate):
                continue

            distance_coords = np.linalg.norm(candidate - robot.position)
            if distance_coords < 1e-6:
                continue
            if self.environment is not None:
                distance_meters = self.environment.coord_to_meters(distance_coords)
            else:
                distance_meters = distance_coords

            var = self._compute_variance(candidate, belief)
            if distance_meters <= 0:
                continue

            score = var / distance_meters
            if not np.isfinite(score):
                continue

            if score > best_score + 1e-12:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None
        
        return best_candidate

    def _sync_completed_targets(self) -> None:
        """Remove reservations for robots that just completed their waypoints."""
        completed_robot_ids = []
        for robot_id, waypoint in list(self.planned_waypoints.items()):
            robot = self._get_robot(robot_id)
            if np.allclose(robot.position, waypoint, atol=1e-6):
                completed_robot_ids.append(robot_id)

        if not completed_robot_ids:
            return

        for robot_id in completed_robot_ids:
            waypoint = self.planned_waypoints.pop(robot_id)
            waypoint_key = tuple(waypoint)
            self.reserved_targets.pop(waypoint_key, None)
            self.completed_targets.add(waypoint_key)

        self._refresh_believer()

    def _refresh_believer(self) -> None:
        """Reset believer GP to actual GP plus all reserved targets."""
        self.gp_believer = self.gp_belief.copy()

        if not self.reserved_targets:
            return

        target_positions = np.array(list(self.reserved_targets.keys()))
        means, _ = self.gp_belief.predict(target_positions, return_std=True)
        self.gp_believer.update(target_positions, means)

    def _reserve_target(self, robot_id: int, waypoint: np.ndarray) -> None:
        """Reserve waypoint for robot and refresh believer with updated KB state."""
        self.planned_waypoints[robot_id] = waypoint.copy()
        self.reserved_targets[tuple(waypoint)] = robot_id
        self._refresh_believer()
    
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Plan next waypoint for each idle robot sequentially with persistent KB coordination.
        
        Maintains a persistent believer GP that accumulates virtual updates for all
        planned waypoints across the entire mission, matching KrigingBelieverAssignment behavior.
        
        Returns:
            Dictionary mapping robot_id -> next_position
        """
        waypoints: Dict[int, np.ndarray] = {}

        if self.gp_belief is None:
            return waypoints

        # Drop reservations for robots that just finished measuring
        self._sync_completed_targets()

        # Initialize believer GP on first call or after reset
        if self.gp_believer is None:
            self._refresh_believer()

        blocked_positions = set(self.reserved_targets.keys()) | self.completed_targets

        # Plan for each idle robot sequentially using the persistent believer
        for robot in self.robots:
            if robot.id not in self.idle_robots:
                continue

            if not robot.is_active:
                continue

            waypoint = self._select_greedy_waypoint(robot, self.gp_believer, blocked_positions)

            if waypoint is None:
                continue

            waypoints[robot.id] = waypoint
            waypoint_key = tuple(waypoint)
            blocked_positions.add(waypoint_key)

            # Reserve target globally and refresh believer to match KrigingBelieverAssignment
            self._reserve_target(robot.id, waypoint)

        return waypoints
    
    def reset(self):
        """Reset planner state for new mission."""
        super().reset()
        # Clear persistent believer state
        self.gp_believer = None
        self.planned_waypoints = {}
        self.reserved_targets = {}
        self.completed_targets = set()
        # Regenerate candidate grid if bounds changed
        self._generate_candidate_grid()
    
    def get_planner_info(self) -> Dict[str, Any]:
        """Return information about planner configuration."""
        return {
            'planner_name': 'SequentialGreedyIGPlanner',
            'candidate_resolution': self.candidate_resolution,
            'n_robots': len(self.robots),
            'n_candidates': len(self.candidate_grid)
        }
