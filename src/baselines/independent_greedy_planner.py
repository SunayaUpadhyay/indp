"""
Independent Greedy Information Gain Planner.

Each robot independently runs greedy IG without Kriging Believer coordination.
All robots share the same underlying GP updated only with real measurements.

This is a pure baseline with NO coordination between robots - each robot greedily
selects waypoints based only on the shared GP belief without considering other
robots' plans. This typically leads to redundant exploration as robots ignore
each other and may converge on the same high-information regions.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from .base_planner import BaseMultiRobotPlanner
from ..core.robot import Robot
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


class IndependentGreedyIGPlanner(BaseMultiRobotPlanner):
    """
    Truly independent greedy planner without any coordination.
    
    Strategy:
    1. All robots independently select waypoints with max IG/cost
    2. No Kriging Believer updates (no virtual observations)
    3. No explicit coordination - robots may select redundant locations
    4. Coordination only through real measurements in shared GP
    
    This represents the simplest multi-robot greedy strategy and typically
    leads to redundant exploration since robots ignore each other's plans.
    The GP's variance reduction naturally downweights already-sampled regions,
    but this is often insufficient to prevent multiple robots from targeting
    the same high-information areas.
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: GaussianProcessBelief,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize independent greedy planner.
        
        Args:
            robots: List of Robot instances
            environment: Environment to explore
            gp_belief: Shared GP belief (REQUIRED for information gain)
            config: Configuration with optional keys:
                - candidate_resolution: Grid resolution for candidates (default: 20)
                - seed: Optional RNG seed for deterministic tie-breaking
        """
        if gp_belief is None:
            raise ValueError("IndependentGreedyIGPlanner requires a GP belief!")
        
        super().__init__(robots, environment, gp_belief, config)
        cfg = self.config
        
        self.candidate_resolution = cfg.get('candidate_resolution', 20)
        self.rng = np.random.RandomState(cfg.get('seed', None))
        
        self._generate_candidate_grid()
    
    def _generate_candidate_grid(self):
        """Generate grid of candidate positions for evaluation."""
        bounds = self.environment.bounds
        
        x = np.linspace(bounds[0, 0], bounds[0, 1], self.candidate_resolution)
        y = np.linspace(bounds[1, 0], bounds[1, 1], self.candidate_resolution)
        
        X, Y = np.meshgrid(x, y)
        self.candidate_grid = np.c_[X.ravel(), Y.ravel()]
    
    def _precompute_information_gains(self) -> np.ndarray:
        """Approximate IG via per-candidate variance (cheap single GP call)."""
        # Evaluate variance for the whole grid in one GP query for efficiency
        _, std = self.gp_belief.predict(self.candidate_grid, return_std=True)
        if std is None:
            return np.zeros(len(self.candidate_grid))
        return std ** 2
    
    def _select_greedy_waypoint(
        self,
        robot: Robot,
        ig_values: np.ndarray
    ) -> Optional[np.ndarray]:
        """Pick the reachable candidate with the best IG/cost score."""
        best_score = -np.inf
        best_candidates = []

        for idx, candidate in enumerate(self.candidate_grid):
            if not robot.can_reach(candidate):
                continue

            cost = np.linalg.norm(candidate - robot.position)
            if cost < 1e-6:
                continue

            ig = ig_values[idx]
            if cost <= 0:
                continue

            score = ig / cost
            if not np.isfinite(score):
                continue

            if score > best_score + 1e-12:
                best_score = score
                best_candidates = [candidate]
            elif abs(score - best_score) <= 1e-12:
                best_candidates.append(candidate)

        if not best_candidates:
            return None

        if len(best_candidates) == 1:
            return best_candidates[0]

        # Tie-break deterministically using RNG if available
        choice = self.rng.randint(len(best_candidates)) if hasattr(self, 'rng') else 0
        return best_candidates[choice]
    
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Plan next waypoint for each idle robot independently.
        
        All robots plan using the SAME current belief (no KB updates).
        No coordination between robots - each independently selects the
        greedy waypoint from its own position. This often leads to 
        redundant exploration as multiple robots may target the same
        high-information regions.
        
        Returns:
            Dictionary mapping robot_id -> next_position
        """
        waypoints = {}
        
        # Precompute IG once per planning call (shared snapshot of belief)
        ig_values = self._precompute_information_gains()
        
        # Each idle robot plans independently using shared belief
        for robot in self.robots:
            # Only plan for idle robots
            if robot.id not in self.idle_robots:
                continue
            
            if not robot.is_active:
                continue
            
            # Select greedy waypoint (no coordination with other robots)
            waypoint = self._select_greedy_waypoint(robot, ig_values)
            
            if waypoint is not None:
                waypoints[robot.id] = waypoint
        
        return waypoints
    
    def reset(self):
        """Reset planner state for new mission."""
        super().reset()
        # Regenerate candidate grid if bounds changed
        self._generate_candidate_grid()
    
    def get_planner_info(self) -> Dict[str, Any]:
        """Return information about planner configuration."""
        return {
            'planner_name': 'IndependentGreedyIGPlanner',
            'candidate_resolution': self.candidate_resolution,
            'n_robots': len(self.robots),
            'n_candidates': len(self.candidate_grid)
        }
