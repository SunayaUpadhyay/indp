"""
Auction-Based Multi-Robot Allocation Planner.

Uses a centralized auction mechanism to assign high-value targets to robots.
Each robot bids on candidates based on utility/cost ratio.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .base_planner import BaseMultiRobotPlanner
from ..core.robot import Robot
from ..core.environment import Environment
from ..core.belief import GaussianProcessBelief


class AuctionVariancePlanner(BaseMultiRobotPlanner):
    """
    Auction-based planner for multi-robot coordination.
    
    Strategy:
    1. Generate candidate set (top-K highest variance cells)
    2. Each robot bids on each candidate: bid = utility / cost
    3. Assign candidates via auction (highest bidder wins)
    4. Robots move to assigned targets
    5. Replan periodically
    """
    
    def __init__(
        self,
        robots: List[Robot],
        environment: Environment,
        gp_belief: GaussianProcessBelief,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize enhanced auction planner with configurable behavior."""
        if gp_belief is None:
            raise ValueError("AuctionVariancePlanner requires a GP belief!")
        
        super().__init__(robots, environment, gp_belief, config)
        cfg = self.config
        
        # Candidate generation / filtering
        self.num_candidates = int(cfg.get('num_candidates', 50))
        self.pool_factor = max(1.0, float(cfg.get('pool_factor', 5.0)))
        self.variance_threshold_frac = float(cfg.get('variance_threshold_frac', 0.2))
        self.diversity_min_dist = float(cfg.get('diversity_min_dist', 0.0))
        self.visited_penalty_weight = float(cfg.get('visited_penalty_weight', 0.0))
        self.grid_resolution = int(cfg.get('grid_resolution', 50))
        
        # Utility / cost definition
        self.use_variance_only = bool(cfg.get('use_variance_only', False))
        self.normalize_utility = bool(cfg.get('normalize_utility', True))
        self.normalize_cost = bool(cfg.get('normalize_cost', True))
        self.utility_exponent = float(cfg.get('utility_exponent', 1.0))
        self.cost_exponent = float(cfg.get('cost_exponent', 1.0))
        self.min_cost_meters = float(cfg.get('min_cost_meters', 1.0))
        self.distance_units = cfg.get('distance_units', 'meters').lower()
        if self.distance_units not in {'meters', 'coords'}:
            raise ValueError("distance_units must be either 'meters' or 'coords'")
        
        # Auction / replanning behavior
        self.replan_interval_steps = int(cfg.get('replan_interval_steps', cfg.get('replan_interval', 10)))
        self.min_idle_fraction_for_replan = float(cfg.get('min_idle_fraction_for_replan', 0.3))
        self.max_assignment_age_steps = int(cfg.get('max_assignment_age_steps', 100))
        self.min_distance_to_target = float(cfg.get('min_distance_to_target', 1.0))
        
        # Believer / redundancy reduction
        self.use_believer_updates = bool(cfg.get('use_believer_updates', True))
        self.believer_mode = cfg.get('believer_mode', 'per_auction')
        
        # Diagnostics
        self.verbose = bool(cfg.get('verbose', False))
        self.log_stats_interval = int(cfg.get('log_stats_interval', 20))
        
        # Initialize random number generator
        seed = cfg.get('seed', None)
        self.rng = np.random.RandomState(seed)
        
        # Planner state
        self.steps_since_auction = 0
        self.global_step = 0
        self.assignments = {robot.id: None for robot in robots}
        self.assignment_targets: Dict[int, Optional[np.ndarray]] = {robot.id: None for robot in robots}
        self.assignment_age = {robot.id: 0 for robot in robots}
        self.assignment_changes = {robot.id: 0 for robot in robots}
        self.num_auctions_run = 0
        self.last_num_candidates = 0
        self.last_bid_stats: Dict[str, Any] = {}
        self.current_candidates: Optional[np.ndarray] = None
        self.current_candidate_variances: Optional[np.ndarray] = None
        self.candidate_min_dist_to_visited: Optional[np.ndarray] = None
        self.believer_gp: Optional[GaussianProcessBelief] = None
        
        # Generate evaluation grid
        self._generate_evaluation_grid()
    
    def _generate_evaluation_grid(self):
        """Generate grid for variance evaluation."""
        bounds = self.environment.bounds
        
        x = np.linspace(bounds[0, 0], bounds[0, 1], self.grid_resolution)
        y = np.linspace(bounds[1, 0], bounds[1, 1], self.grid_resolution)
        
        X, Y = np.meshgrid(x, y)
        self.eval_grid = np.c_[X.ravel(), Y.ravel()]
    
    def _generate_candidates(
        self,
        planning_belief: GaussianProcessBelief
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate a diverse, high-variance candidate set with caching support."""
        _, std = planning_belief.predict(self.eval_grid, return_std=True)
        variance = std ** 2 if std is not None else np.zeros(len(self.eval_grid))
        
        if len(variance) == 0:
            return np.empty((0, 2)), np.empty(0), None
        
        max_var = float(np.max(variance))
        if max_var <= 0:
            var_mask = np.ones_like(variance, dtype=bool)
        else:
            threshold = self.variance_threshold_frac * max_var
            var_mask = variance >= threshold
        
        candidate_pool = self.eval_grid[var_mask]
        variance_pool = variance[var_mask]
        
        if len(candidate_pool) == 0:
            candidate_pool = self.eval_grid.copy()
            variance_pool = variance.copy()
        
        pool_size = max(1, int(self.pool_factor * max(1, self.num_candidates)))
        if len(candidate_pool) < pool_size:
            top_indices = np.argsort(variance)[-pool_size:]
            candidate_pool = self.eval_grid[top_indices]
            variance_pool = variance[top_indices]
        else:
            top_indices = np.argsort(variance_pool)[-pool_size:]
            candidate_pool = candidate_pool[top_indices]
            variance_pool = variance_pool[top_indices]
        
        order = np.argsort(-variance_pool)
        selected_indices: List[int] = []
        if len(order) > 0:
            selected_indices.append(int(order[0]))
        
        def _select_best_candidate(enforce_threshold: bool) -> Optional[int]:
            best_idx = None
            best_score = -np.inf
            for idx in order:
                if idx in selected_indices:
                    continue
                if not selected_indices:
                    return int(idx)
                dists = np.linalg.norm(
                    candidate_pool[idx] - candidate_pool[selected_indices],
                    axis=1
                )
                min_dist = float(np.min(dists)) if dists.size else np.inf
                if enforce_threshold and self.diversity_min_dist > 0 and min_dist < self.diversity_min_dist:
                    continue
                if min_dist > best_score:
                    best_score = min_dist
                    best_idx = int(idx)
            return best_idx
        
        desired = min(self.num_candidates, len(candidate_pool))
        while len(selected_indices) < desired:
            candidate_idx = _select_best_candidate(enforce_threshold=True)
            if candidate_idx is None:
                candidate_idx = _select_best_candidate(enforce_threshold=False)
            if candidate_idx is None:
                break
            selected_indices.append(candidate_idx)
        
        selected_indices = sorted(set(selected_indices), key=lambda i: -variance_pool[i])
        candidates = candidate_pool[selected_indices]
        candidate_variances = variance_pool[selected_indices]
        min_distance_to_visited = self._compute_min_distance_to_visited(candidates)
        return candidates, candidate_variances, min_distance_to_visited
    
    def _compute_bids(
        self,
        auction_robots: List[Robot],
        planning_belief: GaussianProcessBelief
    ) -> Tuple[Dict[int, Dict[int, float]], Dict[str, Any]]:
        """Compute bids and accompanying diagnostics for the provided robots."""
        if self.current_candidates is None or len(self.current_candidates) == 0:
            return {}, {'mean_bid': 0.0, 'max_bid': 0.0, 'num_bids': 0, 'num_unreachable': 0}
        
        # Utility calculation (shared across robots)
        if self.use_variance_only and self.current_candidate_variances is not None:
            utilities = self.current_candidate_variances.copy()
        else:
            utilities = []
            for candidate in self.current_candidates:
                candidate_2d = candidate.reshape(1, -1)
                utility = planning_belief.variance_reduction(
                    X_candidate=candidate_2d,
                    X_eval=self.eval_grid
                )
                utilities.append(utility)
            utilities = np.array(utilities)
        
        if self.visited_penalty_weight > 0 and self.candidate_min_dist_to_visited is not None:
            penalties = self.visited_penalty_weight / (1.0 + self.candidate_min_dist_to_visited)
            utilities = np.maximum(utilities - penalties, 0.0)
        else:
            utilities = np.maximum(utilities, 0.0)
        
        if self.normalize_utility and np.max(utilities) > 0:
            utilities = utilities / (np.max(utilities) + 1e-8)
        
        # Cost computation per robot
        num_candidates = len(self.current_candidates)
        robot_costs: Dict[int, np.ndarray] = {}
        max_cost = 0.0
        num_unreachable = 0
        for robot in auction_robots:
            costs = np.full(num_candidates, np.inf)
            for idx, candidate in enumerate(self.current_candidates):
                if not robot.can_reach(candidate):
                    continue
                distance_coords = np.linalg.norm(candidate - robot.position)
                if self.distance_units == 'meters' and hasattr(self.environment, 'coord_to_meters'):
                    distance = self.environment.coord_to_meters(distance_coords)
                else:
                    distance = distance_coords
                distance = max(distance, self.min_cost_meters)
                costs[idx] = distance
                max_cost = max(max_cost, distance)
            robot_costs[robot.id] = costs
        
        if self.normalize_cost and max_cost > 0:
            for robot_id, costs in robot_costs.items():
                mask = np.isfinite(costs)
                costs[mask] = costs[mask] / (max_cost + 1e-8)
                robot_costs[robot_id] = costs
        
        bids: Dict[int, Dict[int, float]] = {}
        bid_values: List[float] = []
        for robot in auction_robots:
            robot_bids: Dict[int, float] = {}
            costs = robot_costs[robot.id]
            for cand_idx in range(num_candidates):
                cost = costs[cand_idx]
                if not np.isfinite(cost):
                    robot_bids[cand_idx] = -np.inf
                    num_unreachable += 1
                    continue
                numerator = utilities[cand_idx] ** self.utility_exponent
                denominator = (cost ** self.cost_exponent) + 1e-8
                bid_value = numerator / denominator if denominator > 0 else 0.0
                robot_bids[cand_idx] = bid_value
                bid_values.append(bid_value)
            bids[robot.id] = robot_bids
        
        bid_stats = {
            'mean_bid': float(np.mean(bid_values)) if bid_values else 0.0,
            'max_bid': float(np.max(bid_values)) if bid_values else 0.0,
            'num_bids': len(bid_values),
            'num_unreachable': int(num_unreachable),
        }
        return bids, bid_stats
    
    def _run_auction(
        self,
        auction_robots: List[Robot],
        bids: Dict[int, Dict[int, float]]
    ) -> Dict[int, Optional[int]]:
        """
        Run auction to assign candidates to robots.
        
        Greedy auction: repeatedly assign highest-bid (robot, candidate) pair.
        
        Returns candidate indices for the participating robots.
        """
        assignments = {robot.id: None for robot in auction_robots}
        assigned_candidates = set()
        assigned_robots = set()
        
        # Build list of (bid, robot_id, candidate_idx) tuples
        bid_list = []
        for robot in auction_robots:
            robot_id = robot.id
            robot_bids = bids.get(robot_id, {})
            for cand_idx, bid_value in robot_bids.items():
                if bid_value > -np.inf:
                    bid_list.append((bid_value, robot_id, cand_idx))
        
        # Sort by bid value (descending)
        bid_list.sort(reverse=True, key=lambda x: x[0])
        
        # Assign greedily
        for bid_value, robot_id, cand_idx in bid_list:
            # Skip if robot or candidate already assigned
            if robot_id in assigned_robots or cand_idx in assigned_candidates:
                continue
            
            # Assign
            assignments[robot_id] = cand_idx
            assigned_robots.add(robot_id)
            assigned_candidates.add(cand_idx)
            
            # Stop if all robots assigned
            if len(assigned_robots) == len(auction_robots):
                break
        
        return assignments
    
    def plan_step(self) -> Dict[int, np.ndarray]:
        """
        Plan next waypoint using auction mechanism for idle robots.
        
        Replanning cadence and candidate regeneration are governed by
        replan_interval_steps and min_idle_fraction_for_replan.
        """
        self.global_step += 1
        waypoints: Dict[int, np.ndarray] = {}
        active_robots = [r for r in self.robots if r.is_active]
        active_count = max(1, len(active_robots))
        
        # Update assignment ages and clear invalid targets
        for robot in self.robots:
            assigned_target = self.assignment_targets.get(robot.id)
            if assigned_target is None:
                continue
            self.assignment_age[robot.id] += 1
            distance_to_target = np.linalg.norm(assigned_target - robot.position)
            should_drop = (
                distance_to_target < self.min_distance_to_target
                or not robot.can_reach(assigned_target)
                or self.assignment_age[robot.id] > self.max_assignment_age_steps
            )
            if should_drop:
                self.assignment_targets[robot.id] = None
                self.assignments[robot.id] = None
                self.assignment_age[robot.id] = 0
        
        idle_robot_objs = [
            r for r in self.robots
            if r.is_active and r.id in self.idle_robots
        ]
        unassigned_idle = [r for r in idle_robot_objs if self.assignment_targets[r.id] is None]
        idle_fraction = len(idle_robot_objs) / active_count
        ran_auction = False
        new_assignments = 0
        
        if unassigned_idle:
            auction_scope_full = (
                self.steps_since_auction >= self.replan_interval_steps
                or idle_fraction >= self.min_idle_fraction_for_replan
                or self.current_candidates is None
            )
            auction_robots = idle_robot_objs if auction_scope_full else unassigned_idle
            planning_belief = self._get_planning_belief()
            candidates, candidate_variances, min_dist = self._generate_candidates(planning_belief)
            self.current_candidates = candidates
            self.current_candidate_variances = candidate_variances
            self.candidate_min_dist_to_visited = min_dist
            self.last_num_candidates = len(candidates)
            if len(candidates) > 0:
                bids, bid_stats = self._compute_bids(auction_robots, planning_belief)
                assignments = self._run_auction(auction_robots, bids)
                for robot_id in self.assignments.keys():
                    self.assignments[robot_id] = None
                new_assignments = self._apply_assignments(assignments)
                self.last_bid_stats = bid_stats
                self.steps_since_auction = 0
                self.num_auctions_run += 1
                ran_auction = True
                if self.use_believer_updates:
                    self._refresh_believer_model()
            else:
                self.last_bid_stats = {'mean_bid': 0.0, 'max_bid': 0.0, 'num_bids': 0, 'num_unreachable': 0}
        
        for robot in idle_robot_objs:
            assigned_target = self.assignment_targets.get(robot.id)
            if assigned_target is not None:
                waypoints[robot.id] = assigned_target.copy()
        
        if not ran_auction:
            self.steps_since_auction += 1
        
        self._maybe_log_stats(new_assignments, len(idle_robot_objs))
        return waypoints
    
    def reset(self):
        """Reset planner state for new mission."""
        super().reset()
        self.steps_since_auction = 0
        self.global_step = 0
        self.assignments = {robot.id: None for robot in self.robots}
        self.assignment_targets = {robot.id: None for robot in self.robots}
        self.assignment_age = {robot.id: 0 for robot in self.robots}
        self.assignment_changes = {robot.id: 0 for robot in self.robots}
        self.num_auctions_run = 0
        self.last_num_candidates = 0
        self.last_bid_stats = {}
        self.current_candidates = None
        self.current_candidate_variances = None
        self.candidate_min_dist_to_visited = None
        self.believer_gp = None
        self._generate_evaluation_grid()
    
    def _compute_min_distance_to_visited(self, candidates: np.ndarray) -> Optional[np.ndarray]:
        if self.visited_penalty_weight <= 0 or len(candidates) == 0:
            return None
        visited_positions = self._collect_visited_positions()
        if visited_positions is None:
            return None
        diffs = candidates[:, None, :] - visited_positions[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        return np.min(distances, axis=1)
    
    def _collect_visited_positions(self) -> Optional[np.ndarray]:
        visited: List[np.ndarray] = []
        for robot in self.robots:
            if not robot.measurements:
                continue
            visited.extend(meas[0] for meas in robot.measurements)
        if not visited:
            return None
        return np.vstack(visited)
    
    def _apply_assignments(self, assignment_map: Dict[int, Optional[int]]) -> int:
        if self.current_candidates is None:
            return 0
        updates = 0
        for robot_id, cand_idx in assignment_map.items():
            if cand_idx is None:
                continue
            target = self.current_candidates[cand_idx].copy()
            previous = self.assignment_targets.get(robot_id)
            if previous is None or not np.allclose(previous, target):
                self.assignment_changes[robot_id] += 1
            self.assignment_targets[robot_id] = target
            self.assignments[robot_id] = cand_idx
            self.assignment_age[robot_id] = 0
            updates += 1
        return updates
    
    def _get_planning_belief(self) -> GaussianProcessBelief:
        if self.use_believer_updates and self.believer_gp is not None:
            return self.believer_gp
        return self.gp_belief
    
    def _refresh_believer_model(self) -> None:
        if not self.use_believer_updates or self.gp_belief is None:
            self.believer_gp = None
            return
        assigned_points = [target for target in self.assignment_targets.values() if target is not None]
        believer = self.gp_belief.copy()
        if assigned_points:
            points = np.vstack(assigned_points)
            believer.kriging_believer_update(points, inplace=True)
        self.believer_gp = believer
    
    def _maybe_log_stats(self, new_assignments: int, idle_count: int) -> None:
        if not self.verbose:
            return
        if self.global_step % max(1, self.log_stats_interval) != 0:
            return
        stats = self.last_bid_stats or {}
        print(
            f"[AuctionPlanner] step={self.global_step} idle={idle_count} "
            f"candidates={self.last_num_candidates} new_assignments={new_assignments} "
            f"mean_bid={stats.get('mean_bid', 0.0):.4f} max_bid={stats.get('max_bid', 0.0):.4f}"
        )
    
    def get_planner_info(self) -> Dict[str, Any]:
        """Return information about planner configuration."""
        return {
            'planner_name': 'AuctionVariancePlanner',
            'num_candidates': self.num_candidates,
            'pool_factor': self.pool_factor,
            'replan_interval_steps': self.replan_interval_steps,
            'replan_interval': self.replan_interval_steps,
            'grid_resolution': self.grid_resolution,
            'n_robots': len(self.robots),
            'utility_exponent': self.utility_exponent,
            'cost_exponent': self.cost_exponent,
            'normalize_utility': self.normalize_utility,
            'normalize_cost': self.normalize_cost,
            'num_auctions_run': self.num_auctions_run,
            'last_num_candidates': self.last_num_candidates,
            'last_bid_stats': self.last_bid_stats,
            'assignment_changes': dict(self.assignment_changes),
        }
