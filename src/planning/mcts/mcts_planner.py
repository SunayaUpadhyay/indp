"""
Monte Carlo Tree Search planner for informative path planning.

This module implements MCTS to find optimal sampling sequences within
a robot's candidate window, balancing:
- Information gain (reducing GP uncertainty)
- Travel efficiency (minimizing path length/time)
- Budget constraints (respecting robot limitations)
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from copy import deepcopy

from ...core.robot import Robot
from ...core.belief import GaussianProcessBelief
from ..candidates.candidate_generator import CandidateSet


@dataclass
class MCTSConfig:
    """Configuration for MCTS planner."""
    iterations: int = 1000  # Number of MCTS iterations
    exploration_constant: float = 1.414  # UCB exploration parameter (sqrt(2))
    max_depth: int = 10  # Maximum planning depth
    discount_factor: float = 0.95  # Reward discount for future steps
    simulation_depth: int = 5  # Depth for rollout simulations
    use_progressive_widening: bool = True  # Enable progressive widening
    pw_alpha: float = 0.5  # Progressive widening: children = visits^alpha
    pw_constant: float = 1.0  # Progressive widening constant
    time_limit: Optional[float] = None  # Time limit for planning (seconds)
    verbose: bool = False


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Represents a state in the planning problem:
    - Current robot position
    - Remaining budget
    - GP belief state
    - Visited locations in this path
    """
    
    def __init__(
        self,
        position: np.ndarray,
        remaining_budget: float,
        gp_belief: GaussianProcessBelief,
        visited_positions: List[np.ndarray],
        parent: Optional['MCTSNode'] = None,
        action: Optional[np.ndarray] = None,
        depth: int = 0
    ):
        """
        Initialize MCTS node.
        
        Args:
            position: Current position [x, y]
            remaining_budget: Remaining robot budget
            gp_belief: Current GP belief state
            visited_positions: List of positions visited so far in this path
            parent: Parent node (None for root)
            action: Action (target position) that led to this node
            depth: Depth in the tree
        """
        self.position = np.array(position)
        self.remaining_budget = remaining_budget
        self.gp_belief = gp_belief
        self.visited_positions = list(visited_positions)
        self.parent = parent
        self.action = action
        self.depth = depth
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.children: List[MCTSNode] = []
        self.untried_actions: List[np.ndarray] = []
        self.is_terminal = False
        
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        
        UCB1 = exploitation + exploration
             = (Q/N) + c * sqrt(ln(parent_N) / N)
        """
        return max(
            self.children,
            key=lambda child: (
                child.total_reward / child.visits +
                exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            )
        )
    
    def add_child(
        self,
        action: np.ndarray,
        position: np.ndarray,
        remaining_budget: float,
        gp_belief: GaussianProcessBelief,
        visited_positions: List[np.ndarray]
    ) -> 'MCTSNode':
        """Add a child node for the given action."""
        child = MCTSNode(
            position=position,
            remaining_budget=remaining_budget,
            gp_belief=gp_belief,
            visited_positions=visited_positions,
            parent=self,
            action=action,
            depth=self.depth + 1
        )
        self.children.append(child)
        return child


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for informative path planning.
    
    Uses MCTS to find sequences of sampling locations that maximize
    information gain while respecting budget constraints.
    
    Key features:
    - UCB1 for balancing exploration/exploitation
    - Progressive widening to handle continuous action spaces
    - Information-theoretic rewards based on GP uncertainty reduction
    - Budget-aware state transitions
    """
    
    def __init__(self, config: MCTSConfig = None):
        """
        Initialize MCTS planner.
        
        Args:
            config: MCTS configuration (uses defaults if None)
        """
        self.config = config or MCTSConfig()
        self.robot: Optional[Robot] = None
        self.candidates: Optional[CandidateSet] = None
        self.gp_belief: Optional[GaussianProcessBelief] = None
        self.sensor_time: float = 5.0  # Time to take a measurement
        
        # Statistics
        self.planning_stats = {
            'iterations': 0,
            'tree_size': 0,
            'max_depth_reached': 0,
            'planning_time': 0.0
        }
        
        # Store root for visualization
        self.root: Optional[MCTSNode] = None
    
    def plan(
        self,
        robot: Robot,
        candidates: CandidateSet,
        gp_belief: GaussianProcessBelief,
        sensor_time: float = 5.0,
        environment_sampler: Optional[Callable[[np.ndarray], float]] = None
    ) -> List[np.ndarray]:
        """
        Plan optimal sequence of sampling locations using MCTS.
        
        Args:
            robot: Robot agent with position and budget
            candidates: Candidate sampling locations
            gp_belief: Current GP belief state
            sensor_time: Time to collect one measurement
            environment_sampler: Optional function to get true values (for simulation)
            
        Returns:
            List of target positions to visit in order
        """
        self.robot = robot
        self.candidates = candidates
        self.gp_belief = deepcopy(gp_belief)
        self.sensor_time = sensor_time
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"MCTS PLANNING - Robot {robot.id}")
            print(f"{'='*70}")
            print(f"Position: {robot.position}")
            print(f"Budget:   {robot.remaining_budget:.1f}")
            print(f"Candidates: {len(candidates.get_feasible_points())}")
        
        # Create root node
        root = MCTSNode(
            position=robot.position,
            remaining_budget=robot.remaining_budget,
            gp_belief=deepcopy(self.gp_belief),
            visited_positions=[],
            depth=0
        )
        
        # Initialize available actions
        root.untried_actions = self._get_available_actions(root)
        
        if self.config.verbose:
            print(f"  Root has {len(root.untried_actions)} available actions")
            if len(root.untried_actions) == 0:
                print(f"  WARNING: No actions available! Checking why...")
                print(f"    Feasible candidates: {len(self.candidates.get_feasible_points())}")
                print(f"    Robot budget: {robot.remaining_budget}")
                print(f"    Sensor time: {self.sensor_time}")
        
        # Store root for later access
        self.root = root
        
        # Run MCTS
        start_time = time.time()
        iteration = 0
        
        while iteration < self.config.iterations:
            # Check time limit
            if self.config.time_limit and (time.time() - start_time) > self.config.time_limit:
                break
            
            # MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
            iteration += 1
        
        # Update statistics
        self.planning_stats['iterations'] = iteration
        self.planning_stats['tree_size'] = self._count_nodes(root)
        self.planning_stats['max_depth_reached'] = self._max_depth(root)
        self.planning_stats['planning_time'] = time.time() - start_time
        
        if self.config.verbose:
            print(f"\nPlanning complete:")
            print(f"  Iterations: {iteration}")
            print(f"  Tree size: {self.planning_stats['tree_size']}")
            print(f"  Max depth: {self.planning_stats['max_depth_reached']}")
            print(f"  Time: {self.planning_stats['planning_time']:.2f}s")
        
        # Extract best path
        path = self._extract_best_path(root)
        
        if self.config.verbose:
            print(f"  Best path length: {len(path)}")
            if len(path) > 0:
                total_distance = sum(
                    np.linalg.norm(path[i] - (robot.position if i == 0 else path[i-1]))
                    for i in range(len(path))
                )
                print(f"  Total distance: {total_distance:.2f}")
        
        return path
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1 until reaching expandable node.
        """
        while not node.is_terminal:
            # Apply progressive widening if enabled
            if self.config.use_progressive_widening:
                max_children = int(
                    self.config.pw_constant * (node.visits ** self.config.pw_alpha)
                )
                
                # Should we expand more children?
                if len(node.children) < max_children:
                    # Check if there are untried actions
                    if not node.is_fully_expanded():
                        return self._expand(node)
                    else:
                        # Node is fully expanded but we want more children
                        # Get all available actions again
                        all_available = self._get_available_actions(node)
                        # Filter out actions we've already tried (existing children)
                        tried_actions = {tuple(child.action) for child in node.children}
                        node.untried_actions = [a for a in all_available if tuple(a) not in tried_actions]
                        
                        if len(node.untried_actions) > 0:
                            return self._expand(node)
            else:
                # No progressive widening - expand if not fully expanded
                if not node.is_fully_expanded():
                    return self._expand(node)
            
            # At this point, either:
            # - Progressive widening limit reached
            # - Node is fully expanded and no more actions available
            # So select best child and continue traversal
            if len(node.children) == 0:
                # No children exist - this shouldn't happen but handle it
                return node
                
            node = node.best_child(self.config.exploration_constant)
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: add new child node for an untried action.
        """
        if len(node.untried_actions) == 0:
            node.is_terminal = True
            return node
        
        # Select action (random for now, could use heuristics)
        action_idx = np.random.randint(len(node.untried_actions))
        action = node.untried_actions.pop(action_idx)
        
        # Simulate taking this action
        new_position = action
        distance = np.linalg.norm(action - node.position)
        travel_time = distance / self.robot.max_speed
        total_time = travel_time + self.sensor_time
        new_budget = node.remaining_budget - total_time
        
        # Update GP belief with this measurement
        new_gp = deepcopy(node.gp_belief)
        predicted_value, _ = new_gp.predict(action.reshape(1, -1))
        new_gp.update(action.reshape(1, -1), predicted_value)
        
        # Update visited positions
        new_visited = node.visited_positions + [action]
        
        # Create child node
        child = node.add_child(
            action=action,
            position=new_position,
            remaining_budget=new_budget,
            gp_belief=new_gp,
            visited_positions=new_visited
        )
        
        # Initialize child's actions
        child.untried_actions = self._get_available_actions(child)
        
        if len(child.untried_actions) == 0 or child.depth >= self.config.max_depth:
            child.is_terminal = True
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: perform random rollout to estimate node value.
        
        Returns total discounted reward from this state.
        """
        # Start with reward for reaching this node
        reward = self._calculate_reward(node)
        
        # Perform rollout
        current_position = node.position.copy()
        current_budget = node.remaining_budget
        current_gp = deepcopy(node.gp_belief)
        visited = set(tuple(p) for p in node.visited_positions)
        depth = 0
        discount = self.config.discount_factor
        
        while depth < self.config.simulation_depth and current_budget > self.sensor_time:
            # Get available actions
            available = self._get_available_candidates(
                current_position,
                current_budget,
                visited
            )
            
            if len(available) == 0:
                break
            
            # Random action selection for rollout
            action = available[np.random.randint(len(available))]
            
            # Simulate action
            distance = np.linalg.norm(action - current_position)
            travel_time = distance / self.robot.max_speed
            total_time = travel_time + self.sensor_time
            
            # Update state
            current_position = action
            current_budget -= total_time
            visited.add(tuple(action))
            
            # Update GP and calculate reward
            predicted_value, _ = current_gp.predict(action.reshape(1, -1))
            current_gp.update(action.reshape(1, -1), predicted_value)
            
            step_reward = self._calculate_information_gain(current_gp, action)
            reward += (discount ** depth) * step_reward
            
            depth += 1
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: update statistics up the tree.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _calculate_reward(self, node: MCTSNode) -> float:
        """
        Calculate reward for a node.
        
        Combines:
        - Information gain (uncertainty reduction)
        - Travel efficiency (penalize long distances)
        """
        if node.parent is None:
            return 0.0
        
        # Information gain component - variance BEFORE taking the measurement
        # This represents how much uncertainty we reduce by sampling here
        _, std_before = node.parent.gp_belief.predict(node.action.reshape(1, -1), return_std=True)
        variance_before = std_before[0] ** 2
        
        # Use variance before as the information gain (higher variance = more valuable)
        info_gain = variance_before
        
        # Efficiency component (penalize travel distance)
        distance = np.linalg.norm(node.action - node.parent.position)
        efficiency_penalty = -0.01 * distance  # Reduced weight to not dominate
        
        return info_gain + efficiency_penalty
    
    def _calculate_information_gain(
        self,
        gp: GaussianProcessBelief,
        position: np.ndarray
    ) -> float:
        """
        Calculate information gain (uncertainty reduction) at a position.
        
        Uses variance as a proxy for information gain.
        """
        _, std = gp.predict(position.reshape(1, -1), return_std=True)
        variance = std[0] ** 2
        return variance
    
    def _get_available_actions(self, node: MCTSNode) -> List[np.ndarray]:
        """Get available actions (candidate points) from current node."""
        visited = set(tuple(p) for p in node.visited_positions)
        return self._get_available_candidates(
            node.position,
            node.remaining_budget,
            visited
        )
    
    def _get_available_candidates(
        self,
        position: np.ndarray,
        budget: float,
        visited: set
    ) -> List[np.ndarray]:
        """
        Get candidate points that are:
        1. Feasible (in candidate set)
        2. Not yet visited
        3. Reachable within budget
        """
        feasible = self.candidates.get_feasible_points()
        
        available = []
        
        for point in feasible:
            # Skip if already visited
            if tuple(point) in visited:
                continue
            
            # Check if reachable
            distance = np.linalg.norm(point - position)
            travel_time = distance / self.robot.max_speed
            total_time = travel_time + self.sensor_time
            
            if total_time <= budget:
                available.append(point)
        
        return available
    
    def _extract_best_path(self, root: MCTSNode) -> List[np.ndarray]:
        """
        Extract best path by following highest visit count children.
        """
        path = []
        node = root
        
        while len(node.children) > 0:
            # Select child with most visits (most promising)
            node = max(node.children, key=lambda c: c.visits)
            path.append(node.action)
        
        return path
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        """Find maximum depth in tree."""
        if len(node.children) == 0:
            return current_depth
        return max(
            self._max_depth(child, current_depth + 1)
            for child in node.children
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return self.planning_stats.copy()
