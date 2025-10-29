"""
Candidate point generator for informative path planning.

This module implements the candidate generation strategy (Step A):
1. Build adaptive quadtree based on GP variance
2. Sample candidate points from refined cells
3. Filter candidates by feasibility for each robot
4. Ensure spatial diversity

Design rationale:
- Quadtree focuses candidates on high-uncertainty regions
- Multiple sampling strategies (center, corners, random, frontier)
- Feasibility checking prevents infeasible candidate selection
- Diversity promotion prevents clustering
"""

import numpy as np
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass

from .quadtree import QuadTree, QuadTreeNode
from ...core.belief import GaussianProcessBelief
from ...core.robot import Robot


@dataclass
class CandidateSet:
    """
    Set of candidate points for a robot.
    
    Attributes:
        robot_id: ID of robot these candidates are for
        points: Candidate positions of shape (n_candidates, 2)
        scores: Optional pre-computed scores for each candidate
        feasible: Boolean mask indicating feasibility
        metadata: Additional metadata (e.g., which cell, variance, etc.)
    """
    robot_id: int
    points: np.ndarray
    scores: Optional[np.ndarray] = None
    feasible: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return len(self.points)
    
    def get_feasible_points(self) -> np.ndarray:
        """Get only feasible candidate points."""
        if self.feasible is None:
            return self.points
        return self.points[self.feasible]
    
    def get_top_k(self, k: int) -> np.ndarray:
        """Get top k candidates by score."""
        if self.scores is None:
            # Return first k
            return self.points[:k]
        
        # Sort by score (descending)
        indices = np.argsort(self.scores)[::-1][:k]
        return self.points[indices]


class CandidateGenerator:
    """
    Generates candidate sampling points using adaptive quadtree refinement.
    
    This is the main class for Step A of the algorithm.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        quadtree_config: Optional[Dict[str, Any]] = None,
        sampling_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize candidate generator.
        
        Args:
            bounds: Spatial bounds [[x_min, x_max], [y_min, y_max]]
            quadtree_config: Configuration for quadtree refinement
            sampling_config: Configuration for candidate sampling
        """
        self.bounds = bounds
        
        # Quadtree configuration
        qt_config = quadtree_config or {}
        self.max_depth = qt_config.get('max_depth', 6)
        self.min_cell_size = qt_config.get('min_cell_size', 2.0)
        self.variance_threshold = qt_config.get('variance_threshold', 0.1)
        
        # Sampling configuration
        samp_config = sampling_config or {}
        self.points_per_cell = samp_config.get('points_per_cell', 1)
        self.include_corners = samp_config.get('include_corners', False)
        self.include_frontier = samp_config.get('include_frontier', False)
        self.min_spacing = samp_config.get('min_spacing', 5.0)
        self.sampling_method = samp_config.get('method', 'center')  # 'center', 'random', 'grid'
        
        # Current quadtree (rebuilt each planning cycle)
        self.quadtree: Optional[QuadTree] = None
    
    def generate_candidates(
        self,
        gp: GaussianProcessBelief,
        robots: List[Robot],
        budget_reserve: float = 0.0
    ) -> Dict[int, CandidateSet]:
        """
        Generate candidate points for all robots.
        
        This is the main entry point for Step A.
        
        Args:
            gp: Current Gaussian Process belief
            robots: List of all robots
            budget_reserve: Budget to keep in reserve when checking feasibility
            
        Returns:
            Dictionary mapping robot_id -> CandidateSet
        """
        # Step 1: Build adaptive quadtree based on GP variance
        self.quadtree = self._build_quadtree(gp)
        
        # Step 2: Sample candidate points from refined cells
        candidates = self._sample_from_quadtree(self.quadtree)
        
        # Step 3: Promote spatial diversity
        if self.min_spacing > 0:
            candidates = self._enforce_diversity(candidates, self.min_spacing)
        
        # Step 4: Filter by feasibility for each robot
        candidate_sets = {}
        for robot in robots:
            if robot.is_active:
                candidate_set = self._filter_feasible_for_robot(
                    robot, candidates, budget_reserve
                )
                candidate_sets[robot.id] = candidate_set
        
        return candidate_sets
    
    def _build_quadtree(self, gp: GaussianProcessBelief) -> QuadTree:
        """
        Build adaptive quadtree based on GP variance.
        
        Args:
            gp: Gaussian Process belief
            
        Returns:
            Refined QuadTree
        """
        # Create quadtree
        quadtree = QuadTree(
            bounds=self.bounds,
            max_depth=self.max_depth,
            min_cell_size=self.min_cell_size,
            variance_threshold=self.variance_threshold
        )
        
        # Refine based on GP variance
        variance_func = lambda points: gp.get_variance(points)
        quadtree.refine(variance_func)
        
        return quadtree
    
    def _sample_from_quadtree(self, quadtree: QuadTree) -> np.ndarray:
        """
        Sample candidate points from quadtree cells.
        
        Args:
            quadtree: Refined quadtree
            
        Returns:
            Array of candidate points (n_candidates, 2)
        """
        leaves = quadtree.get_leaf_nodes()
        candidates = []
        
        for leaf in leaves:
            # Sample points from this cell
            cell_points = self._sample_from_cell(leaf)
            candidates.extend(cell_points)
        
        return np.array(candidates)
    
    def _sample_from_cell(self, cell: QuadTreeNode) -> List[np.ndarray]:
        """
        Sample points from a single cell.
        
        Args:
            cell: Quadtree cell
            
        Returns:
            List of sampled points
        """
        points = []
        
        if self.sampling_method == 'center':
            # Cell center (always)
            for _ in range(self.points_per_cell):
                points.append(cell.center)
        
        elif self.sampling_method == 'random':
            # Random points within cell
            x_min, x_max, y_min, y_max = cell.bounds
            for _ in range(self.points_per_cell):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                points.append(np.array([x, y]))
        
        elif self.sampling_method == 'grid':
            # Uniform grid within cell
            n_points = int(np.ceil(np.sqrt(self.points_per_cell)))
            x_min, x_max, y_min, y_max = cell.bounds
            x_vals = np.linspace(x_min, x_max, n_points + 2)[1:-1]
            y_vals = np.linspace(y_min, y_max, n_points + 2)[1:-1]
            for x in x_vals:
                for y in y_vals:
                    points.append(np.array([x, y]))
                    if len(points) >= self.points_per_cell:
                        break
                if len(points) >= self.points_per_cell:
                    break
        
        # Optionally add corners
        if self.include_corners:
            corners = cell.get_corners()
            points.extend([corner for corner in corners])
        
        return points
    
    def _enforce_diversity(
        self,
        candidates: np.ndarray,
        min_spacing: float
    ) -> np.ndarray:
        """
        Remove candidates that are too close together.
        
        Uses greedy selection: keep candidates in order of importance
        (high variance first) and remove those too close to already-selected.
        
        Args:
            candidates: Array of candidate points (n, 2)
            min_spacing: Minimum distance between candidates
            
        Returns:
            Filtered array of candidates
        """
        if len(candidates) == 0:
            return candidates
        
        # TODO: Could sort by variance if available
        # For now, just use sequential greedy removal
        
        selected = [candidates[0]]
        
        for i in range(1, len(candidates)):
            candidate = candidates[i]
            
            # Check distance to all selected candidates
            distances = np.linalg.norm(
                np.array(selected) - candidate,
                axis=1
            )
            
            # Only add if far enough from all selected
            if np.all(distances >= min_spacing):
                selected.append(candidate)
        
        return np.array(selected)
    
    def _filter_feasible_for_robot(
        self,
        robot: Robot,
        candidates: np.ndarray,
        budget_reserve: float
    ) -> CandidateSet:
        """
        Filter candidates by feasibility for a specific robot.
        
        Args:
            robot: Robot to check feasibility for
            candidates: All candidate points
            budget_reserve: Budget to keep in reserve
            
        Returns:
            CandidateSet with feasibility mask
        """
        n_candidates = len(candidates)
        feasible = np.zeros(n_candidates, dtype=bool)
        
        for i, candidate in enumerate(candidates):
            # Check if robot can reach this candidate
            feasible[i] = robot.can_reach(candidate, budget_reserve)
        
        return CandidateSet(
            robot_id=robot.id,
            points=candidates,
            feasible=feasible,
            metadata={
                'n_total': n_candidates,
                'n_feasible': np.sum(feasible),
                'budget_reserve': budget_reserve
            }
        )
    
    def visualize_candidates(
        self,
        candidate_sets: Dict[int, CandidateSet],
        robots: Optional[List[Robot]] = None,
        ax=None,
        show_quadtree: bool = True,
        show_robot_radius: bool = True
    ):
        """
        Visualize candidate points and quadtree structure.
        
        Args:
            candidate_sets: Dictionary of candidate sets per robot
            robots: List of robots (to show positions)
            ax: Matplotlib axis
            show_quadtree: Whether to show quadtree cells
            show_robot_radius: Whether to show robot reachable radius
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show quadtree structure
        if show_quadtree and self.quadtree is not None:
            self.quadtree.visualize(ax=ax, show_variance=True)
        
        # Plot robot positions and reachable radius
        if robots is not None:
            robot_positions = np.array([r.position for r in robots])
            
            # Draw reachable radius circles
            if show_robot_radius:
                for robot in robots:
                    circle = Circle(
                        robot.position, 
                        robot.remaining_budget,
                        fill=False,
                        edgecolor='red',
                        linewidth=2,
                        linestyle='--',
                        alpha=0.6,
                        zorder=4
                    )
                    ax.add_patch(circle)
            
            # Plot robot positions
            ax.scatter(robot_positions[:, 0], robot_positions[:, 1],
                      c='red', s=300, marker='o', edgecolors='black',
                      linewidths=3, label='Robots', zorder=10)
            
            # Add robot ID labels
            for robot in robots:
                ax.annotate(f'R{robot.id}', 
                           xy=robot.position,
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           zorder=11)
        
        # Plot candidates for each robot
        colors = plt.cm.tab10(np.linspace(0, 1, len(candidate_sets)))
        for (robot_id, cand_set), color in zip(candidate_sets.items(), colors):
            feasible_points = cand_set.get_feasible_points()
            if len(feasible_points) > 0:
                ax.scatter(feasible_points[:, 0], feasible_points[:, 1],
                          c=[color], s=50, marker='x', alpha=0.7,
                          label=f'Robot {robot_id} candidates ({len(feasible_points)})',
                          zorder=5)
        
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title('Candidate Points with Adaptive Quadtree')
        
        return ax
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current candidate generation."""
        if self.quadtree is None:
            return {}
        
        return {
            'n_quadtree_nodes': self.quadtree.n_nodes,
            'n_leaf_cells': self.quadtree.n_leaves,
            'max_depth_reached': self.quadtree.max_depth,
        }
