"""
Quadtree spatial data structure for adaptive refinement.

The quadtree recursively subdivides the 2D space based on GP variance,
creating finer cells in high-uncertainty regions and coarser cells
in low-uncertainty regions.

Design rationale:
- Adaptive spatial resolution reduces candidate set size
- Focuses computational effort on uncertain areas
- Maintains spatial coverage across the entire domain
- Efficient O(log n) spatial queries
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class QuadTreeNode:
    """
    A node in the quadtree representing a rectangular cell.
    
    Attributes:
        bounds: Cell boundaries [x_min, x_max, y_min, y_max]
        center: Cell center point [x, y]
        depth: Depth in tree (0 = root)
        variance: GP variance at this cell (computed lazily)
        is_leaf: Whether this is a leaf node (no children)
        children: List of 4 child nodes (NW, NE, SW, SE) or None
    """
    bounds: np.ndarray
    center: np.ndarray
    depth: int
    variance: Optional[float] = None
    is_leaf: bool = True
    children: Optional[List['QuadTreeNode']] = None
    
    def __post_init__(self):
        """Compute center if not provided."""
        if self.center is None:
            self.center = np.array([
                (self.bounds[0] + self.bounds[1]) / 2,
                (self.bounds[2] + self.bounds[3]) / 2
            ])
    
    @property
    def width(self) -> float:
        """Cell width."""
        return self.bounds[1] - self.bounds[0]
    
    @property
    def height(self) -> float:
        """Cell height."""
        return self.bounds[3] - self.bounds[2]
    
    @property
    def area(self) -> float:
        """Cell area."""
        return self.width * self.height
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within this cell."""
        return (self.bounds[0] <= point[0] <= self.bounds[1] and
                self.bounds[2] <= point[1] <= self.bounds[3])
    
    def get_corners(self) -> np.ndarray:
        """Get all 4 corner points of the cell."""
        x_min, x_max, y_min, y_max = self.bounds
        return np.array([
            [x_min, y_min],  # SW
            [x_max, y_min],  # SE
            [x_min, y_max],  # NW
            [x_max, y_max]   # NE
        ])


class QuadTree:
    """
    Adaptive quadtree for spatial refinement based on GP uncertainty.
    
    The tree recursively subdivides cells where GP variance exceeds a threshold,
    creating a hierarchical spatial decomposition focused on uncertain regions.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        max_depth: int = 6,
        min_cell_size: float = 2.0,
        variance_threshold: float = 0.1
    ):
        """
        Initialize quadtree.
        
        Args:
            bounds: Spatial bounds [[x_min, x_max], [y_min, y_max]]
            max_depth: Maximum tree depth
            min_cell_size: Minimum cell size (width or height)
            variance_threshold: Variance threshold for splitting cells
        """
        self.bounds = bounds
        self.max_depth = max_depth
        self.min_cell_size = min_cell_size
        self.variance_threshold = variance_threshold
        
        # Create root node
        root_bounds = np.array([
            bounds[0, 0], bounds[0, 1],  # x_min, x_max
            bounds[1, 0], bounds[1, 1]   # y_min, y_max
        ])
        self.root = QuadTreeNode(bounds=root_bounds, center=None, depth=0)
        
        # Statistics
        self.n_nodes = 1
        self.n_leaves = 1
    
    def refine(self, variance_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Refine the quadtree based on GP variance.
        
        Args:
            variance_func: Function that takes points (n_points, 2) and 
                         returns variances (n_points,)
        """
        self._refine_node(self.root, variance_func)
    
    def _refine_node(
        self,
        node: QuadTreeNode,
        variance_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """
        Recursively refine a node if variance exceeds threshold.
        
        Args:
            node: Node to potentially refine
            variance_func: GP variance function
        """
        # Evaluate variance at node center
        if node.variance is None:
            node.variance = variance_func(node.center.reshape(1, -1))[0]
        
        # Check stopping criteria
        if (node.depth >= self.max_depth or
            node.width <= self.min_cell_size or
            node.height <= self.min_cell_size or
            node.variance <= self.variance_threshold):
            # Don't split
            return
        
        # Split into 4 children
        self._split_node(node)
        
        # Recursively refine children
        for child in node.children:
            self._refine_node(child, variance_func)
    
    def _split_node(self, node: QuadTreeNode) -> None:
        """
        Split a node into 4 children (NW, NE, SW, SE).
        
        Args:
            node: Node to split
        """
        x_min, x_max, y_min, y_max = node.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        # Create 4 children
        children_bounds = [
            [x_min, x_mid, y_mid, y_max],  # NW
            [x_mid, x_max, y_mid, y_max],  # NE
            [x_min, x_mid, y_min, y_mid],  # SW
            [x_mid, x_max, y_min, y_mid]   # SE
        ]
        
        node.children = [
            QuadTreeNode(
                bounds=np.array(bounds),
                center=None,
                depth=node.depth + 1
            )
            for bounds in children_bounds
        ]
        
        node.is_leaf = False
        
        # Update statistics
        self.n_nodes += 4
        self.n_leaves += 3  # -1 (parent) + 4 (children)
    
    def get_leaf_nodes(self) -> List[QuadTreeNode]:
        """
        Get all leaf nodes in the tree.
        
        Returns:
            List of leaf nodes
        """
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves
    
    def _collect_leaves(
        self,
        node: QuadTreeNode,
        leaves: List[QuadTreeNode]
    ) -> None:
        """Recursively collect leaf nodes."""
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)
    
    def get_cell_centers(self) -> np.ndarray:
        """
        Get centers of all leaf cells.
        
        Returns:
            Array of shape (n_leaves, 2) with cell centers
        """
        leaves = self.get_leaf_nodes()
        return np.array([leaf.center for leaf in leaves])
    
    def get_high_variance_cells(self, top_k: Optional[int] = None) -> List[QuadTreeNode]:
        """
        Get leaf cells sorted by variance (highest first).
        
        Args:
            top_k: Return only top k cells (all if None)
            
        Returns:
            List of leaf nodes sorted by variance
        """
        leaves = self.get_leaf_nodes()
        
        # Filter out nodes without variance computed
        leaves = [leaf for leaf in leaves if leaf.variance is not None]
        
        # Sort by variance (descending)
        leaves.sort(key=lambda x: x.variance, reverse=True)
        
        if top_k is not None:
            leaves = leaves[:top_k]
        
        return leaves
    
    def visualize(self, ax=None, show_variance=True, linewidth=1.5, edgecolor='black'):
        """
        Visualize the quadtree structure.
        
        Args:
            ax: Matplotlib axis (creates new if None)
            show_variance: Whether to color cells by variance
            linewidth: Width of cell boundaries
            edgecolor: Color of cell boundaries
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        leaves = self.get_leaf_nodes()
        
        # Color by variance if available
        if show_variance and any(leaf.variance is not None for leaf in leaves):
            variances = [leaf.variance if leaf.variance is not None else 0 
                        for leaf in leaves]
            norm = Normalize(vmin=min(variances), vmax=max(variances))
            cmap = plt.cm.Reds
            
            for leaf, var in zip(leaves, variances):
                x_min, x_max, y_min, y_max = leaf.bounds
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor=cmap(norm(var)),
                    alpha=0.5
                )
                ax.add_patch(rect)
            
            # Add colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='GP Variance')
        else:
            # Just draw clean rectangles without fill
            for leaf in leaves:
                x_min, x_max, y_min, y_max = leaf.bounds
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor='none'
                )
                ax.add_patch(rect)
        
        ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_title(f'Quadtree ({self.n_leaves} cells)', fontsize=12, fontweight='bold')
        ax.grid(False)
        
        return ax
    
    def __repr__(self) -> str:
        return (f"QuadTree(n_nodes={self.n_nodes}, n_leaves={self.n_leaves}, "
                f"max_depth={self.max_depth})")
