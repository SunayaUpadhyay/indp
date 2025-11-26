"""
Metrics computation for IPP experiments.

Provides functions to compute:
- RMSE between ground truth and GP prediction
- Integrated variance over evaluation grid
- Coverage metrics
- SAR-specific metrics (hotspot detection, probability mass, etc.)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter

from src.core.environment import Environment
from src.core.belief import GaussianProcessBelief


def compute_rmse(
    environment: Environment,
    gp_belief: GaussianProcessBelief,
    eval_grid: np.ndarray
) -> float:
    """
    Compute RMSE between ground truth and GP mean prediction.
    
    Args:
        environment: Ground truth environment
        gp_belief: GP belief state
        eval_grid: Evaluation points (N, 2)
        
    Returns:
        RMSE value
    """
    # Get ground truth
    true_values = environment.evaluate(eval_grid)
    
    # Get GP predictions
    predicted_values, _ = gp_belief.predict(eval_grid, return_std=False)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    
    return rmse


def compute_integrated_variance(
    gp_belief: GaussianProcessBelief,
    eval_grid: np.ndarray
) -> float:
    """
    Compute sum of predictive variance over evaluation grid.
    
    Args:
        gp_belief: GP belief state
        eval_grid: Evaluation points (N, 2)
        
    Returns:
        Sum of variance values
    """
    _, std = gp_belief.predict(eval_grid, return_std=True)
    variance = std ** 2
    
    integrated_variance = np.sum(variance)
    
    return integrated_variance


def compute_coverage_metrics(
    visited_positions: np.ndarray,
    bounds: np.ndarray,
    cell_size: float = 5.0
) -> Dict[str, float]:
    """
    Compute spatial coverage metrics.
    
    Args:
        visited_positions: All visited positions (N, 2)
        bounds: Environment bounds [[x_min, x_max], [y_min, y_max]]
        cell_size: Size of grid cells for discretization
        
    Returns:
        Dictionary with coverage metrics
    """
    if len(visited_positions) == 0:
        return {
            'coverage_fraction': 0.0,
            'unique_cells_visited': 0,
            'total_cells': 0
        }
    
    # Create grid
    x_cells = int(np.ceil((bounds[0, 1] - bounds[0, 0]) / cell_size))
    y_cells = int(np.ceil((bounds[1, 1] - bounds[1, 0]) / cell_size))
    
    # Convert positions to cell indices
    cell_indices = set()
    for pos in visited_positions:
        x_idx = int((pos[0] - bounds[0, 0]) / cell_size)
        y_idx = int((pos[1] - bounds[1, 0]) / cell_size)
        
        # Clamp to valid range
        x_idx = np.clip(x_idx, 0, x_cells - 1)
        y_idx = np.clip(y_idx, 0, y_cells - 1)
        
        cell_indices.add((x_idx, y_idx))
    
    unique_cells = len(cell_indices)
    total_cells = x_cells * y_cells
    coverage_fraction = unique_cells / total_cells
    
    return {
        'coverage_fraction': coverage_fraction,
        'unique_cells_visited': unique_cells,
        'total_cells': total_cells
    }


def compute_overlap_metrics(
    robot_positions: Dict[int, np.ndarray],
    threshold: float = 10.0
) -> Dict[str, float]:
    """
    Compute redundancy/overlap between robot trajectories.
    
    Args:
        robot_positions: Dict mapping robot_id -> positions array (N_i, 2)
        threshold: Distance threshold for considering positions "overlapping"
        
    Returns:
        Dictionary with overlap metrics
    """
    if len(robot_positions) < 2:
        return {'overlap_fraction': 0.0, 'redundant_visits': 0}
    
    robot_ids = list(robot_positions.keys())
    total_overlaps = 0
    total_positions = 0
    
    # Compare each pair of robots
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            positions_i = robot_positions[robot_ids[i]]
            positions_j = robot_positions[robot_ids[j]]
            
            if len(positions_i) == 0 or len(positions_j) == 0:
                continue
            
            # Compute pairwise distances
            distances = cdist(positions_i, positions_j)
            
            # Count overlaps (positions within threshold)
            overlaps = np.sum(distances < threshold)
            total_overlaps += overlaps
            
            total_positions += len(positions_i)
    
    overlap_fraction = total_overlaps / max(total_positions, 1)
    
    return {
        'overlap_fraction': overlap_fraction,
        'redundant_visits': total_overlaps
    }


# ============================================================================
# SAR-SPECIFIC METRICS
# ============================================================================

def find_local_maxima(
    field_values: np.ndarray,
    grid_shape: Tuple[int, int],
    neighborhood_size: int = 3
) -> List[Tuple[int, int]]:
    """
    Find local maxima in a field.
    
    Args:
        field_values: Flattened field values
        grid_shape: Shape of grid (rows, cols)
        neighborhood_size: Size of neighborhood for maxima detection
        
    Returns:
        List of (row, col) indices of local maxima
    """
    # Reshape to grid
    field_grid = field_values.reshape(grid_shape)
    
    # Apply maximum filter
    local_max = maximum_filter(field_grid, size=neighborhood_size)
    
    # Find where field equals local maximum
    maxima_mask = (field_grid == local_max)
    
    # Get indices
    maxima_indices = np.argwhere(maxima_mask)
    
    return [(int(row), int(col)) for row, col in maxima_indices]


def compute_time_to_first_hotspot(
    trajectory: List[np.ndarray],
    environment: Environment,
    eval_grid: np.ndarray,
    hotspot_percentile: float = 95.0
) -> Optional[int]:
    """
    Compute time (iteration) to first hotspot visit.
    
    Args:
        trajectory: List of positions over time
        environment: Ground truth environment
        eval_grid: Grid for hotspot detection
        hotspot_percentile: Percentile threshold for hotspots (e.g., 95 = top 5%)
        
    Returns:
        Iteration when first hotspot reached, or None if never reached
    """
    # Evaluate field on grid
    field_values = environment.evaluate(eval_grid)
    
    # Define hotspot threshold
    threshold = np.percentile(field_values, hotspot_percentile)
    
    # Check each position in trajectory
    for iteration, position in enumerate(trajectory):
        pos_value = environment.evaluate(position.reshape(1, -1))[0]
        
        if pos_value >= threshold:
            return iteration
    
    return None


def compute_hotspot_recall(
    visited_positions: np.ndarray,
    environment: Environment,
    eval_grid: np.ndarray,
    grid_shape: Tuple[int, int],
    k_hotspots: int = 5,
    visit_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Compute recall over top-K hotspots.
    
    Args:
        visited_positions: All visited positions (N, 2)
        environment: Ground truth environment
        eval_grid: Grid for hotspot detection
        grid_shape: Shape of evaluation grid
        k_hotspots: Number of top hotspots to track
        visit_threshold: Distance threshold for "visiting" a hotspot
        
    Returns:
        Dictionary with recall metrics
    """
    # Evaluate field on grid
    field_values = environment.evaluate(eval_grid)
    
    # Find local maxima
    maxima_indices = find_local_maxima(field_values, grid_shape, neighborhood_size=5)
    
    if len(maxima_indices) == 0:
        return {
            'hotspot_recall': 0.0,
            'hotspots_found': 0,
            'total_hotspots': k_hotspots
        }
    
    # Get values at maxima
    maxima_values = [field_values.reshape(grid_shape)[row, col] 
                     for row, col in maxima_indices]
    
    # Sort by value and take top K
    sorted_indices = np.argsort(maxima_values)[::-1]
    top_k_indices = [maxima_indices[i] for i in sorted_indices[:k_hotspots]]
    
    # Convert grid indices to coordinates
    x_coords = np.linspace(environment.bounds[0, 0], environment.bounds[0, 1], grid_shape[1])
    y_coords = np.linspace(environment.bounds[1, 0], environment.bounds[1, 1], grid_shape[0])
    
    hotspot_positions = []
    for row, col in top_k_indices:
        x = x_coords[col]
        y = y_coords[row]
        hotspot_positions.append(np.array([x, y]))
    
    # Check which hotspots were visited
    visited_count = 0
    for hotspot_pos in hotspot_positions:
        # Check if any visited position is within threshold
        if len(visited_positions) > 0:
            distances = np.linalg.norm(visited_positions - hotspot_pos, axis=1)
            if np.any(distances < visit_threshold):
                visited_count += 1
    
    recall = visited_count / len(hotspot_positions)
    
    return {
        'hotspot_recall': recall,
        'hotspots_found': visited_count,
        'total_hotspots': len(hotspot_positions)
    }


def compute_probability_mass_covered(
    visited_positions: np.ndarray,
    environment: Environment,
    eval_grid: np.ndarray,
    visit_threshold: float = 5.0
) -> float:
    """
    Compute fraction of probability mass covered (for GMM environments).
    
    Args:
        visited_positions: All visited positions (N, 2)
        environment: Ground truth environment
        eval_grid: Grid for field evaluation
        visit_threshold: Distance threshold for coverage
        
    Returns:
        Fraction of total probability mass covered
    """
    # Evaluate field on grid (treat as probability density)
    field_values = environment.evaluate(eval_grid)
    
    # Normalize to probability (sum to 1)
    total_mass = np.sum(field_values)
    if total_mass == 0:
        return 0.0
    
    probabilities = field_values / total_mass
    
    # Mark grid points as visited if within threshold of any visit
    visited_mask = np.zeros(len(eval_grid), dtype=bool)
    
    if len(visited_positions) > 0:
        for grid_idx, grid_point in enumerate(eval_grid):
            distances = np.linalg.norm(visited_positions - grid_point, axis=1)
            if np.any(distances < visit_threshold):
                visited_mask[grid_idx] = True
    
    # Sum probability mass at visited locations
    covered_mass = np.sum(probabilities[visited_mask])
    
    return covered_mass


def compute_redundant_hotspot_coverage(
    robot_positions: Dict[int, np.ndarray],
    environment: Environment,
    eval_grid: np.ndarray,
    grid_shape: Tuple[int, int],
    k_hotspots: int = 5,
    visit_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Compute how many robots visited each hotspot (redundancy metric).
    
    Args:
        robot_positions: Dict mapping robot_id -> positions array
        environment: Ground truth environment
        eval_grid: Grid for hotspot detection
        grid_shape: Shape of evaluation grid
        k_hotspots: Number of top hotspots to track
        visit_threshold: Distance threshold for visiting
        
    Returns:
        Dictionary with redundancy metrics
    """
    # Find top-K hotspots
    field_values = environment.evaluate(eval_grid)
    maxima_indices = find_local_maxima(field_values, grid_shape, neighborhood_size=5)
    
    if len(maxima_indices) == 0:
        return {
            'mean_robots_per_hotspot': 0.0,
            'max_robots_per_hotspot': 0,
            'unique_hotspots_visited': 0
        }
    
    maxima_values = [field_values.reshape(grid_shape)[row, col] 
                     for row, col in maxima_indices]
    sorted_indices = np.argsort(maxima_values)[::-1]
    top_k_indices = [maxima_indices[i] for i in sorted_indices[:k_hotspots]]
    
    # Convert to coordinates
    x_coords = np.linspace(environment.bounds[0, 0], environment.bounds[0, 1], grid_shape[1])
    y_coords = np.linspace(environment.bounds[1, 0], environment.bounds[1, 1], grid_shape[0])
    
    hotspot_positions = []
    for row, col in top_k_indices:
        x = x_coords[col]
        y = y_coords[row]
        hotspot_positions.append(np.array([x, y]))
    
    # Count robots visiting each hotspot
    robots_per_hotspot = []
    
    for hotspot_pos in hotspot_positions:
        visiting_robots = 0
        
        for robot_id, positions in robot_positions.items():
            if len(positions) > 0:
                distances = np.linalg.norm(positions - hotspot_pos, axis=1)
                if np.any(distances < visit_threshold):
                    visiting_robots += 1
        
        if visiting_robots > 0:
            robots_per_hotspot.append(visiting_robots)
    
    if len(robots_per_hotspot) == 0:
        return {
            'mean_robots_per_hotspot': 0.0,
            'max_robots_per_hotspot': 0,
            'unique_hotspots_visited': 0
        }
    
    return {
        'mean_robots_per_hotspot': np.mean(robots_per_hotspot),
        'max_robots_per_hotspot': int(np.max(robots_per_hotspot)),
        'unique_hotspots_visited': len(robots_per_hotspot)
    }
