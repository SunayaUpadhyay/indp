"""
Unit System for IPP Framework
==============================

This framework uses SI units (meters, seconds) throughout for consistency.

SPATIAL: All distances in METERS
TEMPORAL: All times in SECONDS  
VELOCITY: meters/second (m/s)
ENERGY: Dimensionless (relative units)

Examples:
    - bounds = [[0, 1000], [0, 1000]]  # 1km × 1km in meters
    - max_speed = 10.0                  # 10 m/s (36 km/h)
    - budget = 300.0 (TIME) or 5000.0 (DISTANCE)  # seconds or meters

Physical Scale:
    The 'physical_scale' parameter defines meters per coordinate unit.
    - physical_scale = 1.0: coordinates are already in meters
    - physical_scale = 100.0: each coordinate unit = 100 meters
    - physical_scale = 111320.0: coordinates in degrees (lat/lon)
"""

import numpy as np

# ============================================
# Conversion Factors
# ============================================

M_TO_KM = 0.001
KM_TO_M = 1000.0
MS_TO_KMH = 3.6
KMH_TO_MS = 1.0 / 3.6

# Approximate degrees to meters (at equator)
DEG_TO_M_LAT = 111320.0  # 1° latitude ≈ 111.32 km
DEG_TO_M_LON = 111320.0  # 1° longitude ≈ 111.32 km (varies by latitude)

# ============================================
# Standard Scales for Common Scenarios
# ============================================

# Spatial scales
SMALL_AREA_M = 100      # 100m × 100m (lab/indoor)
MEDIUM_AREA_M = 1000    # 1km × 1km (urban search)
LARGE_AREA_M = 10000    # 10km × 10km (wilderness)

# Robot speeds
SLOW_ROBOT_MS = 2.0     # Ground robot (2 m/s = 7.2 km/h)
MEDIUM_ROBOT_MS = 5.0   # Fast ground robot (5 m/s = 18 km/h)
FAST_ROBOT_MS = 15.0    # UAV (15 m/s = 54 km/h)

# ============================================
# Predefined Environment Configurations
# ============================================

ENVIRONMENT_CONFIGS = {
    'townsend': {
        'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
        'physical_scale': 100.0,  # Each unit = 100 meters
        'description': 'Townsend test function, ~475m × 425m area',
        'typical_speed': MEDIUM_ROBOT_MS,
        'typical_budget': 2000.0  # meters
    },
    'small_search': {
        'bounds': np.array([[0, 100], [0, 100]]),
        'physical_scale': 1.0,  # Already in meters
        'description': 'Small search area, 100m × 100m',
        'typical_speed': SLOW_ROBOT_MS,
        'typical_budget': 500.0  # meters
    },
    'medium_search': {
        'bounds': np.array([[0, 100], [0, 100]]),
        'physical_scale': 10.0,  # Each unit = 10 meters
        'description': 'Medium search area, 1km × 1km',
        'typical_speed': MEDIUM_ROBOT_MS,
        'typical_budget': 5000.0  # meters
    },
    'large_search': {
        'bounds': np.array([[0, 100], [0, 100]]),
        'physical_scale': 100.0,  # Each unit = 100 meters
        'description': 'Large search area, 10km × 10km',
        'typical_speed': FAST_ROBOT_MS,
        'typical_budget': 50000.0  # meters
    },
    'ocean_latlon': {
        'bounds': np.array([[-125, -123], [43, 45]]),
        'physical_scale': DEG_TO_M_LAT,  # Degrees to meters
        'description': 'Ocean data in lat/lon, ~220km × 220km',
        'typical_speed': FAST_ROBOT_MS,
        'typical_budget': 100000.0  # meters
    }
}


def get_physical_area(bounds: np.ndarray, physical_scale: float = 1.0) -> tuple:
    """
    Calculate physical area dimensions.
    
    Args:
        bounds: Coordinate bounds [[x_min, x_max], [y_min, y_max]]
        physical_scale: Meters per coordinate unit
        
    Returns:
        (width_m, height_m, area_m2): Physical dimensions in meters
    """
    coord_size = bounds[:, 1] - bounds[:, 0]
    physical_size = coord_size * physical_scale
    area = physical_size[0] * physical_size[1]
    return physical_size[0], physical_size[1], area


def suggest_budget(
    bounds: np.ndarray,
    physical_scale: float,
    coverage_fraction: float = 0.5,
    budget_type: str = 'distance'
) -> float:
    """
    Suggest appropriate budget based on environment size.
    
    Args:
        bounds: Coordinate bounds
        physical_scale: Meters per coordinate unit
        coverage_fraction: Fraction of diagonal distance (0-1)
        budget_type: 'distance', 'time', or 'energy'
        
    Returns:
        Suggested budget value
    """
    coord_size = bounds[:, 1] - bounds[:, 0]
    diagonal_coords = np.linalg.norm(coord_size)
    diagonal_meters = diagonal_coords * physical_scale
    
    if budget_type == 'distance':
        return diagonal_meters * coverage_fraction
    elif budget_type == 'time':
        # Assume medium speed robot
        return (diagonal_meters * coverage_fraction) / MEDIUM_ROBOT_MS
    elif budget_type == 'energy':
        # Quadratic energy model
        distance = diagonal_meters * coverage_fraction
        return distance ** 2
    else:
        raise ValueError(f"Unknown budget_type: {budget_type}")


def print_environment_info(
    env_name: str,
    bounds: np.ndarray,
    physical_scale: float,
    robot_speed: float,
    robot_budget: float,
    budget_type: str = 'distance'
):
    """
    Print detailed environment and robot information.
    
    Args:
        env_name: Name of the environment
        bounds: Coordinate bounds
        physical_scale: Meters per coordinate unit
        robot_speed: Robot speed in m/s
        robot_budget: Robot budget value
        budget_type: Type of budget ('distance', 'time', 'energy')
    """
    width_m, height_m, area_m2 = get_physical_area(bounds, physical_scale)
    coord_size = bounds[:, 1] - bounds[:, 0]
    diagonal_m = np.linalg.norm(coord_size) * physical_scale
    
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print(f"{'='*60}")
    print(f"Coordinate bounds: [{bounds[0,0]:.2f}, {bounds[0,1]:.2f}] × [{bounds[1,0]:.2f}, {bounds[1,1]:.2f}]")
    print(f"Coordinate size: {coord_size[0]:.2f} × {coord_size[1]:.2f} units")
    print(f"Physical scale: {physical_scale:.2f} m/unit")
    print(f"Physical area: {width_m:.1f}m × {height_m:.1f}m ({area_m2/1e6:.3f} km²)")
    print(f"Max distance (diagonal): {diagonal_m:.1f}m ({diagonal_m/1000:.2f} km)")
    print(f"\nRobot Configuration:")
    print(f"  Speed: {robot_speed:.2f} m/s ({robot_speed*MS_TO_KMH:.1f} km/h)")
    
    if budget_type == 'distance':
        print(f"  Budget: {robot_budget:.1f}m ({robot_budget/1000:.2f} km)")
        print(f"  Coverage: {robot_budget/diagonal_m*100:.1f}% of diagonal")
        time_available = robot_budget / robot_speed
        print(f"  Mission time: {time_available:.1f}s ({time_available/60:.1f} min)")
    elif budget_type == 'time':
        print(f"  Budget: {robot_budget:.1f}s ({robot_budget/60:.1f} min)")
        distance_available = robot_budget * robot_speed
        print(f"  Max distance: {distance_available:.1f}m ({distance_available/1000:.2f} km)")
        print(f"  Coverage: {distance_available/diagonal_m*100:.1f}% of diagonal")
    elif budget_type == 'energy':
        print(f"  Budget: {robot_budget:.1f} energy units")
        # Assuming quadratic model: energy = distance²
        distance_available = np.sqrt(robot_budget)
        print(f"  Equivalent distance: ~{distance_available:.1f}m")
        print(f"  Coverage: ~{distance_available/diagonal_m*100:.1f}% of diagonal")
    
    print(f"{'='*60}\n")
