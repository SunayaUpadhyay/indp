"""
Experimental Configuration - Protocol Defaults.

Defines standard defaults for all baseline experiments following the protocol.
"""

import numpy as np
from src.core.robot import BudgetType

# ============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================================

# Global bounds for all synthetic experiments (meters)
BOUNDS = np.array([[0.0, 100.0],   # x from 0 to 100
                   [0.0, 100.0]])  # y from 0 to 100

# Coordinate system
PHYSICAL_SCALE = 1.0  # 1 coordinate unit = 1 meter

# Observation noise
OBSERVATION_NOISE = 0.1  # Standard deviation

# Grid resolution for RMSE/variance evaluation
GRID_RESOLUTION = 50  # 50x50 grid over 100x100m field


# ============================================================================
# 2. ROBOT CONFIGURATION
# ============================================================================

# All robots start at same depot (bottom-left corner)
START_POSITION = np.array([0.0, 0.0])

# Robot physical parameters
BUDGET_TYPE = BudgetType.TIME  # Use time-based budgets
MAX_SPEED = 2.0  # m/s (7.2 km/h - typical ground robot like Clearpath Husky)
SENSOR_RANGE = 5.0  # meters (typical sensor footprint)

# Time budgets (seconds)
BUDGET_TIGHT = 150   # 300m @ 2m/s - quick survey
BUDGET_MEDIUM = 300  # 600m @ 2m/s - standard mission
BUDGET_LOOSE = 500   # 1000m @ 2m/s - extensive coverage

BUDGET_LEVELS = {
    'tight': BUDGET_TIGHT,
    'medium': BUDGET_MEDIUM,
    'loose': BUDGET_LOOSE
}

# Number of robots to sweep
NUM_ROBOTS_LIST = [2, 4, 8, 16]

# All baseline planners to compare
ALL_PLANNERS = [
    'Random',
    'Lawnmower',
    'SequentialGreedy',
    'IndependentGreedy',
    'Auction'
]


# ============================================================================
# 3. GAUSSIAN HOTSPOT (SAR) SCENARIO CONFIGURATION
# ============================================================================

# Gaussian mixture parameters
GMM_N_COMPONENTS_LIST = [4, 8]  # Number of hotspots to test
GMM_COVARIANCE_SCALE = 12.0  # Standard deviation of each Gaussian (meters)

# Hotspot detection threshold
HOTSPOT_PERCENTILE = 95  # Top 5% of field values are "hotspots"
NUM_HOTSPOTS_TO_TRACK = 5  # Track top K local maxima


# ============================================================================
# 4. SMOOTH/RUGGED MAPPING SCENARIO CONFIGURATION
# ============================================================================

SMOOTH_FUNCTIONS = ['rosenbrock']
RUGGED_FUNCTIONS = ['ackley', 'townsend']


# ============================================================================
# 5. EXPERIMENTAL FACTORS
# ============================================================================

# Random seeds for repeated runs
NUM_REPETITIONS = 10  # Number of runs per configuration
SEED_START = 42  # Starting seed value


# ============================================================================
# 6. PLANNER CONFIGURATIONS
# ============================================================================

# Random Planner
RANDOM_PLANNER_CONFIG = {
    'step_size': 15.0,  # Max step size in meters for local random walk
    'max_attempts': 100
}

# Lawnmower Planner
LAWNMOWER_CONFIG = {
    'stripe_width': 10.0,  # Width between parallel sweeps (meters)
    'orientation': 'vertical',  # 'vertical' or 'horizontal' strips
    'waypoint_spacing': 10.0  # Distance between waypoints along each sweep (meters)
}

# Sequential Greedy IG
SEQUENTIAL_GREEDY_CONFIG = {
    'candidate_resolution': 20,  # Grid resolution for candidates (20x20 = 400 candidates)
    'lookahead_depth': 1  # Single-step greedy (can increase for multi-step)
}

# Independent Greedy IG
INDEPENDENT_GREEDY_CONFIG = {
    'candidate_resolution': 20,  # Grid resolution for candidates (20x20 = 400 candidates)
    'lookahead_depth': 1
}

# Auction Planner
AUCTION_CONFIG = {
    'num_candidates': 100,  # Top-K variance cells to auction (increased from 50)
    'replan_interval': 1,  # Replanning frequency (steps) - replan after each measurement
    'grid_resolution': 30  # Grid resolution for variance evaluation (increased from default 50)
}


# ============================================================================
# 7. DRONE CONFIGURATION (ALTERNATIVE)
# ============================================================================

# For drone experiments, override these:
DRONE_MAX_SPEED = 10.0  # m/s (36 km/h - typical quadcopter cruise speed)
DRONE_SENSOR_RANGE = 8.0  # meters (higher altitude = larger footprint)

# Drone time budgets (cover more distance in same time)
DRONE_BUDGET_TIGHT = 100   # 1000m @ 10m/s
DRONE_BUDGET_MEDIUM = 200  # 2000m @ 10m/s
DRONE_BUDGET_LOOSE = 300   # 3000m @ 10m/s


def get_robot_config(use_drones=False):
    """
    Get robot configuration for experiments.
    
    Args:
        use_drones: If True, use drone parameters instead of ground robot
        
    Returns:
        Dictionary with robot configuration
    """
    if use_drones:
        return {
            'max_speed': DRONE_MAX_SPEED,
            'sensor_range': DRONE_SENSOR_RANGE,
            'budget_tight': DRONE_BUDGET_TIGHT,
            'budget_medium': DRONE_BUDGET_MEDIUM,
            'budget_loose': DRONE_BUDGET_LOOSE
        }
    else:
        return {
            'max_speed': MAX_SPEED,
            'sensor_range': SENSOR_RANGE,
            'budget_tight': BUDGET_TIGHT,
            'budget_medium': BUDGET_MEDIUM,
            'budget_loose': BUDGET_LOOSE
        }


# ============================================================================
# 8. OUTPUT CONFIGURATION
# ============================================================================

RESULTS_DIR = 'results/experiments'
PLOTS_DIR = 'results/plots'
LOGS_DIR = 'results/logs'
