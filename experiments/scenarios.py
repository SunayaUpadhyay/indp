"""
Predefined scenario configurations for experiments.
"""

# Gaussian Hotspot Scenarios (SAR)
GAUSSIAN_HOTSPOT_SPARSE = {
    'scenario_type': 'gaussian_hotspot',
    'env_function': 'gaussian_mixture',
    'env_kwargs': {
        'n_components': 4,
        'spread': 'medium'  # Moderate covariance
    }
}

GAUSSIAN_HOTSPOT_DENSE = {
    'scenario_type': 'gaussian_hotspot',
    'env_function': 'gaussian_mixture',
    'env_kwargs': {
        'n_components': 8,
        'spread': 'medium'
    }
}

# Smooth Mapping Scenarios
SMOOTH_MAPPING = {
    'scenario_type': 'smooth_mapping',
    'env_function': 'rosenbrock',
    'env_kwargs': {}
}

# Rugged Mapping Scenarios
RUGGED_ACKLEY = {
    'scenario_type': 'rugged_mapping',
    'env_function': 'ackley',
    'env_kwargs': {}
}

RUGGED_TOWNSEND = {
    'scenario_type': 'rugged_mapping',
    'env_function': 'townsend',
    'env_kwargs': {}
}

# All scenarios
ALL_SCENARIOS = [
    GAUSSIAN_HOTSPOT_SPARSE,
    GAUSSIAN_HOTSPOT_DENSE,
    SMOOTH_MAPPING,
    RUGGED_ACKLEY,
    RUGGED_TOWNSEND
]

# Quick test scenario (smaller scope)
QUICK_TEST = {
    'scenario_type': 'gaussian_hotspot',
    'env_function': 'gaussian_mixture',
    'env_kwargs': {
        'n_components': 4,
        'spread': 'medium'
    }
}
