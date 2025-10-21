"""
Environment representation for IPP.

This module defines the ground truth environment that robots explore.
Supports synthetic test functions and real data.

Design rationale:
- Separation of ground truth from belief (GP)
- Reproducible synthetic environments for testing
- Easy integration of real-world data
- Noise models for realistic sensing
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract base class for environments.
    
    Provides ground truth function and observation capabilities.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        observation_noise: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize environment.
        
        Args:
            bounds: Spatial bounds [[x_min, x_max], [y_min, y_max]]
            observation_noise: Standard deviation of observation noise
            seed: Random seed for reproducibility
        """
        self.bounds = np.array(bounds)
        self.observation_noise = observation_noise
        self.seed = seed
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate ground truth function at points (without noise).
        
        Args:
            X: Points of shape (n_points, n_dims)
            
        Returns:
            True values of shape (n_points,)
        """
        pass
    
    def observe(self, X: np.ndarray) -> np.ndarray:
        """
        Observe environment at points (with noise).
        
        Args:
            X: Points of shape (n_points, n_dims)
            
        Returns:
            Noisy observations of shape (n_points,)
        """
        X = np.atleast_2d(X)
        true_values = self.evaluate(X)
        noise = self.rng.normal(0, self.observation_noise, size=true_values.shape)
        return true_values + noise
    
    def evaluate_grid(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate environment on a regular grid for visualization.
        
        Args:
            resolution: Grid resolution per dimension
            
        Returns:
            X: x-coordinates of grid of shape (resolution,)
            Y: y-coordinates of grid of shape (resolution,)
            Z: Function values of shape (resolution, resolution)
        """
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], resolution)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate on grid
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self.evaluate(points).reshape(X.shape)
        
        return X, Y, Z
    
    def is_within_bounds(self, X: np.ndarray) -> np.ndarray:
        """
        Check if points are within environment bounds.
        
        Args:
            X: Points of shape (n_points, n_dims)
            
        Returns:
            Boolean array of shape (n_points,)
        """
        X = np.atleast_2d(X)
        within = np.all(
            (X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]),
            axis=1
        )
        return within


class SyntheticEnvironment(Environment):
    """
    Synthetic test functions for benchmarking.
    
    Provides several standard test functions used in optimization and GP research.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        function_name: str = 'peaks',
        observation_noise: float = 0.1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize synthetic environment.
        
        Args:
            bounds: Spatial bounds
            function_name: Name of test function ('peaks', 'ackley', 'rastrigin', etc.)
            observation_noise: Observation noise level
            seed: Random seed
            **kwargs: Additional function-specific parameters
        """
        super().__init__(bounds, observation_noise, seed)
        self.function_name = function_name.lower()
        self.kwargs = kwargs
        
        # Select function
        self.function = self._get_function(self.function_name)
    
    def _get_function(self, name: str) -> Callable:
        """Get function by name."""
        functions = {
            'peaks': self._peaks,
            'ackley': self._ackley,
            'rastrigin': self._rastrigin,
            'rosenbrock': self._rosenbrock,
            'sphere': self._sphere,
            'branin': self._branin,
            'forrester': self._forrester,
            'townsend': self._townsend,
            'gaussian_mixture': self._gaussian_mixture,
        }
        
        if name not in functions:
            raise ValueError(f"Unknown function: {name}. Available: {list(functions.keys())}")
        
        return functions[name]
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate synthetic function."""
        X = np.atleast_2d(X)
        return self.function(X)
    
    # Test functions
    
    def _peaks(self, X: np.ndarray) -> np.ndarray:
        """
        MATLAB peaks function (2D).
        A standard test function with multiple peaks and valleys.
        """
        x, y = X[:, 0], X[:, 1]
        z = (3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
             - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
             - 1/3 * np.exp(-(x + 1)**2 - y**2))
        return z
    
    def _ackley(self, X: np.ndarray) -> np.ndarray:
        """
        Ackley function (N-D).
        Highly multimodal with many local minima.
        """
        a = self.kwargs.get('a', 20)
        b = self.kwargs.get('b', 0.2)
        c = self.kwargs.get('c', 2 * np.pi)
        
        d = X.shape[1]
        sum1 = np.sum(X**2, axis=1)
        sum2 = np.sum(np.cos(c * X), axis=1)
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        
        return term1 + term2 + a + np.exp(1)
    
    def _rastrigin(self, X: np.ndarray) -> np.ndarray:
        """
        Rastrigin function (N-D).
        Highly multimodal with regularly distributed local minima.
        """
        A = self.kwargs.get('A', 10)
        d = X.shape[1]
        return A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    
    def _rosenbrock(self, X: np.ndarray) -> np.ndarray:
        """
        Rosenbrock function (N-D).
        Banana-shaped valley, difficult to optimize.
        """
        a = self.kwargs.get('a', 1)
        b = self.kwargs.get('b', 100)
        
        if X.shape[1] == 1:
            return (a - X[:, 0])**2
        
        result = np.zeros(len(X))
        for i in range(X.shape[1] - 1):
            result += (a - X[:, i])**2 + b * (X[:, i+1] - X[:, i]**2)**2
        
        return result
    
    def _sphere(self, X: np.ndarray) -> np.ndarray:
        """
        Sphere function (N-D).
        Simple convex function, easy to optimize.
        """
        return np.sum(X**2, axis=1)
    
    def _branin(self, X: np.ndarray) -> np.ndarray:
        """
        Branin function (2D).
        Three global minima, commonly used in Bayesian optimization.
        """
        if X.shape[1] != 2:
            raise ValueError("Branin function requires 2D input")
        
        x1, x2 = X[:, 0], X[:, 1]
        a = self.kwargs.get('a', 1)
        b = self.kwargs.get('b', 5.1 / (4 * np.pi**2))
        c = self.kwargs.get('c', 5 / np.pi)
        r = self.kwargs.get('r', 6)
        s = self.kwargs.get('s', 10)
        t = self.kwargs.get('t', 1 / (8 * np.pi))
        
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s
        
        return term1 + term2 + term3
    
    def _forrester(self, X: np.ndarray) -> np.ndarray:
        """
        Forrester function (1D).
        Simple 1D function used in surrogate modeling examples.
        """
        if X.shape[1] != 1:
            raise ValueError("Forrester function requires 1D input")
        
        x = X[:, 0]
        return (6 * x - 2)**2 * np.sin(12 * x - 4)
    
    def _townsend(self, X: np.ndarray) -> np.ndarray:
        """
        Townsend function (2D).
        
        Non-convex function with multiple local minima.
        Commonly used to test if planners can escape local minima.
        
        Domain: x ∈ [-2.25, 2.5], y ∈ [-2.5, 1.75]
        Global minimum at approximately (-0.0299, -0.1151) with f ≈ -2.0239
        
        Reference: Benchmark functions for optimization
        """
        if X.shape[1] != 2:
            raise ValueError("Townsend function requires 2D input")
        
        x1, x2 = X[:, 0], X[:, 1]
        
        # Main function
        term1 = -(np.cos((x1 - 0.1) * x2))**2
        term2 = -x1 * np.sin(3 * x1 + x2)
        
        return term1 + term2
    
    def _gaussian_mixture(self, X: np.ndarray) -> np.ndarray:
        """
        Gaussian Mixture Model (N-D).
        
        Simulates a field with multiple Gaussian "hotspots" or survivor clusters.
        Used in Search & Rescue scenarios (Singh, Krause & Kaiser, IJCAI 2009).
        
        Configuration via kwargs:
        - n_components: Number of Gaussian components (default: 4)
        - means: Cluster centers, shape (n_components, n_dims) or 'random'
        - covs: Covariances for each component or scalar for isotropic
        - weights: Mixture weights (default: uniform)
        
        Returns: Probability density or potential field value at each point.
        """
        n_components = self.kwargs.get('n_components', 4)
        means = self.kwargs.get('means', None)
        covs = self.kwargs.get('covs', 5.0)  # Default: isotropic with std=5.0
        weights = self.kwargs.get('weights', None)
        
        n_dims = X.shape[1]
        
        # Generate random means if not provided
        if means is None or means == 'random':
            # Place means uniformly within bounds
            means = self.rng.uniform(
                low=self.bounds[:, 0],
                high=self.bounds[:, 1],
                size=(n_components, n_dims)
            )
        else:
            means = np.array(means)
            if means.shape != (n_components, n_dims):
                raise ValueError(f"means shape {means.shape} doesn't match "
                               f"({n_components}, {n_dims})")
        
        # Set up covariances
        if np.isscalar(covs):
            # Isotropic covariance
            covs = np.array([covs**2 * np.eye(n_dims) for _ in range(n_components)])
        else:
            covs = np.array(covs)
        
        # Set up weights
        if weights is None:
            weights = np.ones(n_components) / n_components
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
        
        # Store for reproducibility (first time)
        if not hasattr(self, '_gmm_params_cached'):
            self._gmm_params_cached = {
                'means': means,
                'covs': covs,
                'weights': weights
            }
        else:
            # Use cached params for consistency
            means = self._gmm_params_cached['means']
            covs = self._gmm_params_cached['covs']
            weights = self._gmm_params_cached['weights']
        
        # Evaluate Gaussian mixture
        result = np.zeros(len(X))
        for i in range(n_components):
            diff = X - means[i]
            try:
                cov_inv = np.linalg.inv(covs[i])
                mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
                det = np.linalg.det(covs[i])
                normalizer = 1.0 / np.sqrt((2 * np.pi)**n_dims * det)
                result += weights[i] * normalizer * np.exp(-0.5 * mahalanobis)
            except np.linalg.LinAlgError:
                # Fallback to Euclidean if covariance is singular
                distances = np.linalg.norm(diff, axis=1)
                result += weights[i] * np.exp(-0.5 * distances**2 / covs[i][0, 0])
        
        return result


class InterpolatedDataEnvironment(Environment):
    """
    Environment based on real data with interpolation.
    
    Uses scipy interpolation methods to create a continuous field from
    discrete measurements. Suitable for:
    - LAMP lunar crater data
    - Lake Haviland temperature measurements
    - Any gridded or scattered real-world data
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        data_points: np.ndarray,
        data_values: np.ndarray,
        observation_noise: float = 0.1,
        seed: Optional[int] = None,
        interpolation_method: str = 'linear',
        fill_value: float = 0.0
    ):
        """
        Initialize interpolated data environment.
        
        Args:
            bounds: Spatial bounds
            data_points: Known data locations of shape (n_data, n_dims)
            data_values: Known data values of shape (n_data,)
            observation_noise: Observation noise level
            seed: Random seed
            interpolation_method: 'linear', 'cubic', 'nearest', or 'rbf'
            fill_value: Value to use outside data range
        """
        super().__init__(bounds, observation_noise, seed)
        self.data_points = np.atleast_2d(data_points)
        self.data_values = np.atleast_1d(data_values)
        self.interpolation_method = interpolation_method
        self.fill_value = fill_value
        
        # Create interpolator
        self._setup_interpolator()
    
    def _setup_interpolator(self):
        """Set up the interpolation function."""
        try:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
        except ImportError:
            raise ImportError("scipy is required for InterpolatedDataEnvironment")
        
        n_dims = self.data_points.shape[1]
        
        if self.interpolation_method == 'linear':
            self.interpolator = LinearNDInterpolator(
                self.data_points,
                self.data_values,
                fill_value=self.fill_value
            )
        elif self.interpolation_method == 'nearest':
            self.interpolator = NearestNDInterpolator(
                self.data_points,
                self.data_values
            )
        elif self.interpolation_method == 'rbf':
            # Radial basis function interpolation
            if n_dims == 1:
                self.interpolator = Rbf(
                    self.data_points[:, 0],
                    self.data_values,
                    function='thin_plate'
                )
            elif n_dims == 2:
                self.interpolator = Rbf(
                    self.data_points[:, 0],
                    self.data_points[:, 1],
                    self.data_values,
                    function='thin_plate'
                )
            else:
                # Fallback to linear for higher dimensions
                print(f"Warning: RBF not well-supported for {n_dims}D, using linear")
                self.interpolator = LinearNDInterpolator(
                    self.data_points,
                    self.data_values,
                    fill_value=self.fill_value
                )
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate using interpolation."""
        X = np.atleast_2d(X)
        
        if self.interpolation_method == 'rbf' and X.shape[1] <= 2:
            # RBF has different calling convention
            if X.shape[1] == 1:
                return self.interpolator(X[:, 0])
            else:
                return self.interpolator(X[:, 0], X[:, 1])
        else:
            return self.interpolator(X)


class NetCDFEnvironment(Environment):
    """
    Environment loaded from NetCDF files (e.g., ROMS ocean data).
    
    Handles spatiotemporal datasets with time-varying fields.
    Suitable for:
    - ROMS Oregon Coast ocean simulations
    - Climate/weather data
    - Any NetCDF gridded dataset
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        netcdf_path: str,
        variable_name: str,
        time_index: int = 0,
        observation_noise: float = 0.1,
        seed: Optional[int] = None,
        spatial_coords: Tuple[str, str] = ('lon', 'lat'),
        time_coord: str = 'time'
    ):
        """
        Initialize NetCDF environment.
        
        Args:
            bounds: Spatial bounds
            netcdf_path: Path to NetCDF file
            variable_name: Name of variable to use as field
            time_index: Which time slice to use (for spatiotemporal data)
            observation_noise: Observation noise level
            seed: Random seed
            spatial_coords: Names of spatial coordinate variables (x, y)
            time_coord: Name of time coordinate variable
        """
        super().__init__(bounds, observation_noise, seed)
        self.netcdf_path = netcdf_path
        self.variable_name = variable_name
        self.time_index = time_index
        self.spatial_coords = spatial_coords
        self.time_coord = time_coord
        
        # Load data
        self._load_netcdf()
    
    def _load_netcdf(self):
        """Load NetCDF file and extract relevant slice."""
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 is required for NetCDFEnvironment. "
                            "Install with: pip install netCDF4")
        
        # Open dataset
        self.dataset = nc.Dataset(self.netcdf_path, 'r')
        
        # Extract coordinates
        x_coord = self.dataset.variables[self.spatial_coords[0]][:]
        y_coord = self.dataset.variables[self.spatial_coords[1]][:]
        
        # Extract field values at specified time
        variable = self.dataset.variables[self.variable_name]
        if self.time_coord in variable.dimensions:
            # Time-varying field
            self.field_values = variable[self.time_index, :, :]
        else:
            # Static field
            self.field_values = variable[:, :]
        
        # Create meshgrid
        if x_coord.ndim == 1 and y_coord.ndim == 1:
            # Regular grid
            self.X_grid, self.Y_grid = np.meshgrid(x_coord, y_coord)
        else:
            # Irregular grid
            self.X_grid = x_coord
            self.Y_grid = y_coord
        
        # Flatten for interpolation
        points = np.c_[self.X_grid.ravel(), self.Y_grid.ravel()]
        values = self.field_values.ravel()
        
        # Remove NaN values
        valid = ~np.isnan(values)
        self.data_points = points[valid]
        self.data_values = values[valid]
        
        # Create interpolator
        from scipy.interpolate import LinearNDInterpolator
        self.interpolator = LinearNDInterpolator(
            self.data_points,
            self.data_values,
            fill_value=np.nan
        )
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate using interpolation of NetCDF data."""
        X = np.atleast_2d(X)
        result = self.interpolator(X)
        
        # Handle NaN (outside data range)
        result = np.nan_to_num(result, nan=0.0)
        
        return result
    
    def set_time_index(self, time_index: int):
        """
        Change the time slice for spatiotemporal data.
        
        Args:
            time_index: New time index
        """
        self.time_index = time_index
        self._load_netcdf()  # Reload with new time


# Factory function
def create_environment(
    bounds: np.ndarray,
    env_type: str = 'synthetic',
    **kwargs
) -> Environment:
    """
    Factory function to create environments.
    
    Args:
        bounds: Spatial bounds
        env_type: Type of environment
            - 'synthetic': Synthetic test functions
            - 'interpolated': Real data with interpolation
            - 'netcdf': NetCDF file (e.g., ROMS ocean data)
        **kwargs: Additional parameters for environment
        
    Returns:
        Environment instance
        
    Examples:
        # Synthetic Townsend function
        env = create_environment(
            bounds=np.array([[-2.25, 2.5], [-2.5, 1.75]]),
            env_type='synthetic',
            function_name='townsend'
        )
        
        # Gaussian mixture for S&R scenario
        env = create_environment(
            bounds=np.array([[0, 100], [0, 100]]),
            env_type='synthetic',
            function_name='gaussian_mixture',
            n_components=9,
            seed=42
        )
        
        # Real data (e.g., LAMP lunar crater)
        env = create_environment(
            bounds=np.array([[0, 1], [0, 1]]),
            env_type='interpolated',
            data_points=lamp_positions,
            data_values=lamp_measurements,
            interpolation_method='rbf'
        )
        
        # ROMS ocean data
        env = create_environment(
            bounds=np.array([[-125, -123], [43, 45]]),
            env_type='netcdf',
            netcdf_path='roms_oregon_coast.nc',
            variable_name='temp',
            time_index=0
        )
    """
    if env_type.lower() == 'synthetic':
        return SyntheticEnvironment(bounds=bounds, **kwargs)
    elif env_type.lower() in ['real_data', 'interpolated']:
        return InterpolatedDataEnvironment(bounds=bounds, **kwargs)
    elif env_type.lower() == 'netcdf':
        return NetCDFEnvironment(bounds=bounds, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}. "
                       f"Available: 'synthetic', 'interpolated', 'netcdf'")
