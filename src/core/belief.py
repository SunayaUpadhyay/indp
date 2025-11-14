"""
Gaussian Process belief representation.

This module provides a unified interface for GP beliefs with different backends
(scikit-learn, GPy, GPyTorch, custom implementations).

Design rationale:
- Abstract interface allows swapping GP implementations
- Supports exact and sparse/approximate methods
- Efficient variance-only updates for MCTS simulation
- Kriging Believer virtual updates for coordination
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Literal
from abc import ABC, abstractmethod
import warnings


class GaussianProcessBelief(ABC):
    """
    Abstract base class for GP belief representations.
    
    This provides a common interface for different GP backends while
    allowing efficient implementations for each.
    
    IMPORTANT: GP internally works in NORMALIZED [0,1] coordinates for numerical
    stability. User-facing methods (predict, update) accept original coordinates
    and handle conversion automatically.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 0.1,
        prior_mean: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
        use_normalized_coords: bool = True
    ):
        """
        Initialize GP belief.
        
        Args:
            bounds: Spatial bounds [[x_min, x_max], [y_min, y_max]] in original coordinates
            kernel_type: Type of kernel ('rbf', 'matern', etc.)
            length_scale: Kernel length scale parameter (in normalized coords if use_normalized=True)
            variance: Kernel variance parameter
            noise: Observation noise level
            prior_mean: Prior mean value
            config: Additional configuration
            use_normalized_coords: If True, internally work in [0,1] space for numerical stability
        """
        self.bounds = np.array(bounds)
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.prior_mean = prior_mean
        self.config = config or {}
        self.use_normalized_coords = use_normalized_coords
        
        # Calculate normalization parameters
        self.coord_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.coord_min = self.bounds[:, 0]
        
        # Internal bounds (always [0,1] if normalized)
        if use_normalized_coords:
            self.internal_bounds = np.array([[0, 1], [0, 1]])
        else:
            self.internal_bounds = self.bounds.copy()
        
        # Training data (stored in INTERNAL coordinates for numerical stability)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
    
    def to_internal(self, X: np.ndarray) -> np.ndarray:
        """
        Convert from original coordinates to internal (possibly normalized).
        
        Args:
            X: Points in original coordinate system, shape (n, 2)
            
        Returns:
            Points in internal coordinate system, shape (n, 2)
        """
        if not self.use_normalized_coords:
            return X
        X = np.atleast_2d(X)
        return (X - self.coord_min) / self.coord_range
    
    def from_internal(self, X_internal: np.ndarray) -> np.ndarray:
        """
        Convert from internal coordinates to original.
        
        Args:
            X_internal: Points in internal system, shape (n, 2)
            
        Returns:
            Points in original coordinate system, shape (n, 2)
        """
        if not self.use_normalized_coords:
            return X_internal
        X_internal = np.atleast_2d(X_internal)
        return X_internal * self.coord_range + self.coord_min
        
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict mean and optionally standard deviation at test points.
        
        NOTE: Accepts X in ORIGINAL coordinates. Subclasses must convert to
        internal coordinates using self.to_internal(X) before GP operations.
        
        Args:
            X: Test points in ORIGINAL coordinates, shape (n_points, n_dims)
            return_std: Whether to return standard deviation
            
        Returns:
            mean: Predicted mean of shape (n_points,)
            std: Predicted std of shape (n_points,) if return_std, else None
        """
        pass
    
    @abstractmethod
    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        optimize: bool = False
    ) -> None:
        """
        Update GP with new observations.
        
        NOTE: Accepts X_new in ORIGINAL coordinates. Subclasses must convert to
        internal coordinates using self.to_internal(X_new) before storing.
        
        Args:
            X_new: New observation locations in ORIGINAL coordinates, shape (n_new, n_dims)
            y_new: New observation values of shape (n_new,)
            optimize: Whether to optimize hyperparameters
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'GaussianProcessBelief':
        """Create a deep copy of this GP belief."""
        pass
    
    def get_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictive variance at test points.
        
        Args:
            X: Test points of shape (n_points, n_dims)
            
        Returns:
            Variance of shape (n_points,)
        """
        _, std = self.predict(X, return_std=True)
        return std ** 2 if std is not None else np.zeros(len(X))
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictive standard deviation at test points.
        
        Args:
            X: Test points of shape (n_points, n_dims)
            
        Returns:
            Standard deviation of shape (n_points,)
        """
        _, std = self.predict(X, return_std=True)
        return std if std is not None else np.zeros(len(X))
    
    def kriging_believer_update(
        self,
        X_virtual: np.ndarray,
        inplace: bool = False
    ) -> Optional['GaussianProcessBelief']:
        """
        Perform Kriging Believer update (virtual observation at predicted mean).
        
        This is used for multi-robot coordination: pretend we observed the
        current predicted mean at a location to see how uncertainty would change.
        
        Args:
            X_virtual: Virtual observation locations of shape (n_virtual, n_dims)
            inplace: Whether to update this GP or return a new copy
            
        Returns:
            Updated GP belief (new copy if not inplace, else None)
        """
        # Predict mean at virtual locations
        y_virtual, _ = self.predict(X_virtual, return_std=False)
        
        # Choose which GP to update
        if inplace:
            self.update(X_virtual, y_virtual, optimize=False)
            return None
        else:
            gp_copy = self.copy()
            gp_copy.update(X_virtual, y_virtual, optimize=False)
            return gp_copy
    
    def variance_reduction(
        self,
        X_candidate: np.ndarray,
        X_eval: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute expected variance reduction from observing at candidate points.
        
        This is useful for acquisition functions and MCTS reward computation.
        
        Args:
            X_candidate: Candidate observation points of shape (n_cand, n_dims)
            X_eval: Points to evaluate variance reduction on (uses candidate if None)
            
        Returns:
            Total variance reduction (scalar)
        """
        if X_eval is None:
            X_eval = X_candidate
        
        # Current variance
        var_before = self.get_variance(X_eval)
        
        # Variance after virtual observation
        gp_after = self.kriging_believer_update(X_candidate, inplace=False)
        var_after = gp_after.get_variance(X_eval)
        
        # Total reduction
        return np.sum(var_before - var_after)
    
    @property
    def n_observations(self) -> int:
        """Number of observations in training set."""
        return 0 if self.X_train is None else len(self.X_train)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(kernel={self.kernel_type}, "
                f"n_obs={self.n_observations})")


class SKLearnGPBelief(GaussianProcessBelief):
    """
    Gaussian Process belief using scikit-learn backend.
    
    This is a simple, reliable implementation suitable for moderate-scale problems.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Import sklearn GP
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
        except ImportError:
            raise ImportError("scikit-learn is required for SKLearnGPBelief")
        
        # Create kernel
        if self.kernel_type.lower() == 'rbf':
            kernel = ConstantKernel(self.variance) * RBF(length_scale=self.length_scale)
        elif self.kernel_type.lower() == 'matern':
            kernel = ConstantKernel(self.variance) * Matern(
                length_scale=self.length_scale,
                nu=2.5  # Smoothness parameter
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Create GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise ** 2,  # Noise variance
            normalize_y=False,
            n_restarts_optimizer=2 if self.config.get('optimize_hyperparams', False) else 0
        )
        
        self._is_fitted = False
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict using scikit-learn GP.
        
        Accepts X in ORIGINAL coordinates, converts to internal for prediction.
        """
        X = np.atleast_2d(X)
        
        # Convert to internal coordinates
        X_internal = self.to_internal(X)
        
        if not self._is_fitted:
            # Return prior
            mean = np.full(len(X_internal), self.prior_mean)
            std = np.full(len(X_internal), np.sqrt(self.variance)) if return_std else None
            return mean, std
        
        if return_std:
            mean, std = self.gp.predict(X_internal, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(X_internal, return_std=False)
            return mean, None
    
    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        optimize: bool = False
    ) -> None:
        """
        Update scikit-learn GP with new data.
        
        Accepts X_new in ORIGINAL coordinates, converts to internal before storing.
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)
        
        # Convert to internal coordinates
        X_new_internal = self.to_internal(X_new)
        
        # Append to training data (stored in internal coordinates)
        if self.X_train is None:
            self.X_train = X_new_internal.copy()
            self.y_train = y_new.copy()
        else:
            self.X_train = np.vstack([self.X_train, X_new_internal])
            self.y_train = np.hstack([self.y_train, y_new])
        
        # Refit GP with internal coordinates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings
            self.gp.fit(self.X_train, self.y_train)
        
        self._is_fitted = True
    
    def copy(self) -> 'SKLearnGPBelief':
        """Create a deep copy."""
        import copy as copy_module
        return copy_module.deepcopy(self)
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about training data and coordinate system."""
        info = {
            'n_train': len(self.X_train) if self.X_train is not None else 0,
            'is_fitted': self._is_fitted,
            'use_normalized': self.use_normalized_coords,
            'bounds_original': self.bounds,
            'bounds_internal': self.internal_bounds,
            'coord_range': self.coord_range,
        }
        
        if self.X_train is not None:
            info['X_train_min'] = self.X_train.min(axis=0)
            info['X_train_max'] = self.X_train.max(axis=0)
            info['y_train_mean'] = self.y_train.mean()
            info['y_train_std'] = self.y_train.std()
        
        return info


# Factory function for creating GP beliefs
def create_gp_belief(
    bounds: np.ndarray,
    backend: Literal['sklearn', 'gpy', 'gpytorch'] = 'sklearn',
    **kwargs
) -> GaussianProcessBelief:
    """
    Factory function to create GP belief with specified backend.
    
    NOTE: By default, GP beliefs use normalized [0,1] coordinates internally
    for numerical stability. Pass use_normalized_coords=False to disable.
    
    Args:
        bounds: Spatial bounds in original coordinates
        backend: Which GP backend to use
        **kwargs: Additional parameters passed to GP constructor
        
    Returns:
        GaussianProcessBelief instance
        
    Example:
        >>> gp = create_gp_belief(
        ...     bounds=np.array([[-2.25, 2.5], [-2.5, 1.75]]),
        ...     kernel_type='matern',
        ...     length_scale=0.1,  # In normalized space
        ...     use_normalized_coords=True  # Default
        ... )
        >>> # User always passes original coordinates
        >>> gp.update(X_new=[[-2.0, -2.0]], y_new=[0.5])
        >>> mean, std = gp.predict([[0.0, 0.0]])
        
    Returns:
        GaussianProcessBelief instance
    """
    if backend.lower() == 'sklearn':
        return SKLearnGPBelief(bounds=bounds, **kwargs)
    elif backend.lower() == 'gpy':
        # TODO: Implement GPyBelief
        raise NotImplementedError("GPy backend not yet implemented")
    elif backend.lower() == 'gpytorch':
        # TODO: Implement GPyTorchBelief
        raise NotImplementedError("GPyTorch backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend}")
