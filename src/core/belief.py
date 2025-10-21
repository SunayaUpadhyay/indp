"""
Gaussian Process belief representation.

This module provides a unified interface for GP beliefs with different backends
(scikit-learn, GPy, GPyTorch, custom implementations).

Design rationale:
- Abstract interface allows swapping GP implementations
- Supports exact and sparse/approximate methods
- Efficient variance-only updates for MCTS simulation
-Kriging Believer virtual updates for coordination
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
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 0.1,
        prior_mean: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GP belief.
        
        Args:
            bounds: Spatial bounds [[x_min, x_max], [y_min, y_max]]
            kernel_type: Type of kernel ('rbf', 'matern', etc.)
            length_scale: Kernel length scale parameter
            variance: Kernel variance parameter
            noise: Observation noise level
            prior_mean: Prior mean value
            config: Additional configuration
        """
        self.bounds = np.array(bounds)
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.prior_mean = prior_mean
        self.config = config or {}
        
        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict mean and optionally standard deviation at test points.
        
        Args:
            X: Test points of shape (n_points, n_dims)
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
        
        Args:
            X_new: New observation locations of shape (n_new, n_dims)
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
        """Predict using scikit-learn GP."""
        X = np.atleast_2d(X)
        
        if not self._is_fitted:
            # Return prior
            mean = np.full(len(X), self.prior_mean)
            std = np.full(len(X), np.sqrt(self.variance)) if return_std else None
            return mean, std
        
        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(X, return_std=False)
            return mean, None
    
    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        optimize: bool = False
    ) -> None:
        """Update scikit-learn GP with new data."""
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)
        
        # Append to training data
        if self.X_train is None:
            self.X_train = X_new.copy()
            self.y_train = y_new.copy()
        else:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.hstack([self.y_train, y_new])
        
        # Refit GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings
            self.gp.fit(self.X_train, self.y_train)
        
        self._is_fitted = True
    
    def copy(self) -> 'SKLearnGPBelief':
        """Create a deep copy."""
        import copy as copy_module
        return copy_module.deepcopy(self)


# Factory function for creating GP beliefs
def create_gp_belief(
    bounds: np.ndarray,
    backend: Literal['sklearn', 'gpy', 'gpytorch'] = 'sklearn',
    **kwargs
) -> GaussianProcessBelief:
    """
    Factory function to create GP belief with specified backend.
    
    Args:
        bounds: Spatial bounds
        backend: Which GP backend to use
        **kwargs: Additional parameters passed to GP constructor
        
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
