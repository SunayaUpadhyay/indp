"""
Test suite for GaussianProcessBelief coordinate normalization and conversions.

Tests:
- Internal coordinate normalization
- Coordinate conversion (to_internal, from_internal)
- Prediction and update with original coordinates
- Training data stored in internal coordinates
- Different coordinate systems (negative, positive, mixed)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.belief import SKLearnGPBelief, create_gp_belief


class TestGPCoordinateNormalization:
    """Test GP coordinate normalization to [0, 1] space."""
    
    def test_normalization_enabled(self):
        """Test that normalization is enabled by default."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        assert gp.use_normalized_coords is True
        assert np.allclose(gp.internal_bounds, [[0, 1], [0, 1]])
    
    def test_normalization_disabled(self):
        """Test that normalization can be disabled."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=False)
        
        assert gp.use_normalized_coords is False
        assert np.allclose(gp.internal_bounds, bounds)
    
    def test_to_internal_positive_coords(self):
        """Test conversion to internal coordinates with positive range."""
        bounds = np.array([[0, 100], [0, 50]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Test corner points
        points = np.array([
            [0, 0],
            [100, 50],
            [50, 25],
            [25, 12.5]
        ])
        
        internal = gp.to_internal(points)
        
        assert np.allclose(internal[0], [0, 0])
        assert np.allclose(internal[1], [1, 1])
        assert np.allclose(internal[2], [0.5, 0.5])
        assert np.allclose(internal[3], [0.25, 0.25])
    
    def test_to_internal_negative_coords(self):
        """Test conversion with negative coordinates."""
        bounds = np.array([[-50, -10], [-30, -5]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        points = np.array([
            [-50, -30],  # Min -> [0, 0]
            [-10, -5],   # Max -> [1, 1]
            [-30, -17.5] # Center -> [0.5, 0.5]
        ])
        
        internal = gp.to_internal(points)
        
        assert np.allclose(internal[0], [0, 0])
        assert np.allclose(internal[1], [1, 1])
        assert np.allclose(internal[2], [0.5, 0.5])
    
    def test_to_internal_mixed_coords(self):
        """Test conversion with mixed positive/negative coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        points = np.array([
            [-2.25, -2.5],
            [2.5, 1.75],
            [0.125, -0.375]  # Center
        ])
        
        internal = gp.to_internal(points)
        
        assert np.allclose(internal[0], [0, 0])
        assert np.allclose(internal[1], [1, 1])
        assert np.allclose(internal[2], [0.5, 0.5])
    
    def test_round_trip_conversion(self):
        """Test that to_internal and from_internal are inverses."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        original_points = np.array([
            [-2.25, -2.5],
            [0.0, 0.0],
            [2.5, 1.75],
            [-1.0, 0.5],
            [1.5, -1.0]
        ])
        
        internal = gp.to_internal(original_points)
        back = gp.from_internal(internal)
        
        assert np.allclose(back, original_points)
        
        # Check that internal is in [0, 1]
        assert np.all(internal >= 0)
        assert np.all(internal <= 1)


class TestGPPredictWithCoordinates:
    """Test GP prediction with coordinate conversion."""
    
    def test_predict_before_training(self):
        """Test prediction returns prior before any training."""
        bounds = np.array([[0, 100], [0, 100]])
        prior_mean = 5.0
        gp = SKLearnGPBelief(
            bounds=bounds,
            prior_mean=prior_mean,
            use_normalized_coords=True
        )
        
        # Predict at random points
        test_points = np.array([[25, 25], [50, 50], [75, 75]])
        mean, std = gp.predict(test_points, return_std=True)
        
        # Should return prior mean and prior std
        assert np.allclose(mean, prior_mean)
        assert std is not None
        assert np.all(std > 0)
    
    def test_predict_accepts_original_coords(self):
        """Test that predict() accepts original coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Train with some data (in original coordinates)
        X_train = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 1.0]])
        y_train = np.array([1.0, 2.0, 1.5])
        gp.update(X_train, y_train)
        
        # Predict at original coordinates
        X_test = np.array([[-1.0, -1.0], [1.0, 0.5]])
        mean, std = gp.predict(X_test, return_std=True)
        
        # Should work without errors
        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
    
    def test_predict_consistency_across_coords(self):
        """Test that predictions are consistent regardless of coordinate system."""
        # Two GPs with different coordinate systems but same physical setup
        gp1 = SKLearnGPBelief(
            bounds=np.array([[0, 10], [0, 10]]),
            use_normalized_coords=True,
            noise=0.01,
            length_scale=0.2
        )
        
        gp2 = SKLearnGPBelief(
            bounds=np.array([[0, 100], [0, 100]]),
            use_normalized_coords=True,
            noise=0.01,
            length_scale=0.2
        )
        
        # Train both with proportional data
        X1 = np.array([[2, 2], [5, 5], [8, 8]])
        X2 = np.array([[20, 20], [50, 50], [80, 80]])
        y = np.array([1.0, 2.0, 1.5])
        
        gp1.update(X1, y)
        gp2.update(X2, y)
        
        # Predict at center (proportional points)
        mean1, std1 = gp1.predict([[5, 5]])
        mean2, std2 = gp2.predict([[50, 50]])
        
        # Should give similar predictions (values are not coordinates)
        assert np.allclose(mean1, mean2, rtol=0.1)
        assert np.allclose(std1, std2, rtol=0.1)


class TestGPUpdateWithCoordinates:
    """Test GP update with coordinate conversion."""
    
    def test_update_accepts_original_coords(self):
        """Test that update() accepts original coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Update with original coordinates
        X_new = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 1.0]])
        y_new = np.array([1.0, 2.0, 1.5])
        
        gp.update(X_new, y_new)
        
        # Check training data is stored
        assert gp.X_train is not None
        assert gp.y_train is not None
        assert len(gp.X_train) == 3
        assert len(gp.y_train) == 3
    
    def test_training_data_stored_in_internal_coords(self):
        """Test that training data is stored in internal (normalized) coordinates."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Update with original coordinates
        X_original = np.array([[0, 0], [50, 50], [100, 100]])
        y = np.array([1.0, 2.0, 3.0])
        
        gp.update(X_original, y)
        
        # Training data should be in [0, 1] range
        assert np.all(gp.X_train >= 0)
        assert np.all(gp.X_train <= 1)
        
        # Check specific values
        assert np.allclose(gp.X_train[0], [0, 0])
        assert np.allclose(gp.X_train[1], [0.5, 0.5])
        assert np.allclose(gp.X_train[2], [1, 1])
        
        # y values should be unchanged
        assert np.allclose(gp.y_train, y)
    
    def test_update_values_not_normalized(self):
        """Test that y values are NOT normalized."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Update with data
        X = np.array([[25, 25], [75, 75]])
        y = np.array([100.0, 200.0])  # Large values
        
        gp.update(X, y)
        
        # y values should be stored as-is
        assert np.allclose(gp.y_train, y)
        
        # Predictions should also be in original scale
        mean, _ = gp.predict(X)
        assert np.allclose(mean, y, rtol=0.1)  # Should predict close to training data
    
    def test_incremental_updates(self):
        """Test that multiple updates accumulate data."""
        bounds = np.array([[0, 10], [0, 10]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # First update
        gp.update(np.array([[1, 1]]), np.array([1.0]))
        assert gp.n_observations == 1
        
        # Second update
        gp.update(np.array([[5, 5]]), np.array([2.0]))
        assert gp.n_observations == 2
        
        # Third update
        gp.update(np.array([[9, 9]]), np.array([1.5]))
        assert gp.n_observations == 3
        
        # All data should be stored
        assert len(gp.X_train) == 3
        assert len(gp.y_train) == 3


class TestGPTrainingInfo:
    """Test get_training_info() method."""
    
    def test_training_info_structure(self):
        """Test that training info contains expected keys."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        info = gp.get_training_info()
        
        required_keys = [
            'n_train', 'is_fitted', 'use_normalized',
            'bounds_original', 'bounds_internal', 'coord_range'
        ]
        for key in required_keys:
            assert key in info
    
    def test_training_info_before_update(self):
        """Test training info before any data."""
        bounds = np.array([[0, 100], [0, 100]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        info = gp.get_training_info()
        
        assert info['n_train'] == 0
        assert info['is_fitted'] is False
        assert info['use_normalized'] is True
    
    def test_training_info_after_update(self):
        """Test training info after adding data."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        X = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 1.0]])
        y = np.array([1.0, 2.0, 1.5])
        gp.update(X, y)
        
        info = gp.get_training_info()
        
        assert info['n_train'] == 3
        assert info['is_fitted'] is True
        assert np.allclose(info['bounds_original'], bounds)
        assert np.allclose(info['bounds_internal'], [[0, 1], [0, 1]])
        
        # Check training data statistics
        assert 'X_train_min' in info
        assert 'X_train_max' in info
        assert 'y_train_mean' in info
        assert 'y_train_std' in info
        
        assert np.allclose(info['y_train_mean'], np.mean(y))


class TestGPKrigingBeliever:
    """Test Kriging Believer updates."""
    
    def test_kriging_believer_virtual_update(self):
        """Test that Kriging Believer reduces uncertainty."""
        bounds = np.array([[0, 10], [0, 10]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Train with some data
        X_train = np.array([[1, 1], [9, 9]])
        y_train = np.array([1.0, 2.0])
        gp.update(X_train, y_train)
        
        # Virtual observation point
        X_virtual = np.array([[5, 5]])
        
        # Get uncertainty before
        _, std_before = gp.predict(X_virtual, return_std=True)
        
        # Apply Kriging Believer
        gp_after = gp.kriging_believer_update(X_virtual, inplace=False)
        
        # Get uncertainty after
        _, std_after = gp_after.predict(X_virtual, return_std=True)
        
        # Uncertainty should decrease
        assert std_after[0] < std_before[0]
    
    def test_kriging_believer_accepts_original_coords(self):
        """Test that Kriging Believer accepts original coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        gp = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Train
        X_train = np.array([[-2.0, -2.0], [2.0, 1.0]])
        y_train = np.array([1.0, 2.0])
        gp.update(X_train, y_train)
        
        # Virtual update with original coordinates
        X_virtual = np.array([[0.0, 0.0]])
        gp_updated = gp.kriging_believer_update(X_virtual, inplace=False)
        
        # Should work without errors
        assert gp_updated.n_observations == 3  # Original 2 + virtual 1


class TestGPCopy:
    """Test GP copy functionality."""
    
    def test_copy_creates_independent_instance(self):
        """Test that copy creates an independent GP."""
        bounds = np.array([[0, 10], [0, 10]])
        gp1 = SKLearnGPBelief(bounds=bounds, use_normalized_coords=True)
        
        # Train original
        gp1.update(np.array([[1, 1]]), np.array([1.0]))
        
        # Copy
        gp2 = gp1.copy()
        
        # Modify copy
        gp2.update(np.array([[5, 5]]), np.array([2.0]))
        
        # Original should be unchanged
        assert gp1.n_observations == 1
        assert gp2.n_observations == 2


class TestGPFactoryFunction:
    """Test create_gp_belief factory function."""
    
    def test_factory_creates_sklearn_gp(self):
        """Test factory creates SKLearnGPBelief."""
        bounds = np.array([[0, 10], [0, 10]])
        gp = create_gp_belief(bounds=bounds, backend='sklearn')
        
        assert isinstance(gp, SKLearnGPBelief)
    
    def test_factory_default_normalization(self):
        """Test that factory enables normalization by default."""
        bounds = np.array([[0, 10], [0, 10]])
        gp = create_gp_belief(bounds=bounds, backend='sklearn')
        
        assert gp.use_normalized_coords is True


def test_gp_end_to_end_consistency():
    """Test complete workflow with different coordinate systems."""
    test_configs = [
        {'bounds': np.array([[0, 100], [0, 100]])},
        {'bounds': np.array([[-50, 50], [-50, 50]])},
        {'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]])},
    ]
    
    for config in test_configs:
        gp = SKLearnGPBelief(
            bounds=config['bounds'],
            use_normalized_coords=True,
            noise=0.01
        )
        
        # Generate training data
        bounds = config['bounds']
        X_train = np.array([
            bounds[:, 0],  # Min corner
            bounds[:, 1],  # Max corner
            (bounds[:, 0] + bounds[:, 1]) / 2  # Center
        ])
        y_train = np.array([1.0, 3.0, 2.0])
        
        # Update
        gp.update(X_train, y_train)
        
        # Predict
        mean, std = gp.predict(X_train)
        
        # Should predict close to training data
        assert np.allclose(mean, y_train, atol=0.1)
        
        # Check internal coordinates are normalized
        assert np.all(gp.X_train >= 0)
        assert np.all(gp.X_train <= 1)
        
        # Check y values are unchanged
        assert np.allclose(gp.y_train, y_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
