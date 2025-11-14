"""
Test suite for Environment coordinate and physical distance conversions.

Tests:
- Physical scale conversion (coord_to_meters, meters_to_coord)
- Normalized coordinates (to_normalized, from_normalized)
- Physical info calculations
- Different coordinate systems (negative, positive, mixed ranges)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.environment import SyntheticEnvironment


class TestEnvironmentCoordinateConversion:
    """Test coordinate conversions in different coordinate systems."""
    
    def test_positive_coords_scaling(self):
        """Test environment with positive coordinate range [0, 100] x [0, 100]."""
        bounds = np.array([[0, 100], [0, 100]])
        physical_scale = 10.0  # Each coord unit = 10 meters
        
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='peaks',
            physical_scale=physical_scale
        )
        
        # Check coordinate range
        assert np.allclose(env.coord_range, [100, 100])
        
        # Check physical size
        assert np.allclose(env.physical_size, [1000, 1000])  # 100 * 10 = 1000m
        assert np.allclose(env.physical_area, 1_000_000)  # 1km²
        
        # Test distance conversion
        coord_dist = 10.0  # 10 coordinate units
        physical_dist = env.coord_to_meters(coord_dist)
        assert np.allclose(physical_dist, 100.0)  # 10 * 10 = 100 meters
        
        # Test reverse conversion
        back_to_coord = env.meters_to_coord(physical_dist)
        assert np.allclose(back_to_coord, coord_dist)
    
    def test_negative_coords_scaling(self):
        """Test environment with negative coordinate range [-50, -10] x [-30, -5]."""
        bounds = np.array([[-50, -10], [-30, -5]])
        physical_scale = 2.0  # Each coord unit = 2 meters
        
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='sphere',
            physical_scale=physical_scale
        )
        
        # Check coordinate range (should be positive)
        assert np.allclose(env.coord_range, [40, 25])  # |-10 - (-50)| = 40, |-5 - (-30)| = 25
        
        # Check physical size
        assert np.allclose(env.physical_size, [80, 50])  # 40*2 = 80m, 25*2 = 50m
        assert np.allclose(env.physical_area, 4000)  # 80 * 50 = 4000m²
        
        # Test distance conversion
        coord_dist = 5.0
        physical_dist = env.coord_to_meters(coord_dist)
        assert np.allclose(physical_dist, 10.0)  # 5 * 2 = 10 meters
    
    def test_mixed_coords_scaling(self):
        """Test environment with mixed range [-2.25, 2.5] x [-2.5, 1.75] (Townsend)."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        physical_scale = 100.0  # Each coord unit = 100 meters
        
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='townsend',
            physical_scale=physical_scale
        )
        
        # Check coordinate range
        expected_range = [4.75, 4.25]  # 2.5 - (-2.25) = 4.75, 1.75 - (-2.5) = 4.25
        assert np.allclose(env.coord_range, expected_range)
        
        # Check physical size
        assert np.allclose(env.physical_size, [475, 425])  # meters
        assert np.allclose(env.physical_area, 201875)  # 475 * 425 = 201,875 m²
        
        # Test distance conversion
        coord_dist = 1.0
        physical_dist = env.coord_to_meters(coord_dist)
        assert np.allclose(physical_dist, 100.0)
        
        # Test diagonal distance
        diagonal_coord = np.sqrt(4.75**2 + 4.25**2)
        diagonal_physical = env.coord_to_meters(diagonal_coord)
        assert np.allclose(diagonal_physical, diagonal_coord * 100.0)


class TestEnvironmentNormalization:
    """Test coordinate normalization to [0, 1] space."""
    
    def test_normalization_positive_coords(self):
        """Test normalization with positive coordinates."""
        bounds = np.array([[0, 100], [0, 50]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='peaks',
            use_normalized_coords=True
        )
        
        # Test corner points
        corners = np.array([
            [0, 0],      # Min corner
            [100, 50],   # Max corner
            [50, 25],    # Center
            [0, 50],     # Top-left
            [100, 0]     # Bottom-right
        ])
        
        normalized = env.to_normalized(corners)
        
        assert np.allclose(normalized[0], [0, 0])
        assert np.allclose(normalized[1], [1, 1])
        assert np.allclose(normalized[2], [0.5, 0.5])
        assert np.allclose(normalized[3], [0, 1])
        assert np.allclose(normalized[4], [1, 0])
        
        # Test round-trip conversion
        back = env.from_normalized(normalized)
        assert np.allclose(back, corners)
    
    def test_normalization_negative_coords(self):
        """Test normalization with negative coordinates."""
        bounds = np.array([[-50, -10], [-30, -5]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='sphere',
            use_normalized_coords=True
        )
        
        # Test specific points
        points = np.array([
            [-50, -30],  # Min corner -> [0, 0]
            [-10, -5],   # Max corner -> [1, 1]
            [-30, -17.5] # Center -> [0.5, 0.5]
        ])
        
        normalized = env.to_normalized(points)
        
        assert np.allclose(normalized[0], [0, 0])
        assert np.allclose(normalized[1], [1, 1])
        assert np.allclose(normalized[2], [0.5, 0.5])
        
        # Round-trip
        back = env.from_normalized(normalized)
        assert np.allclose(back, points)
    
    def test_normalization_mixed_coords(self):
        """Test normalization with mixed positive/negative coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='townsend',
            use_normalized_coords=True
        )
        
        # Test points
        points = np.array([
            [-2.25, -2.5],   # Min corner
            [2.5, 1.75],     # Max corner
            [0.125, -0.375], # Center: (-2.25+2.5)/2, (-2.5+1.75)/2
        ])
        
        normalized = env.to_normalized(points)
        
        assert np.allclose(normalized[0], [0, 0])
        assert np.allclose(normalized[1], [1, 1])
        assert np.allclose(normalized[2], [0.5, 0.5])
        
        # Round-trip
        back = env.from_normalized(normalized)
        assert np.allclose(back, points)


class TestEnvironmentPhysicalInfo:
    """Test get_physical_info() method."""
    
    def test_physical_info_complete(self):
        """Test that physical info contains all expected keys."""
        bounds = np.array([[0, 100], [0, 100]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='peaks',
            physical_scale=10.0
        )
        
        info = env.get_physical_info()
        
        # Check all keys exist
        required_keys = [
            'coord_bounds', 'coord_size', 'physical_scale',
            'physical_size_m', 'physical_area_m2', 'physical_area_km2',
            'diagonal_m', 'diagonal_km', 'use_normalized'
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        # Check values
        assert np.allclose(info['coord_size'], [100, 100])
        assert info['physical_scale'] == 10.0
        assert np.allclose(info['physical_size_m'], [1000, 1000])
        assert np.allclose(info['physical_area_m2'], 1_000_000)
        assert np.allclose(info['physical_area_km2'], 1.0)
        
        # Check diagonal
        expected_diagonal_m = np.sqrt(1000**2 + 1000**2)
        assert np.allclose(info['diagonal_m'], expected_diagonal_m)
        assert np.allclose(info['diagonal_km'], expected_diagonal_m / 1000)
    
    def test_physical_info_different_scales(self):
        """Test physical info with different scales."""
        test_cases = [
            {
                'bounds': np.array([[0, 100], [0, 100]]),
                'scale': 1.0,
                'expected_area_m2': 10_000,
                'expected_area_km2': 0.01
            },
            {
                'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
                'scale': 100.0,
                'expected_area_m2': 201_875,
                'expected_area_km2': 0.201875
            },
            {
                'bounds': np.array([[0, 1000], [0, 1000]]),
                'scale': 0.5,
                'expected_area_m2': 250_000,
                'expected_area_km2': 0.25
            }
        ]
        
        for case in test_cases:
            env = SyntheticEnvironment(
                bounds=case['bounds'],
                function_name='peaks',
                physical_scale=case['scale']
            )
            
            info = env.get_physical_info()
            assert np.allclose(info['physical_area_m2'], case['expected_area_m2'])
            assert np.allclose(info['physical_area_km2'], case['expected_area_km2'], rtol=1e-5)


class TestEnvironmentBoundsChecking:
    """Test is_within_bounds() method."""
    
    def test_within_bounds_simple(self):
        """Test point checking for simple bounds."""
        bounds = np.array([[0, 100], [0, 100]])
        env = SyntheticEnvironment(bounds=bounds, function_name='peaks')
        
        # Points inside
        inside = np.array([
            [50, 50],
            [0, 0],
            [100, 100],
            [25, 75]
        ])
        assert np.all(env.is_within_bounds(inside))
        
        # Points outside
        outside = np.array([
            [-1, 50],
            [50, 101],
            [101, 101],
            [-10, -10]
        ])
        assert not np.any(env.is_within_bounds(outside))
    
    def test_within_bounds_negative_coords(self):
        """Test point checking with negative coordinates."""
        bounds = np.array([[-2.25, 2.5], [-2.5, 1.75]])
        env = SyntheticEnvironment(bounds=bounds, function_name='townsend')
        
        # Points inside
        inside = np.array([
            [0, 0],
            [-2.25, -2.5],
            [2.5, 1.75],
            [-1, 0.5]
        ])
        assert np.all(env.is_within_bounds(inside))
        
        # Points outside
        outside = np.array([
            [-3, 0],
            [0, 2],
            [3, 3],
            [-2.3, 0]
        ])
        assert not np.any(env.is_within_bounds(outside))


class TestEnvironmentEvaluation:
    """Test environment evaluation methods."""
    
    def test_evaluate_single_point(self):
        """Test evaluation at single points."""
        bounds = np.array([[0, 10], [0, 10]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='sphere',
            observation_noise=0.0
        )
        
        # Sphere function: f(x,y) = x^2 + y^2
        point = np.array([[3, 4]])
        value = env.evaluate(point)
        assert np.allclose(value, [25])  # 3^2 + 4^2 = 25
    
    def test_evaluate_multiple_points(self):
        """Test evaluation at multiple points."""
        bounds = np.array([[-5, 5], [-5, 5]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='sphere',
            observation_noise=0.0
        )
        
        points = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [3, 4]
        ])
        values = env.evaluate(points)
        
        expected = [0, 1, 1, 25]
        assert np.allclose(values, expected)
    
    def test_observe_with_noise(self):
        """Test that observe() adds noise."""
        bounds = np.array([[0, 10], [0, 10]])
        env = SyntheticEnvironment(
            bounds=bounds,
            function_name='sphere',
            observation_noise=0.1,
            seed=42
        )
        
        point = np.array([[3, 4]])
        true_value = env.evaluate(point)
        observed_value = env.observe(point)
        
        # Should be close but not exact
        assert not np.allclose(true_value, observed_value)
        assert np.abs(true_value - observed_value) < 1.0  # Within reasonable noise range


def test_environment_consistency():
    """Test that all conversions are consistent across different coordinate systems."""
    test_configs = [
        {'bounds': np.array([[0, 100], [0, 100]]), 'scale': 10.0},
        {'bounds': np.array([[-50, 50], [-50, 50]]), 'scale': 1.0},
        {'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]), 'scale': 100.0},
    ]
    
    for config in test_configs:
        env = SyntheticEnvironment(
            bounds=config['bounds'],
            function_name='peaks',
            physical_scale=config['scale'],
            use_normalized_coords=True
        )
        
        # Test that physical_size = coord_range * physical_scale
        expected_physical_size = env.coord_range * config['scale']
        assert np.allclose(env.physical_size, expected_physical_size)
        
        # Test round-trip: meters -> coord -> meters
        test_distances = [1.0, 10.0, 100.0, 1000.0]
        for dist_m in test_distances:
            dist_coord = env.meters_to_coord(dist_m)
            dist_m_back = env.coord_to_meters(dist_coord)
            assert np.allclose(dist_m, dist_m_back)
        
        # Test round-trip: coord -> normalized -> coord
        test_points = np.array([
            env.bounds[:, 0],  # Min corner
            env.bounds[:, 1],  # Max corner
            (env.bounds[:, 0] + env.bounds[:, 1]) / 2  # Center
        ])
        normalized = env.to_normalized(test_points)
        back = env.from_normalized(normalized)
        assert np.allclose(back, test_points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
