"""
Example usage of different environment types.

This script demonstrates how to create and use the various benchmark
environments available in the IPP framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.environment import create_environment


def demo_synthetic_environments():
    """Demonstrate synthetic test functions."""
    print("=" * 60)
    print("SYNTHETIC ENVIRONMENTS")
    print("=" * 60)
    
    # 1. Peaks function
    print("\n1. Peaks Function (2D MATLAB peaks)")
    env_peaks = create_environment(
        bounds=np.array([[-3, 3], [-3, 3]]),
        env_type='synthetic',
        function_name='peaks',
        observation_noise=0.1,
        seed=42
    )
    X, Y, Z = env_peaks.evaluate_grid(resolution=100)
    print(f"   Domain: {env_peaks.bounds}")
    print(f"   Value range: [{Z.min():.2f}, {Z.max():.2f}]")
    
    # 2. Townsend function (for testing local minima escape)
    print("\n2. Townsend Function (2D, multiple local minima)")
    env_townsend = create_environment(
        bounds=np.array([[-2.25, 2.5], [-2.5, 1.75]]),
        env_type='synthetic',
        function_name='townsend',
        observation_noise=0.05,
        seed=42
    )
    test_point = np.array([[-0.0299, -0.1151]])  # Near global minimum
    value = env_townsend.evaluate(test_point)
    print(f"   Domain: {env_townsend.bounds}")
    print(f"   Value at global min: {value[0]:.4f}")
    
    # 3. Gaussian Mixture (Search & Rescue scenario)
    print("\n3. Gaussian Mixture (S&R scenario with survivor clusters)")
    env_sar = create_environment(
        bounds=np.array([[0, 100], [0, 100]]),
        env_type='synthetic',
        function_name='gaussian_mixture',
        n_components=9,  # 9 survivor clusters
        observation_noise=0.01,
        seed=42
    )
    print(f"   Domain: {env_sar.bounds}")
    print(f"   Number of clusters: 9")
    if hasattr(env_sar.function.__self__, '_gmm_params_cached'):
        params = env_sar.function.__self__._gmm_params_cached
        print(f"   Cluster centers:\n{params['means']}")
    
    # 4. Ackley function (highly multimodal)
    print("\n4. Ackley Function (highly multimodal)")
    env_ackley = create_environment(
        bounds=np.array([[-5, 5], [-5, 5]]),
        env_type='synthetic',
        function_name='ackley',
        observation_noise=0.1,
        seed=42
    )
    print(f"   Domain: {env_ackley.bounds}")
    X, Y, Z = env_ackley.evaluate_grid(resolution=100)
    print(f"   Value range: [{Z.min():.2f}, {Z.max():.2f}]")
    
    return env_peaks, env_townsend, env_sar, env_ackley


def demo_interpolated_environment():
    """Demonstrate interpolated data environment (for real data)."""
    print("\n" + "=" * 60)
    print("INTERPOLATED DATA ENVIRONMENT")
    print("=" * 60)
    
    # Simulate some scattered data points (like LAMP lunar data)
    print("\nSimulating scattered data (like LAMP lunar crater data)")
    np.random.seed(42)
    n_points = 100
    data_points = np.random.uniform([0, 0], [10, 10], size=(n_points, 2))
    
    # Generate values from a simple function + noise
    data_values = (np.sin(data_points[:, 0]) * np.cos(data_points[:, 1]) + 
                   np.random.normal(0, 0.2, n_points))
    
    env_interp = create_environment(
        bounds=np.array([[0, 10], [0, 10]]),
        env_type='interpolated',
        data_points=data_points,
        data_values=data_values,
        observation_noise=0.1,
        interpolation_method='rbf',  # Try 'linear', 'cubic', 'nearest', 'rbf'
        seed=42
    )
    
    print(f"   Number of data points: {len(data_points)}")
    print(f"   Interpolation method: RBF")
    print(f"   Domain: {env_interp.bounds}")
    
    # Test interpolation
    test_points = np.array([[5, 5], [2, 3], [8, 7]])
    interpolated_values = env_interp.evaluate(test_points)
    print(f"   Interpolated values at test points: {interpolated_values}")
    
    return env_interp


def demo_netcdf_environment():
    """
    Demonstrate NetCDF environment (requires actual NetCDF file).
    
    This is a placeholder - you need to download ROMS data first.
    """
    print("\n" + "=" * 60)
    print("NetCDF ENVIRONMENT (ROMS Ocean Data)")
    print("=" * 60)
    
    print("\nTo use NetCDF environment:")
    print("1. Download ROMS Oregon Coast data from:")
    print("   https://data.ioos.us/dataset/regional-ocean-modeling-system-roms-oregon-coast")
    print("2. Example usage:")
    print("""
    env_roms = create_environment(
        bounds=np.array([[-125, -123], [43, 45]]),
        env_type='netcdf',
        netcdf_path='path/to/roms_oregon.nc',
        variable_name='temp',  # or 'salt', 'u', 'v', etc.
        time_index=0,
        spatial_coords=('lon', 'lat'),
        observation_noise=0.1
    )
    
    # For spatiotemporal planning, change time:
    env_roms.set_time_index(12)  # Move to 12th time step
    """)
    
    print("\nNote: NetCDF4 must be installed: pip install netCDF4")


def visualize_environments(envs, titles):
    """Visualize multiple environments side by side."""
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(5*n_envs, 4))
    
    if n_envs == 1:
        axes = [axes]
    
    for i, (env, title) in enumerate(zip(envs, titles)):
        X, Y, Z = env.evaluate_grid(resolution=100)
        
        ax = axes[i]
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    plt.savefig('results/environment_examples.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: results/environment_examples.png")
    plt.show()


def test_noisy_observations():
    """Demonstrate noisy observations vs ground truth."""
    print("\n" + "=" * 60)
    print("NOISY OBSERVATIONS")
    print("=" * 60)
    
    env = create_environment(
        bounds=np.array([[0, 10], [0, 10]]),
        env_type='synthetic',
        function_name='peaks',
        observation_noise=0.5,  # Relatively high noise
        seed=42
    )
    
    test_points = np.array([[5, 5], [2, 2], [8, 8]])
    
    print("\nComparing ground truth vs noisy observations:")
    for point in test_points:
        ground_truth = env.evaluate(point.reshape(1, -1))[0]
        noisy_obs = env.observe(point.reshape(1, -1))[0]
        print(f"   Point {point}: Ground truth = {ground_truth:.3f}, "
              f"Observed = {noisy_obs:.3f}, Noise = {noisy_obs - ground_truth:.3f}")


if __name__ == '__main__':
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    print("IPP Framework - Environment Examples")
    print("=" * 60)
    
    # Demo 1: Synthetic environments
    env_peaks, env_townsend, env_sar, env_ackley = demo_synthetic_environments()
    
    # Demo 2: Interpolated data
    env_interp = demo_interpolated_environment()
    
    # Demo 3: NetCDF (instructions only)
    demo_netcdf_environment()
    
    # Demo 4: Noisy observations
    test_noisy_observations()
    
    # Visualize
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    envs_to_plot = [env_peaks, env_townsend, env_sar, env_ackley]
    titles = ['Peaks', 'Townsend', 'Gaussian Mixture (S&R)', 'Ackley']
    
    visualize_environments(envs_to_plot, titles)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Created {len(envs_to_plot)} synthetic environments")
    print(f"✓ Created 1 interpolated data environment")
    print(f"✓ Demonstrated noisy observations")
    print(f"\nSee BENCHMARKS.md for full documentation of all environments and baselines.")
