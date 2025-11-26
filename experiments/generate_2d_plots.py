"""
Generate clear 2D trajectory plots for all planners.
Similar to the candidate generation and assignment demos.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.environment import create_environment


def load_results(results_dir='results/experiments'):
    """Load all experiment results."""
    results_path = Path(results_dir)
    results = []
    
    for json_file in results_path.glob('*_gaussian_hotspot_2r_150s_1042.json'):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    
    return sorted(results, key=lambda x: x['planner_name'])


def plot_planner_2d(result, env, ax):
    """Create 2D trajectory plot for a single planner."""
    planner_name = result['planner_name']
    bounds = np.array(result['bounds'])
    
    # Create background heatmap of true environment
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    Z = env.evaluate(grid).reshape(X.shape)
    
    # Plot environment
    im = ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.6)
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    
    # Get trajectories
    trajectories = result['robot_trajectories']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot each robot's trajectory
    for robot_id, traj in trajectories.items():
        if not traj:
            continue
        
        traj = np.array(traj)
        color = colors[int(robot_id) % len(colors)]
        
        # Plot trajectory line
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, 
                alpha=0.7, label=f'Robot {robot_id}')
        
        # Plot measurement points
        ax.scatter(traj[:, 0], traj[:, 1], c=color, s=100, 
                  edgecolors='white', linewidths=1.5, zorder=5, alpha=0.9)
        
        # Mark start position
        ax.scatter(traj[0, 0], traj[0, 1], c=color, s=300, 
                  marker='*', edgecolors='white', linewidths=2, 
                  zorder=10, label=f'Start R{robot_id}')
        
        # Mark end position
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=200, 
                  marker='X', edgecolors='white', linewidths=2, zorder=10)
    
    # Formatting
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Title with metrics
    title = f"{planner_name}\n"
    title += f"Measurements: {result['total_measurements']} | "
    title += f"Coverage: {result['coverage_fraction']*100:.1f}% | "
    title += f"Distance: {result['total_distance']:.0f}m"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate start labels
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), loc='upper right', 
             fontsize=9, framealpha=0.9)
    
    return im


def generate_2d_plots():
    """Generate 2D trajectory plots for all planners."""
    
    print("\n" + "="*70)
    print("GENERATING 2D TRAJECTORY PLOTS")
    print("="*70 + "\n")
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No results found! Run experiments first.")
        return
    
    print(f"Found {len(results)} planners to plot\n")
    
    # Create environment for background
    env = create_environment(
        bounds=np.array([[0.0, 100.0], [0.0, 100.0]]),
        env_type='synthetic',
        function_name='gaussian_mixture',
        observation_noise=0.1,
        seed=42,
        physical_scale=1.0
    )
    
    # Create individual plots for each planner
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        planner_name = result['planner_name']
        
        fig, ax = plt.subplots(figsize=(10, 9))
        im = plot_planner_2d(result, env, ax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Field Value', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = output_dir / f'2d_{planner_name.lower()}_trajectory.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    # Create comparison grid
    n_planners = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
        im = plot_planner_2d(result, env, axes[idx])
    
    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    # Add overall colorbar
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, 
                label='Field Value')
    
    plt.suptitle('Multi-Robot Planner Comparison\n2 Robots, 150s Budget, Gaussian Hotspot Scenario',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = output_dir / '2d_comparison_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved: {output_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ All 2D plots generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  • Individual trajectory plots: 2d_<planner>_trajectory.png")
    print("  • Comparison grid: 2d_comparison_grid.png")
    print()


if __name__ == '__main__':
    generate_2d_plots()
