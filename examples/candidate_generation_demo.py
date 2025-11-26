"""
Demo script for Step A: Candidate Generation

This demonstrates the adaptive quadtree-based candidate generation system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from src.core.environment import create_environment
from src.core.belief import create_gp_belief
from src.core.robot import Robot, BudgetType
from src.planning.candidates import CandidateGenerator


# Visualization constants
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
ROBOT_SIZE, ROBOT_EDGE, ROBOT_FONT = 100, 1.5, 5
CIRCLE_LINE, CIRCLE_ALPHA = 1.2, 0.6


def setup_plot_style():
    """Configure matplotlib styling."""
    plt.rcParams.update({
        'font.size': 11, 'axes.titleweight': 'bold',
        'axes.grid': False, 'figure.facecolor': 'white'
    })


def plot_3d_surface(ax, X, Y, Z, points, z_vals, cmap, title, bounds):
    """Plot 3D surface with observation markers."""
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, alpha=0.95,
                          vmin=Z.min(), vmax=Z.max())
    ax.scatter(points[:, 0], points[:, 1], z_vals, c='red', s=40,
              marker='o', edgecolors='black', linewidths=1.5, zorder=10)
    ax.set(xlabel='X', ylabel='Y', zlabel=title.split()[0],
           xlim=bounds[0], ylim=bounds[1], title=title)
    ax.view_init(25, 220)
    ax.grid(alpha=0.3)
    return surf


def draw_robot(ax, robot, color, label=None):
    """Draw robot marker with reachable circle."""
    circle = Circle(robot.position, robot.remaining_budget, fill=False,
                   edgecolor=color, linewidth=CIRCLE_LINE, linestyle='--',
                   alpha=CIRCLE_ALPHA)
    ax.add_patch(circle)
    scatter = ax.scatter(*robot.position, c=color, s=ROBOT_SIZE, marker='o',
                        edgecolors='black', linewidths=ROBOT_EDGE, zorder=15,
                        label=label)
    ax.annotate(f'R{robot.id}', xy=robot.position, xytext=(0, 0),
               textcoords='offset points', fontsize=ROBOT_FONT, fontweight='bold',
               ha='center', va='center', color='white', zorder=16)
    return scatter


def add_quadtree_cells(ax, quadtree, linewidth=1.2, alpha=0.7):
    """Overlay quadtree cell boundaries."""
    for leaf in quadtree.get_leaf_nodes():
        x_min, x_max, y_min, y_max = leaf.bounds
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        fill=False, edgecolor='gray' if alpha < 0.7 else 'black',
                        linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)


def visualize_results(env, gp, generator, candidate_sets, robots, init_points, bounds):
    """Create visualizations of candidate generation results."""
    setup_plot_style()
    
    # Compute grid predictions
    res = 100
    x, y = np.linspace(*bounds[0], res), np.linspace(*bounds[1], res)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]
    mean, std = gp.predict(points, return_std=True)
    mean, variance = mean.reshape(X.shape), (std ** 2).reshape(X.shape)
    X_true, Y_true, true_values = env.evaluate_grid(resolution=res)
    
    fig = plt.figure(figsize=(22, 14))
    
    # Row 1: 3D plots
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    surf1 = plot_3d_surface(ax1, X_true, Y_true, true_values, init_points,
                           env.evaluate(init_points), 'viridis',
                           'Ground Truth Environment', bounds)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.05).ax.tick_params(labelsize=9)
    
    ax2 = fig.add_subplot(3, 4, 2, projection='3d')
    surf2 = plot_3d_surface(ax2, X, Y, mean, init_points,
                           gp.predict(init_points)[0], 'viridis',
                           'GP Mean (Belief)', bounds)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.05).ax.tick_params(labelsize=9)
    
    ax3 = fig.add_subplot(3, 4, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, variance, cmap='YlOrRd', linewidth=0,
                            alpha=0.95, vmin=0, vmax=variance.max())
    ax3.set(xlabel='X', ylabel='Y', zlabel='Variance', title='GP Uncertainty',
           xlim=bounds[0], ylim=bounds[1])
    ax3.view_init(25, 220)
    ax3.grid(alpha=0.3)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.05).ax.tick_params(labelsize=9)
    
    # Quadtree visualization
    ax4 = fig.add_subplot(3, 4, 4)
    generator.quadtree.visualize(ax=ax4, show_variance=False, linewidth=1.2)
    for cand_set in candidate_sets.values():
        ax4.scatter(cand_set.points[::3, 0], cand_set.points[::3, 1],
                   s=15, c='red', marker='.', alpha=0.6)
    ax4.set(title='Quadtree Adaptive Segmentation', xlabel='X', ylabel='Y')
    
    # Row 2: 2D heatmaps
    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.contourf(X, Y, mean, levels=25, cmap='viridis')
    ax5.scatter(init_points[:, 0], init_points[:, 1], c='red', s=50,
               marker='o', edgecolors='black', linewidths=1.2,
               label='Initial Observations', zorder=10)
    ax5.set(xlim=bounds[0], ylim=bounds[1], title='GP Mean (2D)',
           xlabel='X', ylabel='Y', aspect='equal')
    plt.colorbar(im5, ax=ax5).set_label('Mean')
    ax5.legend(loc='upper right', fontsize=8)
    
    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.contourf(X, Y, variance, levels=25, cmap='YlOrRd', alpha=0.8)
    add_quadtree_cells(ax6, generator.quadtree)
    ax6.set(xlim=bounds[0], ylim=bounds[1], title='GP Variance + Quadtree',
           xlabel='X', ylabel='Y', aspect='equal')
    plt.colorbar(im6, ax=ax6).set_label('Variance')
    
    # All robots view
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.contourf(X, Y, variance, levels=20, cmap='YlOrRd', alpha=0.35)
    handles = [draw_robot(ax7, r, COLORS[i], f'Robot {r.id}')
              for i, r in enumerate(robots)]
    for robot_id, cand_set in candidate_sets.items():
        feasible = cand_set.get_feasible_points()
        if len(feasible) > 0:
            ax7.scatter(feasible[:, 0], feasible[:, 1], c=COLORS[robot_id],
                       s=35, marker='x', alpha=0.75, linewidths=1.5, zorder=5)
    ax7.set(xlim=bounds[0], ylim=bounds[1], title='All Robots + Reachability',
           xlabel='X', ylabel='Y', aspect='equal')
    ax7.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9)
    
    # Quadtree with variance
    ax8 = fig.add_subplot(3, 4, 8)
    generator.quadtree.visualize(ax=ax8, show_variance=True, linewidth=1.0)
    ax8.scatter(init_points[:, 0], init_points[:, 1], c='black', s=50,
               marker='o', edgecolors='white', linewidths=1.5,
               label='Observations', zorder=10)
    ax8.set_title('Quadtree (Variance Colored)', fontweight='bold')
    ax8.legend()
    
    # Row 3: Individual robots
    for idx, (robot_id, robot) in enumerate(zip(candidate_sets.keys(), robots)):
        ax = fig.add_subplot(3, 4, 9 + idx)
        cand_set = candidate_sets[robot_id]
        
        ax.contourf(X, Y, variance, levels=20, cmap='YlOrRd', alpha=0.3)
        add_quadtree_cells(ax, generator.quadtree, 0.6, 0.5)
        draw_robot(ax, robot, COLORS[idx])
        
        # Add range circle for legend
        range_circle = Circle(robot.position, robot.remaining_budget, fill=False,
              edgecolor=COLORS[idx], linewidth=CIRCLE_LINE,
              linestyle='--', alpha=CIRCLE_ALPHA,
              label=f'Range: {robot.remaining_budget:.1f}')
        ax.add_patch(range_circle)
        
        feasible, infeasible = cand_set.get_feasible_points(), cand_set.points[~cand_set.feasible]
        if len(infeasible) > 0:
            ax.scatter(infeasible[:, 0], infeasible[:, 1], c='dimgray', s=28,
                      marker='x', alpha=0.5, label=f'Infeasible ({len(infeasible)})',
                      zorder=3, linewidths=1)
        if len(feasible) > 0:
            ax.scatter(feasible[:, 0], feasible[:, 1], c='limegreen', s=75,
                      marker='X', alpha=0.95, edgecolors='darkgreen',
                      linewidths=1.2, label=f'Feasible ({len(feasible)})', zorder=10)
        
        ax.set(xlim=bounds[0], ylim=bounds[1], title=f'Robot {robot_id} Candidates',
              xlabel='X', ylabel='Y', aspect='equal')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Statistics panel
    if len(robots) < 4:
        ax_empty = fig.add_subplot(3, 4, 12)
        stats = (f'Statistics:\n\nQuadtree cells: {generator.quadtree.n_leaves}\n'
                f'Max depth: {generator.quadtree.max_depth}\n'
                f'Total candidates: {sum(len(cs.points) for cs in candidate_sets.values())}\n'
                f'Total feasible: {sum(len(cs.get_feasible_points()) for cs in candidate_sets.values())}')
        ax_empty.text(0.5, 0.5, stats, ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_empty.axis('off')
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/candidate_generation_professional.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("   Saved to: results/candidate_generation_professional.png")
    plt.show()


def demo_candidate_generation(env_name='townsend', bounds=None, n_init=35, seed=42,
                             robot_configs=None, quadtree_config=None, sampling_config=None):
    """
    Run candidate generation demo.
    
    Args:
        env_name: Environment function name
        bounds: Spatial bounds
        n_init: Number of initial observations
        seed: Random seed
        robot_configs: List of (position, budget) tuples
        quadtree_config: Quadtree configuration
        sampling_config: Sampling configuration
    """
    print("=" * 70)
    print("STEP A: CANDIDATE GENERATION DEMO")
    print("=" * 70)
    
    bounds = bounds if bounds is not None else np.array([[0, 100], [0, 100]])
    
    # Create environment and collect initial data
    print(f"\n1. Creating environment ({env_name})...")
    env = create_environment(bounds, env_type='synthetic', function_name=env_name)
    
    print("2. Initializing GP belief...")
    np.random.seed(seed)
    init_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, 2))
    init_values = env.observe(init_points)
    
    # Use existing create_gp_belief with Matern kernel
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0,
                         variance=1.0, noise=0.1)
    gp.update(init_points, init_values)
    print(f"   Trained with {n_init} observations")
    
    # Create robots
    print("\n3. Creating robots...")
    if robot_configs is None:
        robot_configs = [([15, 20], 30), ([75, 75], 35), ([45, 50], 28)]
    
    robots = [Robot(i, np.array(pos), BudgetType.DISTANCE, budget)
             for i, (pos, budget) in enumerate(robot_configs)]
    for r in robots:
        print(f"   Robot {r.id}: pos={r.position}, budget={r.remaining_budget:.1f}")
    
    # Setup candidate generation
    if quadtree_config is None:
        var_thresh = np.percentile(gp.get_variance(
            np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 2))), 50)
        quadtree_config = {'max_depth': 5, 'min_cell_size': 6.0,
                          'variance_threshold': var_thresh}
    
    if sampling_config is None:
        sampling_config = {'method': 'grid', 'points_per_cell': 4, 'min_spacing': 7.0}
    
    print("\n4. Generating candidates...")
    generator = CandidateGenerator(bounds, quadtree_config, sampling_config)
    candidate_sets = generator.generate_candidates(gp, robots, budget_reserve=2.0)
    
    # Print statistics
    print("\n5. Statistics:")
    print(f"   Quadtree: {generator.quadtree.n_leaves} cells, depth {generator.quadtree.max_depth}")
    for robot_id, cand_set in candidate_sets.items():
        feasible = cand_set.get_feasible_points()
        print(f"   Robot {robot_id}: {len(feasible)}/{len(cand_set.points)} feasible")
    
    print("\n6. Creating visualizations...")
    visualize_results(env, gp, generator, candidate_sets, robots, init_points, bounds)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    demo_candidate_generation()
