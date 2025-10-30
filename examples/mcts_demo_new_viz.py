"""
Cleaner visualization functions for MCTS demo.
Copy these to replace the old ones in mcts_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def visualize_mcts_results(
    env, gp_before, gp_after, robot_start, candidate_set,
    planned_path, samples, bounds, tree_root, mcts_stats
):
    """
    Clean visualization of MCTS planning results (similar to assignment demo style).
    
    Layout (2 rows x 3 columns):
    Row 1: Ground Truth | GP Before | GP After
    Row 2: Candidates & Path | Uncertainty & Path | MCTS Statistics
    """
    # Setup plot style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })
    
    # Grid for predictions
    resolution = 100
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Evaluate
    true_values = env.evaluate(grid_points).reshape(X.shape)
    mean_before, std_before = gp_before.predict(grid_points, return_std=True)
    mean_before = mean_before.reshape(X.shape)
    var_before = (std_before ** 2).reshape(X.shape)
    
    mean_after, std_after = gp_after.predict(grid_points, return_std=True)
    mean_after = mean_after.reshape(X.shape)
    var_after = (std_after ** 2).reshape(X.shape)
    
    # Create figure
    fig = plt.figure(figsize=(20, 13))
    
    # === ROW 1: 3D Surfaces ===
    
    # 1. Ground Truth
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, true_values, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    # Add sample points
    if len(samples) > 0:
        sample_pos = np.array([s[0] for s in samples])
        sample_vals = env.evaluate(sample_pos)
        ax1.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_vals,
                   c='red', s=50, marker='o', edgecolors='darkred',
                   linewidths=1.5, zorder=10)
    
    ax1.set_xlabel('X', labelpad=8)
    ax1.set_ylabel('Y', labelpad=8)
    ax1.set_zlabel('Value', labelpad=8)
    ax1.set_title('Ground Truth + Samples', fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=220)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.05)
    
    # 2. GP Before
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, mean_before, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    ax2.set_xlabel('X', labelpad=8)
    ax2.set_ylabel('Y', labelpad=8)
    ax2.set_zlabel('Value', labelpad=8)
    ax2.set_title('GP Mean (Before Planning)', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=220)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.05)
    
    # 3. GP After
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, mean_after, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.95)
    
    if len(samples) > 0:
        sample_pos = np.array([s[0] for s in samples])
        gp_sample_mean, _ = gp_after.predict(sample_pos)
        ax3.scatter(sample_pos[:, 0], sample_pos[:, 1], gp_sample_mean,
                   c='red', s=50, marker='o', edgecolors='darkred',
                   linewidths=1.5, zorder=10)
    
    ax3.set_xlabel('X', labelpad=8)
    ax3.set_ylabel('Y', labelpad=8)
    ax3.set_zlabel('Value', labelpad=8)
    ax3.set_title('GP Mean (After Execution)', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=220)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.05)
    
    # === ROW 2: 2D Plots ===
    
    # 4. Candidates and Planned Path
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.contourf(X, Y, var_before, levels=20, cmap='YlOrRd', alpha=0.4)
    
    # Candidates
    feasible = candidate_set.get_feasible_points()
    if len(feasible) > 0:
        ax4.scatter(feasible[:, 0], feasible[:, 1], c='lightgray',
                   s=30, alpha=0.5, label=f'Candidates ({len(feasible)})')
    
    # Robot start
    ax4.scatter(robot_start[0], robot_start[1], s=200, c='blue',
               marker='o', edgecolors='black', linewidths=2,
               label='Start', zorder=10)
    
    # Planned path
    if len(planned_path) > 0:
        path_positions = [robot_start] + planned_path
        path_array = np.array(path_positions)
        ax4.plot(path_array[:, 0], path_array[:, 1], 'b-',
                linewidth=3, alpha=0.7, label='MCTS Path', zorder=5)
        
        for i, pos in enumerate(planned_path):
            ax4.scatter(pos[0], pos[1], s=150, c='darkblue', marker='X',
                       edgecolors='white', linewidths=2, zorder=8)
            ax4.annotate(f'{i+1}', pos, fontsize=9, color='white',
                        weight='bold', ha='center', va='center', zorder=9)
    
    ax4.set_xlim(bounds[0, 0], bounds[0, 1])
    ax4.set_ylim(bounds[1, 0], bounds[1, 1])
    ax4.set_title(f'Planned Path ({len(planned_path)} waypoints)', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Uncertainty and Path
    ax5 = fig.add_subplot(2, 3, 5)
    cs5 = ax5.contourf(X, Y, var_before, levels=20, cmap='hot', alpha=0.6)
    
    # Start and path
    ax5.scatter(robot_start[0], robot_start[1], s=200, c='white',
               marker='o', edgecolors='black', linewidths=2, label='Start', zorder=10)
    
    if len(planned_path) > 0:
        path_positions = [robot_start] + planned_path
        path_array = np.array(path_positions)
        ax5.plot(path_array[:, 0], path_array[:, 1], 'k-',
                linewidth=3, alpha=0.8, zorder=5)
        ax5.scatter(path_array[1:, 0], path_array[1:, 1], s=150,
                   c='white', marker='X', edgecolors='black',
                   linewidths=2, zorder=8, label='Targets')
    
    ax5.set_xlim(bounds[0, 0], bounds[0, 1])
    ax5.set_ylim(bounds[1, 0], bounds[1, 1])
    ax5.set_title('Path on Uncertainty Map', fontweight='bold')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_aspect('equal')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(cs5, ax=ax5, label='Variance')
    
    # 6. MCTS Tree Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Collect tree stats
    tree_stats = collect_tree_statistics(tree_root)
    
    # Display statistics
    stats_text = "MCTS PLANNING STATISTICS\n" + "="*40 + "\n\n"
    stats_text += f"Iterations:      {mcts_stats['iterations']}\n"
    stats_text += f"Tree Size:       {tree_stats['total_nodes']} nodes\n"
    stats_text += f"Max Depth:       {tree_stats['max_depth']}\n"
    stats_text += f"Planning Time:   {mcts_stats['planning_time']:.2f}s\n\n"
    
    stats_text += "ROOT NODE\n" + "-"*40 + "\n"
    stats_text += f"Visits:          {tree_root.visits}\n"
    stats_text += f"Avg Reward:      {tree_root.total_reward/tree_root.visits if tree_root.visits > 0 else 0:.4f}\n"
    stats_text += f"Children:        {len(tree_root.children)}\n\n"
    
    stats_text += "DEPTH DISTRIBUTION\n" + "-"*40 + "\n"
    for depth in sorted(tree_stats['nodes_per_depth'].keys()):
        n_nodes = tree_stats['nodes_per_depth'][depth]
        avg_visits = np.mean(tree_stats['visits_per_depth'][depth])
        avg_reward = np.mean(tree_stats['rewards_per_depth'][depth]) if tree_stats['rewards_per_depth'][depth] else 0
        stats_text += f"Depth {depth}:  {n_nodes:4d} nodes  "
        stats_text += f"Visits: {avg_visits:6.1f}  "
        stats_text += f"Reward: {avg_reward:7.4f}\n"
    
    if len(planned_path) > 0:
        stats_text += "\n" + "PATH DETAILS\n" + "-"*40 + "\n"
        total_dist = np.linalg.norm(planned_path[0] - robot_start)
        for i in range(1, len(planned_path)):
            total_dist += np.linalg.norm(planned_path[i] - planned_path[i-1])
        
        stats_text += f"Waypoints:       {len(planned_path)}\n"
        stats_text += f"Total Distance:  {total_dist:.2f} m\n"
    
    if len(samples) > 0:
        stats_text += "\n" + "EXECUTION RESULTS\n" + "-"*40 + "\n"
        sample_values = [s[1] for s in samples]
        stats_text += f"Samples:         {len(samples)}\n"
        stats_text += f"Min Value:       {min(sample_values):.4f}\n"
        stats_text += f"Max Value:       {max(sample_values):.4f}\n"
        stats_text += f"Mean Value:      {np.mean(sample_values):.4f}\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    return fig


def collect_tree_statistics(root):
    """Collect depth-wise statistics from MCTS tree."""
    stats = {
        'total_nodes': 0,
        'max_depth': 0,
        'nodes_per_depth': {},
        'visits_per_depth': {},
        'rewards_per_depth': {}
    }
    
    def traverse(node, depth=0):
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        if depth not in stats['nodes_per_depth']:
            stats['nodes_per_depth'][depth] = 0
            stats['visits_per_depth'][depth] = []
            stats['rewards_per_depth'][depth] = []
        
        stats['nodes_per_depth'][depth] += 1
        stats['visits_per_depth'][depth].append(node.visits)
        if node.visits > 0:
            stats['rewards_per_depth'][depth].append(node.total_reward / node.visits)
        
        for child in node.children:
            traverse(child, depth + 1)
    
    traverse(root)
    return stats
