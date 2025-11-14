"""
Demo: MCTS Planning (Step C)

This demo demonstrates Step C of the IPP algorithm in detail:
- Using MCTS to plan optimal sampling sequences within a robot's candidate window
- Visualizing the MCTS tree structure, node visits, and UCB scores
- Showing different rollout paths explored
- Balancing information gain with travel efficiency
- Respecting budget constraints

The demo shows:
1. Environment setup and initial GP belief
2. Candidate generation (Step A)
3. MCTS tree building process with detailed visualization
4. Node exploration statistics
5. Best path extraction and execution

UNITS:
  - All distances in METERS
  - All times in SECONDS
  - Speeds in METERS/SECOND
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from src.core.environment import create_environment
from src.core.robot import Robot, BudgetType
from src.core.belief import create_gp_belief
from src.planning import CandidateGenerator, MCTSPlanner, MCTSConfig, MCTSNode
from config.units import print_environment_info


def collect_tree_statistics(root: MCTSNode):
    """
    Collect depth-wise statistics about the MCTS tree.
    """
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


# Stub for backward compatibility - tree visualization removed (too complex)
def visualize_mcts_tree(root: MCTSNode, title="MCTS Tree"):
    """
    Simple MCTS tree summary (deprecated - use visualize_mcts_results instead).
    """
    stats = collect_tree_statistics(root)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')
    
    # Just show text statistics
    stats_text = f"{title}\n" + "="*50 + "\n\n"
    stats_text += f"Total Nodes:     {stats['total_nodes']}\n"
    stats_text += f"Max Depth:       {stats['max_depth']}\n"
    stats_text += f"Root Visits:     {root.visits}\n"
    stats_text += f"Root Reward:     {root.total_reward/root.visits if root.visits > 0 else 0:.4f}\n"
    stats_text += f"Root Children:   {len(root.children)}\n\n"
    
    stats_text += "Depth Distribution:\n" + "-"*50 + "\n"
    for depth in sorted(stats['nodes_per_depth'].keys()):
        n_nodes = stats['nodes_per_depth'][depth]
        avg_visits = np.mean(stats['visits_per_depth'][depth])
        avg_reward = np.mean(stats['rewards_per_depth'][depth]) if stats['rewards_per_depth'][depth] else 0
        stats_text += f"  Depth {depth}: {n_nodes:4d} nodes, "
        stats_text += f"{avg_visits:6.1f} avg visits, "
        stats_text += f"{avg_reward:7.4f} avg reward\n"
    
    stats_text += "\n" + "Note: Full visualization available in planning results.\n"
    stats_text += "Tree structure visualization removed due to complexity.\n"
    stats_text += "Use visualize_mcts_results() for comprehensive plots."
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    return fig


def _old_visualize_mcts_tree_REMOVED(root: MCTSNode, title="MCTS Tree"):
    """OLD COMPLEX TREE VISUALIZATION - REMOVED"""
    stats = collect_tree_statistics(root)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # === LEFT: Tree Structure ===
    ax1.set_title(f'{title}\nTree Structure (Visits & Rewards)', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Build position layout (simple hierarchical)
    positions = {}
    level_counts = {}
    level_positions = {}
    
    def assign_positions(node, depth=0, x_offset=0, x_spacing=1.0):
        """Assign x,y positions for tree layout."""
        node_id = id(node)
        
        if depth not in level_counts:
            level_counts[depth] = 0
            level_positions[depth] = []
        
        # Y position based on depth
        y = -depth * 2
        
        # X position
        if len(node.children) == 0:
            # Leaf node
            x = x_offset + level_counts[depth] * x_spacing
            level_counts[depth] += 1
        else:
            # Internal node - center between children
            child_positions = []
            for i, child in enumerate(node.children):
                child_x = assign_positions(child, depth + 1, x_offset, x_spacing)
                child_positions.append(child_x)
            x = np.mean(child_positions) if child_positions else x_offset
        
        positions[node_id] = (x, y)
        level_positions[depth].append(x)
        return x
    
    assign_positions(root)
    
    # Draw edges first
    for edge in stats['edges']:
        parent_pos = positions[edge['parent']]
        child_pos = positions[edge['child']]
        
        # Edge thickness based on child visits
        child_visits = edge['child_visits']
        max_visits = max([n['visits'] for n in stats['nodes']])
        thickness = 0.5 + 3.5 * (child_visits / max_visits) if max_visits > 0 else 1.0
        
        ax1.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]],
                'gray', linewidth=thickness, alpha=0.6, zorder=1)
    
    # Draw nodes
    max_visits = max([n['visits'] for n in stats['nodes']])
    max_reward = max([abs(n['avg_reward']) for n in stats['nodes']])
    
    for node_info in stats['nodes']:
        node_id = node_info['id']
        x, y = positions[node_id]
        
        # Size based on visits
        size = 200 + 800 * (node_info['visits'] / max_visits) if max_visits > 0 else 200
        
        # Color based on average reward
        if max_reward > 0:
            color_val = node_info['avg_reward'] / max_reward
        else:
            color_val = 0
        color = plt.cm.RdYlGn((color_val + 1) / 2)  # Map [-1, 1] to [0, 1]
        
        # Draw node
        ax1.scatter(x, y, s=size, c=[color], edgecolors='black', 
                   linewidths=2, zorder=2, alpha=0.9)
        
        # Label with visits and reward
        label = f'{node_info["visits"]}'
        if node_info['avg_reward'] != 0:
            label += f'\n{node_info["avg_reward"]:.2f}'
        
        ax1.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=3)
    
    ax1.set_xlim(min([p[0] for p in positions.values()]) - 1,
                 max([p[0] for p in positions.values()]) + 1)
    ax1.set_ylim(min([p[1] for p in positions.values()]) - 1, 1)
    
    # === RIGHT: Statistics ===
    ax2.axis('off')
    ax2.set_title('MCTS Statistics', fontweight='bold', fontsize=12)
    
    # Collect stats by depth
    depth_stats = {}
    for node_info in stats['nodes']:
        depth = node_info['depth']
        if depth not in depth_stats:
            depth_stats[depth] = {
                'count': 0,
                'total_visits': 0,
                'total_reward': 0,
                'avg_reward': []
            }
        depth_stats[depth]['count'] += 1
        depth_stats[depth]['total_visits'] += node_info['visits']
        if node_info['visits'] > 0:
            depth_stats[depth]['avg_reward'].append(node_info['avg_reward'])
    
    # Display text statistics
    text_y = 0.95
    line_height = 0.05
    
    def add_text(text, bold=False):
        nonlocal text_y
        weight = 'bold' if bold else 'normal'
        ax2.text(0.1, text_y, text, transform=ax2.transAxes,
                fontsize=10, fontweight=weight, verticalalignment='top',
                family='monospace')
        text_y -= line_height
    
    add_text('TREE SUMMARY', bold=True)
    add_text(f'Total nodes:        {stats["total_nodes"]}')
    add_text(f'Maximum depth:      {stats["max_depth"]}')
    add_text(f'Root visits:        {root.visits}')
    add_text(f'Root avg reward:    {root.total_reward/root.visits if root.visits > 0 else 0:.4f}')
    add_text('')
    
    add_text('DEPTH STATISTICS', bold=True)
    for depth in sorted(depth_stats.keys()):
        ds = depth_stats[depth]
        avg_visits = ds['total_visits'] / ds['count']
        avg_reward = np.mean(ds['avg_reward']) if ds['avg_reward'] else 0
        add_text(f'Depth {depth}:')
        add_text(f'  Nodes: {ds["count"]:3d}  Visits: {avg_visits:6.1f}  Reward: {avg_reward:7.4f}')
    
    add_text('')
    add_text('TOP 5 NODES BY VISITS', bold=True)
    sorted_nodes = sorted(stats['nodes'], key=lambda n: n['visits'], reverse=True)[:5]
    for i, node in enumerate(sorted_nodes):
        add_text(f'{i+1}. Depth {node["depth"]}: {node["visits"]} visits, reward {node["avg_reward"]:.4f}')
    
    plt.tight_layout()
    return fig


def visualize_mcts_planning_results(
    env, gp_before, gp_after, robot_start, candidate_set, 
    planned_path, samples, bounds, tree_root, mcts_stats
):
    """Alias for backward compatibility - calls the new visualization function."""
    return visualize_mcts_results(
        env, gp_before, gp_after, robot_start, candidate_set,
        planned_path, samples, bounds, tree_root, mcts_stats
    )


def _old_visualize_mcts_planning_results(
    env, gp_before, gp_after, robot_start, candidate_set, 
    planned_path, samples, bounds, tree_root, mcts_stats
):
    """OLD - KEEP FOR REFERENCE ONLY"""
    # Grid for predictions
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Ground truth
    true_values = env.evaluate(grid_points).reshape(X.shape)
    
    # GP predictions
    mean_before, std_before = gp_before.predict(grid_points, return_std=True)
    mean_before = mean_before.reshape(X.shape)
    var_before = (std_before ** 2).reshape(X.shape)
    
    mean_after, std_after = gp_after.predict(grid_points, return_std=True)
    mean_after = mean_after.reshape(X.shape)
    var_after = (std_after ** 2).reshape(X.shape)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # === ROW 1: Environment, Candidates, and Planned Path ===
    
    # 1.1: True environment
    ax1 = fig.add_subplot(3, 4, 1)
    cs1 = ax1.contourf(X, Y, true_values, levels=20, cmap='viridis')
    ax1.plot(robot_start[0], robot_start[1], 'r*', markersize=20, label='Start', zorder=10)
    ax1.set_title('Ground Truth Environment', fontsize=11, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    plt.colorbar(cs1, ax=ax1)
    
    # 1.2: Candidate points
    ax2 = fig.add_subplot(3, 4, 2)
    cs2 = ax2.contourf(X, Y, var_before, levels=20, cmap='hot', alpha=0.3)
    feasible = candidate_set.get_feasible_points()
    if len(feasible) > 0:
        ax2.scatter(feasible[:, 0], feasible[:, 1], c='blue', s=30, alpha=0.6,
                   label=f'Candidates ({len(feasible)})')
    ax2.plot(robot_start[0], robot_start[1], 'r*', markersize=20, label='Start', zorder=10)
    ax2.set_title('Candidate Points (Step A)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    plt.colorbar(cs2, ax=ax2, label='Variance')
    
    # 1.3: Planned path on environment
    ax3 = fig.add_subplot(3, 4, 3)
    cs3 = ax3.contourf(X, Y, true_values, levels=20, cmap='viridis', alpha=0.4)
    if len(feasible) > 0:
        ax3.scatter(feasible[:, 0], feasible[:, 1], c='lightgray', s=20, alpha=0.3)
    ax3.plot(robot_start[0], robot_start[1], 'r*', markersize=20, label='Start', zorder=10)
    
    if len(planned_path) > 0:
        path_positions = [robot_start] + planned_path
        path_array = np.array(path_positions)
        ax3.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=3, alpha=0.7, label='MCTS Path')
        
        for i, pos in enumerate(planned_path):
            ax3.scatter(pos[0], pos[1], c='red', s=150, marker='X',
                       edgecolors='darkred', linewidths=2, zorder=5)
            ax3.annotate(f'{i+1}', pos, fontsize=9, color='white', weight='bold',
                        ha='center', va='center', zorder=6)
    
    ax3.set_title(f'MCTS Planned Path ({len(planned_path)} waypoints)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    plt.colorbar(cs3, ax=ax3)
    
    # 1.4: Path on uncertainty map
    ax4 = fig.add_subplot(3, 4, 4)
    cs4 = ax4.contourf(X, Y, var_before, levels=20, cmap='hot', alpha=0.6)
    ax4.plot(robot_start[0], robot_start[1], 'k*', markersize=20, label='Start', zorder=10)
    
    if len(planned_path) > 0:
        path_positions = [robot_start] + planned_path
        path_array = np.array(path_positions)
        ax4.plot(path_array[:, 0], path_array[:, 1], 'k-', linewidth=3, alpha=0.8)
        ax4.scatter(path_array[1:, 0], path_array[1:, 1], c='white', s=150,
                   marker='X', edgecolors='black', linewidths=2, zorder=5,
                   label='Targets')
    
    ax4.set_title('Path on Uncertainty Map', fontsize=11, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.legend()
    plt.colorbar(cs4, ax=ax4, label='Variance')
    
    # === ROW 2: GP Before and After ===
    
    # 2.1: GP mean before
    ax5 = fig.add_subplot(3, 4, 5)
    cs5 = ax5.contourf(X, Y, mean_before, levels=20, cmap='viridis')
    ax5.plot(robot_start[0], robot_start[1], 'r*', markersize=15)
    ax5.set_title('GP Mean (Before)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(cs5, ax=ax5)
    
    # 2.2: GP variance before
    ax6 = fig.add_subplot(3, 4, 6)
    cs6 = ax6.contourf(X, Y, var_before, levels=20, cmap='hot')
    ax6.plot(robot_start[0], robot_start[1], 'r*', markersize=15)
    ax6.set_title('GP Variance (Before)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(cs6, ax=ax6)
    
    # 2.3: GP mean after
    ax7 = fig.add_subplot(3, 4, 7)
    cs7 = ax7.contourf(X, Y, mean_after, levels=20, cmap='viridis')
    if len(samples) > 0:
        sample_pos = np.array([s[0] for s in samples])
        ax7.scatter(sample_pos[:, 0], sample_pos[:, 1], c='red', s=100,
                   marker='X', edgecolors='darkred', linewidths=2, label='Samples', zorder=5)
    ax7.plot(robot_start[0], robot_start[1], 'k*', markersize=15, zorder=10)
    ax7.set_title('GP Mean (After Execution)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.legend()
    plt.colorbar(cs7, ax=ax7)
    
    # 2.4: GP variance after
    ax8 = fig.add_subplot(3, 4, 8)
    cs8 = ax8.contourf(X, Y, var_after, levels=20, cmap='hot')
    if len(samples) > 0:
        sample_pos = np.array([s[0] for s in samples])
        ax8.scatter(sample_pos[:, 0], sample_pos[:, 1], c='blue', s=100,
                   marker='X', edgecolors='darkblue', linewidths=2, label='Samples')
    ax8.set_title('GP Variance (After Execution)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    ax8.legend()
    plt.colorbar(cs8, ax=ax8)
    
    # === ROW 3: MCTS Statistics ===
    
    # 3.1: Variance reduction
    var_reduction = var_before - var_after
    ax9 = fig.add_subplot(3, 4, 9)
    cs9 = ax9.contourf(X, Y, var_reduction, levels=20, cmap='RdYlGn')
    if len(samples) > 0:
        sample_pos = np.array([s[0] for s in samples])
        ax9.scatter(sample_pos[:, 0], sample_pos[:, 1], c='black', s=100,
                   marker='X', edgecolors='white', linewidths=2, label='Samples')
    ax9.set_title('Information Gain', fontsize=11, fontweight='bold')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    ax9.legend()
    plt.colorbar(cs9, ax=ax9)
    
    # 3.2-3.4: MCTS Statistics (text)
    for idx, ax_idx in enumerate([10, 11, 12]):
        ax = fig.add_subplot(3, 4, ax_idx)
        ax.axis('off')
        
        if idx == 0:
            # Planning stats
            ax.text(0.5, 0.95, 'MCTS PLANNING STATS', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')
            
            stats_text = f"""
Iterations:     {mcts_stats['iterations']}
Tree Size:      {mcts_stats['tree_size']} nodes
Max Depth:      {mcts_stats['max_depth_reached']}
Planning Time:  {mcts_stats['planning_time']:.2f}s

Root Visits:    {tree_root.visits}
Root Reward:    {tree_root.total_reward/tree_root.visits if tree_root.visits > 0 else 0:.4f}
Children:       {len(tree_root.children)}
"""
            ax.text(0.1, 0.80, stats_text, transform=ax.transAxes,
                   fontsize=9, family='monospace', va='top')
        
        elif idx == 1 and len(planned_path) > 0:
            # Path stats
            ax.text(0.5, 0.95, 'PATH STATISTICS', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')
            
            total_dist = np.linalg.norm(planned_path[0] - robot_start)
            for i in range(1, len(planned_path)):
                total_dist += np.linalg.norm(planned_path[i] - planned_path[i-1])
            
            path_text = f"""
Waypoints:      {len(planned_path)}
Total Distance: {total_dist:.2f} m

Waypoint Details:
"""
            for i, wp in enumerate(planned_path):
                if i < 5:  # Show first 5
                    path_text += f"  {i+1}. ({wp[0]:.1f}, {wp[1]:.1f})\n"
            if len(planned_path) > 5:
                path_text += f"  ... ({len(planned_path)-5} more)\n"
            
            ax.text(0.1, 0.80, path_text, transform=ax.transAxes,
                   fontsize=9, family='monospace', va='top')
        
        elif idx == 2 and len(samples) > 0:
            # Execution stats
            ax.text(0.5, 0.95, 'EXECUTION RESULTS', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')
            
            sample_values = [s[1] for s in samples]
            exec_text = f"""
Samples:        {len(samples)}
Min Value:      {min(sample_values):.4f}
Max Value:      {max(sample_values):.4f}
Mean Value:     {np.mean(sample_values):.4f}

Sample Details:
"""
            for i, (pos, val, time) in enumerate(samples):
                if i < 5:  # Show first 5
                    exec_text += f"  {i+1}. Val={val:.3f} @t={time:.1f}s\n"
            if len(samples) > 5:
                exec_text += f"  ... ({len(samples)-5} more)\n"
            
            ax.text(0.1, 0.80, exec_text, transform=ax.transAxes,
                   fontsize=9, family='monospace', va='top')
    
    plt.tight_layout()
    return fig


def demo_mcts_planning():
    """Demonstrate MCTS planning for a single robot."""
    
    print(f"{'='*70}")
    print(f"MCTS PLANNING DEMO (STEP C)")
    print(f"{'='*70}\n")
    
    # Configuration - medium search area: 1km × 1km
    bounds = np.array([[0, 100], [0, 100]])  # Coordinate bounds
    physical_scale = 10.0  # Each coordinate unit = 10 meters  
    robot_speed_ms = 5.0  # 5 m/s = 18 km/h
    env_name = 'townsend'
    n_init = 1  # Minimal initial samples
    robot_budget = 200.0  # 200 seconds mission time
    sensor_time = 1.0  # 1 second per measurement
    
    # === STEP 0: ENVIRONMENT SETUP ===
    print(f"\n{'='*70}")
    print(f"STEP 0: ENVIRONMENT SETUP")
    print(f"{'='*70}")
    
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name=env_name,
        physical_scale=physical_scale
    )
    
    # Print environment information
    print_environment_info(
        env_name=env_name,
        bounds=bounds,
        physical_scale=physical_scale,
        robot_speed=robot_speed_ms,
        robot_budget=robot_budget,
        budget_type='time'
    )
    
    # Create initial GP belief
    print(f"Generating {n_init} initial samples...")
    init_points = np.random.uniform(
        [bounds[0, 0], bounds[1, 0]],
        [bounds[0, 1], bounds[1, 1]],
        size=(n_init, 2)
    )
    init_values = env.evaluate(init_points)
    gp = create_gp_belief(bounds, kernel_type='matern', length_scale=15.0,
                          variance=1.0, noise=0.1)
    gp.update(init_points, init_values)
    print(f"Initial GP trained with {n_init} samples")
    
    # Create single robot with environment link
    robot_start = np.array([50.0, 50.0])  # Center of coordinate system
    robot = Robot(
        robot_id=0,
        initial_position=robot_start,
        budget_type=BudgetType.TIME,
        initial_budget=robot_budget,  # seconds
        max_speed=robot_speed_ms,     # m/s
        environment=env               # Link for coordinate conversion
    )
    
    # Display robot info in physical units
    phys_pos = robot_start * physical_scale
    max_distance = robot_budget * robot_speed_ms
    print(f"\nRobot initialized:")
    print(f"  Position: ({phys_pos[0]:.0f}m, {phys_pos[1]:.0f}m)")
    print(f"  Budget: {robot.remaining_budget:.1f}s ({robot.remaining_budget/60:.1f} min)")
    print(f"  Speed: {robot.max_speed} m/s")
    print(f"  Max distance: {max_distance:.0f}m")
    
    # === STEP A: CANDIDATE GENERATION ===
    print(f"\n{'='*70}")
    print(f"STEP A: CANDIDATE GENERATION")
    print(f"{'='*70}")
    
    # Use EXACT SAME settings as assignment_demo
    generator = CandidateGenerator(
        bounds=bounds,
        quadtree_config={'max_depth': 8, 'min_cell_size': 2.0, 'variance_threshold': 0.01},
        sampling_config={'method': 'grid', 'points_per_cell': 4, 'min_spacing': 7.0}
    )
    
    # Generate candidates for single robot (pass as list like assignment demo)
    candidate_sets = generator.generate_candidates(gp, [robot])
    candidate_set = candidate_sets[0]  # Get candidates for robot 0
    feasible = candidate_set.get_feasible_points()
    
    print(f"\nCandidate generation complete:")
    print(f"  Quadtree cells: {generator.quadtree.n_leaves}")
    print(f"  Max depth: {generator.quadtree.max_depth}")
    print(f"  Total candidates: {len(candidate_set.points)}")
    print(f"  Feasible candidates: {len(feasible)}")
    
    # DEBUG: Print some candidate details
    if len(feasible) > 0:
        print(f"\n  Sample candidates (first 10):")
        for i, cand in enumerate(feasible[:10]):
            dist = np.linalg.norm(cand - robot.position)
            time_needed = dist / robot.max_speed + sensor_time
            print(f"    {i+1}. [{cand[0]:6.2f}, {cand[1]:6.2f}] - dist={dist:6.2f}m, time={time_needed:6.2f}s")
    else:
        print(f"  ERROR: No feasible candidates! MCTS cannot plan.")
        return
    
    # === STEP C: MCTS PLANNING ===
    print(f"\n{'='*70}")
    print(f"STEP C: MCTS PLANNING")
    print(f"{'='*70}")
    
    # Configure MCTS - reasonable settings for visualization
    mcts_config = MCTSConfig(
        iterations=1000,  # Good number for thorough search
        exploration_constant=1.414,  # sqrt(2) for UCB1
        max_depth=20,  # Allow paths up to 5 waypoints
        discount_factor=0.95,  # Future reward discount
        simulation_depth=3,  # Rollout 3 steps ahead
        use_progressive_widening=True,  # Enable progressive widening
        pw_alpha=0.5,
        pw_constant=1.0,
        verbose=True
    )
    
    # Create planner
    planner = MCTSPlanner(config=mcts_config)
    
    # Plan optimal path
    print(f"\nRunning MCTS with {mcts_config.iterations} iterations...")
    planned_path = planner.plan(
        robot=robot,
        candidates=candidate_set,
        gp_belief=gp,
        sensor_time=sensor_time
    )
    
    stats = planner.get_statistics()
    
    print(f"\n{'='*70}")
    print(f"PLANNING RESULTS")
    print(f"{'='*70}")
    print(f"Planned waypoints: {len(planned_path)}")
    
    if len(planned_path) > 0:
        total_distance = np.linalg.norm(planned_path[0] - robot.position)
        for i in range(1, len(planned_path)):
            total_distance += np.linalg.norm(planned_path[i] - planned_path[i-1])
        
        total_time = total_distance / robot.max_speed + len(planned_path) * sensor_time
        
        print(f"Total distance: {total_distance:.2f} m")
        print(f"Estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Travel time: {total_distance/robot.max_speed:.1f}s")
        print(f"  Sensor time: {len(planned_path) * sensor_time:.1f}s")
        print(f"Budget usage: {total_time/robot.remaining_budget*100:.1f}%")
        
        print(f"\nPlanned waypoints:")
        for i, waypoint in enumerate(planned_path):
            print(f"  {i+1}. {waypoint}")
    
    # === STEP D: EXECUTE PLAN ===
    print(f"\n{'='*70}")
    print(f"STEP D: EXECUTE PLAN (SIMULATION)")
    print(f"{'='*70}")
    
    from copy import deepcopy
    gp_before = deepcopy(gp)
    gp_after = deepcopy(gp)
    
    samples = []
    executed_path = []
    current_time = 0.0
    
    print(f"\nExecuting plan...")
    for i, target in enumerate(planned_path):
        # Simulate travel
        distance = np.linalg.norm(target - robot.position)
        travel_time = distance / robot.max_speed
        current_time += travel_time
        
        # Take measurement
        value = env.evaluate(target.reshape(1, -1))[0]
        current_time += sensor_time
        
        # Record sample
        samples.append((target, value, current_time))
        executed_path.append(target)
        
        # Update GP
        gp_after.update(target.reshape(1, -1), np.array([value]))
        
        # Update robot position
        robot.move_to(target, timestamp=current_time, update_budget=True)
        
        print(f"  Waypoint {i+1}/{len(planned_path)}:")
        print(f"    Position: {target}")
        print(f"    Value: {value:.4f}")
        print(f"    Time: {current_time:.1f}s")
        print(f"    Budget remaining: {robot.remaining_budget:.1f}s")
    
    print(f"\nExecution complete!")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total time: {current_time:.1f}s ({current_time/60:.1f} min)")
    print(f"  Budget used: {robot.initial_budget - robot.remaining_budget:.1f}s")
    
    # === VISUALIZATION ===
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Visualize MCTS tree structure
    print("\n1. MCTS Tree Structure...")
    tree_fig = visualize_mcts_tree(planner.root, title="MCTS Search Tree")
    Path('results').mkdir(exist_ok=True)
    tree_fig.savefig('results/mcts_tree.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   Saved to: results/mcts_tree.png")
    
    # Visualize planning results
    print("\n2. Planning and Execution Results...")
    results_fig = visualize_mcts_planning_results(
        env, gp_before, gp_after, robot_start, candidate_set,
        planned_path, samples, bounds, planner.root, stats
    )
    results_fig.savefig('results/mcts_planning.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   Saved to: results/mcts_planning.png")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"DEMO COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    demo_mcts_planning()
