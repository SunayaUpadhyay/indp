"""
Plotting and visualization for IPP experiments.

Creates publication-quality figures for baseline comparisons:
- RMSE vs iterations/time curves
- Integrated variance reduction plots
- Coverage and overlap boxplots
- Trajectory visualizations with heatmaps
- Scalability plots (planning time vs N)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from typing import Dict, List, Any, Optional
import seaborn as sns

from src.core.environment import create_environment


# Visualization constants (matching demo style)
# Using distinct, colorblind-friendly colors
COLORS = {
    'Random': '#E74C3C',           # Red/Coral
    'Lawnmower': '#3498DB',        # Blue
    'SequentialGreedy': '#2ECC71', # Green
    'IndependentGreedy': '#F39C12', # Orange
    'Auction': '#9B59B6'           # Purple
}

# Robot colors for trajectories (colorblind-friendly palette)
ROBOT_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf'   # Cyan
]

ROBOT_SIZE = 100
ROBOT_EDGE = 1.5
ROBOT_FONT = 5
CIRCLE_LINE = 1.2
CIRCLE_ALPHA = 0.6


def setup_plot_style():
    """Configure matplotlib for clean visualizations (matching assignment_demo style)."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.2,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'lines.linewidth': 2
    })


def load_experiment_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from directory."""
    results = []
    results_path = Path(results_dir)
    
    for json_file in results_path.glob('*.json'):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    
    return results


def filter_results(
    results: List[Dict[str, Any]],
    scenario_type: Optional[str] = None,
    num_robots: Optional[int] = None,
    budget: Optional[float] = None,
    planner_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter results by criteria."""
    filtered = results
    
    if scenario_type:
        filtered = [r for r in filtered if r['scenario_type'] == scenario_type]
    if num_robots:
        filtered = [r for r in filtered if r['num_robots'] == num_robots]
    if budget:
        filtered = [r for r in filtered if r['initial_budget'] == budget]
    if planner_name:
        filtered = [r for r in filtered if r['planner_name'] == planner_name]
    
    return filtered


def plot_final_metrics_comparison(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Baseline Planner Comparison"
):
    """
    Create bar plots comparing final metrics across planners.
    
    Shows: RMSE, Coverage, Hotspot Recall (if SAR), Planning Time
    """
    setup_plot_style()
    
    # Group by planner
    planners = sorted(set(r['planner_name'] for r in results))
    
    metrics = {
        'RMSE': [],
        'Coverage (%)': [],
        'Hotspot Recall (%)': [],
        'Planning Time (s)': []
    }
    
    for planner in planners:
        planner_results = [r for r in results if r['planner_name'] == planner]
        
        # Average over repetitions
        metrics['RMSE'].append(np.mean([r['rmse_final'] for r in planner_results]))
        metrics['Coverage (%)'].append(np.mean([r['coverage_fraction'] * 100 for r in planner_results]))
        
        # Hotspot recall (may be None for non-SAR)
        recalls = [r.get('hotspot_recall', 0) for r in planner_results if r.get('hotspot_recall') is not None]
        metrics['Hotspot Recall (%)'].append(np.mean(recalls) * 100 if recalls else 0)
        
        metrics['Planning Time (s)'].append(np.mean([r['total_planning_time'] for r in planner_results]))
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(planners))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        bars = ax.bar(x_pos, values, color=[COLORS.get(p, 'gray') for p in planners], alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(planners, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}' if metric_name == 'RMSE' else f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_metric_boxplots(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Metric Distributions"
):
    """
    Create boxplots showing metric distributions across repetitions.
    """
    setup_plot_style()
    
    planners = sorted(set(r['planner_name'] for r in results))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    metrics_to_plot = [
        ('rmse_final', 'RMSE', axes[0]),
        ('coverage_fraction', 'Coverage Fraction', axes[1]),
        ('hotspot_recall', 'Hotspot Recall', axes[2])
    ]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        data = []
        labels = []
        
        for planner in planners:
            planner_results = [r for r in results if r['planner_name'] == planner]
            values = [r[metric_key] for r in planner_results if metric_key in r and r[metric_key] is not None]
            
            if values:
                data.append(values)
                labels.append(planner)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            for patch, label in zip(bp['boxes'], labels):
                patch.set_facecolor(COLORS.get(label, 'gray'))
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric_label)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_scalability(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Planning Time Scalability"
):
    """
    Plot planning time vs number of robots.
    """
    setup_plot_style()
    
    planners = sorted(set(r['planner_name'] for r in results))
    robot_counts = sorted(set(r['num_robots'] for r in results))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for planner in planners:
        mean_times = []
        std_times = []
        
        for n_robots in robot_counts:
            planner_n_results = [
                r for r in results 
                if r['planner_name'] == planner and r['num_robots'] == n_robots
            ]
            
            if planner_n_results:
                times = [r['total_planning_time'] for r in planner_n_results]
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
            else:
                mean_times.append(None)
                std_times.append(None)
        
        # Filter out None values
        valid_data = [(n, m, s) for n, m, s in zip(robot_counts, mean_times, std_times) if m is not None]
        if valid_data:
            n_vals, m_vals, s_vals = zip(*valid_data)
            
            ax.errorbar(n_vals, m_vals, yerr=s_vals, marker='o', linewidth=2,
                       markersize=8, label=planner, color=COLORS.get(planner, 'gray'),
                       capsize=5)
    
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Planning Time (seconds)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(robot_counts)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_coverage_vs_time(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Coverage vs Time Budget"
):
    """
    Plot coverage fraction vs budget level for each planner.
    """
    setup_plot_style()
    
    planners = sorted(set(r['planner_name'] for r in results))
    budgets = sorted(set(r['initial_budget'] for r in results))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for planner in planners:
        mean_coverage = []
        std_coverage = []
        
        for budget in budgets:
            planner_budget_results = [
                r for r in results 
                if r['planner_name'] == planner and r['initial_budget'] == budget
            ]
            
            if planner_budget_results:
                coverages = [r['coverage_fraction'] * 100 for r in planner_budget_results]
                mean_coverage.append(np.mean(coverages))
                std_coverage.append(np.std(coverages))
            else:
                mean_coverage.append(None)
                std_coverage.append(None)
        
        # Filter out None values
        valid_data = [(b, m, s) for b, m, s in zip(budgets, mean_coverage, std_coverage) if m is not None]
        if valid_data:
            b_vals, m_vals, s_vals = zip(*valid_data)
            
            ax.errorbar(b_vals, m_vals, yerr=s_vals, marker='o', linewidth=2,
                       markersize=8, label=planner, color=COLORS.get(planner, 'gray'),
                       capsize=5)
    
    ax.set_xlabel('Time Budget (seconds)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(budgets)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_sar_metrics(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "SAR-Specific Metrics"
):
    """
    Plot SAR-specific metrics: hotspot recall and probability mass covered.
    """
    setup_plot_style()
    
    planners = sorted(set(r['planner_name'] for r in results))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Hotspot Recall
    ax1 = axes[0]
    recalls = []
    labels = []
    for planner in planners:
        planner_results = [r for r in results if r['planner_name'] == planner]
        recall_vals = [r.get('hotspot_recall', 0) * 100 for r in planner_results if r.get('hotspot_recall') is not None]
        if recall_vals:
            recalls.append(np.mean(recall_vals))
            labels.append(planner)
    
    bars1 = ax1.bar(range(len(labels)), recalls, color=[COLORS.get(p, 'gray') for p in labels], alpha=0.8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Hotspot Recall (%)')
    ax1.set_title('Hotspot Detection Performance')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Probability Mass Covered
    ax2 = axes[1]
    prob_masses = []
    labels2 = []
    for planner in planners:
        planner_results = [r for r in results if r['planner_name'] == planner]
        mass_vals = [r.get('prob_mass_covered', 0) * 100 for r in planner_results if r.get('prob_mass_covered') is not None]
        if mass_vals:
            prob_masses.append(np.mean(mass_vals))
            labels2.append(planner)
    
    bars2 = ax2.bar(range(len(labels2)), prob_masses, color=[COLORS.get(p, 'gray') for p in labels2], alpha=0.8)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, rotation=45, ha='right')
    ax2.set_ylabel('Probability Mass Covered (%)')
    ax2.set_title('Hotspot Probability Coverage')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_trajectories_comparison(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Robot Trajectories - Baseline Comparison"
):
    """
    Visualize robot trajectories for all 5 planners with 3D environment surfaces.
    
    Creates a professional grid matching assignment_demo.py style:
    - Row 1: 3D surfaces for planners 1-3
    - Row 2: 3D surfaces for planners 4-5 + legend
    """
    setup_plot_style()
    
    # Get all 5 planners in consistent order
    all_planners = ['Random', 'Lawnmower', 'SequentialGreedy', 'IndependentGreedy', 'Auction']
    planners = [p for p in all_planners if any(r['planner_name'] == p for r in results)]
    
    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    for idx, planner_name in enumerate(planners):
        # Get one result for this planner
        planner_results = [r for r in results if r['planner_name'] == planner_name]
        if not planner_results:
            continue
        
        result = planner_results[0]  # Use first repetition
        
        # Create 3D subplot
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Recreate environment for ground truth surface
        try:
            env = create_environment(
                bounds=np.array(result['bounds']),
                env_type='synthetic',
                function_name=result['env_function'],
                observation_noise=result['noise_std'],
                seed=result['env_seed'],
                physical_scale=1.0,
                **({'n_components': 4, 'spread': 'medium'} if result['env_function'] == 'gaussian_mixture' else {})
            )
            
            # Evaluate environment on grid
            X_grid, Y_grid, true_values = env.evaluate_grid(resolution=80)
            
            # Plot 3D surface with viridis colormap
            surf = ax.plot_surface(X_grid, Y_grid, true_values, cmap='viridis',
                                  linewidth=0, antialiased=True, alpha=0.85,
                                  vmin=true_values.min(), vmax=true_values.max())
            
        except Exception as e:
            print(f"  Warning: Could not recreate environment for {planner_name}: {e}")
            continue
        
        # Plot robot trajectories on 3D surface
        robot_trajectories = result.get('robot_trajectories', {})
        num_robots = result['num_robots']
        
        for robot_id in range(num_robots):
            traj_key = str(robot_id) if str(robot_id) in robot_trajectories else robot_id
            trajectory = robot_trajectories.get(traj_key, [])
            
            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
                
                # Get z-values for trajectory points
                traj_values = env.evaluate(trajectory)
                
                # Plot 3D trajectory path
                ax.plot(trajectory[:, 0], trajectory[:, 1], traj_values,
                       color=color, linewidth=3, alpha=0.9, zorder=10)
                
                # Mark start position with large sphere
                ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [traj_values[0]],
                          s=200, c=color, marker='o', edgecolors='black',
                          linewidths=2.5, zorder=15, alpha=1.0)
                
                # Mark measurement points
                ax.scatter(trajectory[:, 0], trajectory[:, 1], traj_values,
                          s=25, c=color, marker='o', alpha=0.6, zorder=12,
                          edgecolors='white', linewidths=0.5)
        
        # Mark depot with gold star
        start_pos = result['start_position']
        start_z = env.evaluate(np.array([start_pos]))[0]
        ax.scatter([start_pos[0]], [start_pos[1]], [start_z],
                  s=400, c='gold', marker='*', edgecolors='black',
                  linewidths=3, zorder=20)
        
        # Styling
        ax.set_xlabel('X (m)', labelpad=8)
        ax.set_ylabel('Y (m)', labelpad=8)
        ax.set_zlabel('Value', labelpad=8)
        ax.view_init(elev=25, azim=220)  # Match assignment_demo viewing angle
        
        # Title with planner name and metrics
        coverage = result['coverage_fraction'] * 100
        rmse = result['rmse_final']
        measurements = result['total_measurements']
        
        title_text = f"{planner_name}"
        subtitle = f"RMSE: {rmse:.4f} | Cov: {coverage:.1f}% | {measurements} meas"
        
        ax.set_title(f"{title_text}\n{subtitle}", 
                    fontweight='bold', fontsize=11, pad=10,
                    color=COLORS.get(planner_name, 'black'))
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.05)
    
    # Use the last subplot for legend/info
    if len(planners) <= 5:
        ax_legend = fig.add_subplot(2, 3, 6)
        ax_legend.axis('off')
        
        legend_text = """
        LEGEND
        
        ⭐  Depot (Start Position)
        ●   Robot Start/Path
        ·   Measurement Points
        
        Surface: Environment values
        Colormap: Viridis (blue→yellow)
        
        View: 25° elevation, 220° azimuth
        (matching assignment_demo style)
        
        Metrics:
        • RMSE: Root mean square error
        • Cov: Spatial coverage (%)
        • Meas: Total measurements
        """
        
        ax_legend.text(0.15, 0.5, legend_text, fontsize=10,
                      verticalalignment='center', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=1', facecolor='wheat', 
                               alpha=0.3, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_single_planner_detailed(
    result: Dict[str, Any],
    output_path: str
):
    """
    Create detailed visualization for a single planner result.
    
    Shows:
    - Ground truth environment
    - Robot trajectories with measurements
    - Individual robot paths
    """
    setup_plot_style()
    
    planner_name = result['planner_name']
    num_robots = result['num_robots']
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"{planner_name} Detailed Analysis", fontsize=16, fontweight='bold')
    
    # Recreate environment
    try:
        env = create_environment(
            bounds=np.array(result['bounds']),
            env_type='synthetic',
            function_name=result['env_function'],
            observation_noise=result['noise_std'],
            seed=result['env_seed'],
            physical_scale=1.0,
            **({'n_components': 4, 'spread': 'medium'} if result['env_function'] == 'gaussian_mixture' else {})
        )
        
        X_grid, Y_grid, true_values = env.evaluate_grid(resolution=50)
        
    except Exception as e:
        print(f"  Warning: Could not recreate environment: {e}")
        return
    
    # 1. Ground truth with all trajectories
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(X_grid, Y_grid, true_values, levels=25, cmap='viridis', alpha=0.8)
    plt.colorbar(im1, ax=ax1, label='Value', shrink=0.8)
    
    robot_trajectories = result.get('robot_trajectories', {})
    
    for robot_id in range(num_robots):
        traj_key = str(robot_id) if str(robot_id) in robot_trajectories else robot_id
        trajectory = robot_trajectories.get(traj_key, [])
        
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
            
            ax1.plot(trajectory[:, 0], trajectory[:, 1], 
                    color=color, linewidth=2.5, alpha=0.8, label=f'R{robot_id}', zorder=5)
            ax1.scatter(trajectory[0, 0], trajectory[0, 1],
                       s=120, c=color, marker='o', edgecolors='black', linewidths=2, zorder=10)
    
    start_pos = result['start_position']
    ax1.scatter(start_pos[0], start_pos[1], s=250, c='gold', marker='*',
               edgecolors='black', linewidths=2.5, zorder=15)
    
    ax1.set_title('Ground Truth + All Trajectories', fontweight='bold', fontsize=12)
    ax1.set_xlabel('X (m)', fontweight='bold')
    ax1.set_ylabel('Y (m)', fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_aspect('equal')
    
    # 2-4. Individual robot plots (up to 3 robots shown)
    for robot_id in range(min(3, num_robots)):
        ax = plt.subplot(2, 3, 2 + robot_id)
        im = ax.contourf(X_grid, Y_grid, true_values, levels=25, cmap='viridis', alpha=0.5)
        
        traj_key = str(robot_id) if str(robot_id) in robot_trajectories else robot_id
        trajectory = robot_trajectories.get(traj_key, [])
        
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            color = ROBOT_COLORS[robot_id]
            
            # Draw path with thicker line
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, linewidth=3.5, alpha=0.9, zorder=5)
            
            # Draw measurement points
            ax.scatter(trajectory[:, 0], trajectory[:, 1],
                      s=50, c=color, marker='o', alpha=0.6, 
                      edgecolors='white', linewidths=1.5, zorder=8)
            
            # Start marker (green circle)
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      s=180, c='lime', marker='o', edgecolors='black',
                      linewidths=2.5, zorder=10, label='Start')
            
            # End marker (red square)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      s=180, c='red', marker='s', edgecolors='black',
                      linewidths=2.5, zorder=10, label='End')
        
        measurements = result['per_robot_measurements'][robot_id]
        budget_used = result['per_robot_budget_used'][robot_id]
        
        ax.set_title(f'Robot {robot_id}\n{measurements} measurements | {budget_used:.1f}s used', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('X (m)', fontweight='bold')
        ax.set_ylabel('Y (m)', fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_aspect('equal')
    
    # 5. Metrics summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    metrics_text = f"""
    Planner: {planner_name}
    Scenario: {result['scenario_type']}
    
    Performance Metrics:
    • RMSE: {result['rmse_final']:.4f}
    • Coverage: {result['coverage_fraction']*100:.2f}%
    • Unique cells: {result['unique_cells_visited']}
    • Overlap: {result['overlap_fraction']:.2f}%
    
    Mission Stats:
    • Total measurements: {result['total_measurements']}
    • Total distance: {result['total_distance']:.1f}m
    • Iterations: {result['iterations']}
    • Planning time: {result['total_planning_time']:.2f}s
    """
    
    if 'hotspot_recall' in result and result['hotspot_recall'] is not None:
        metrics_text += f"""
    SAR Metrics:
    • Hotspot recall: {result['hotspot_recall']*100:.1f}%
    • Prob mass covered: {result.get('prob_mass_covered', 0)*100:.1f}%
        """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    
    # 6. Coverage heatmap (visited cells)
    ax6 = plt.subplot(2, 3, 6)
    
    # Create visited cell map
    bounds = np.array(result['bounds'])
    cell_size = 5.0  # Match coverage metric cell size
    nx = int((bounds[0, 1] - bounds[0, 0]) / cell_size)
    ny = int((bounds[1, 1] - bounds[1, 0]) / cell_size)
    visited_map = np.zeros((ny, nx))
    
    for robot_id in range(num_robots):
        traj_key = str(robot_id) if str(robot_id) in robot_trajectories else robot_id
        trajectory = robot_trajectories.get(traj_key, [])
        
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            for pos in trajectory:
                xi = int((pos[0] - bounds[0, 0]) / cell_size)
                yi = int((pos[1] - bounds[1, 0]) / cell_size)
                if 0 <= xi < nx and 0 <= yi < ny:
                    visited_map[yi, xi] += 1
    
    im6 = ax6.imshow(visited_map, cmap='Reds', origin='lower', 
                     extent=[bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]],
                     aspect='equal')
    plt.colorbar(im6, ax=ax6, label='Visit Count')
    ax6.set_title('Spatial Coverage Heatmap', fontweight='bold')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_results_summary(
    results: List[Dict[str, Any]],
    output_path: str
):
    """
    Create comprehensive visualization suite for experiment results.
    
    Generates all plots and saves to output directory.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"  Total results: {len(results)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Group results by scenario and configuration
    scenarios = set(r['scenario_type'] for r in results)
    robot_counts = set(r['num_robots'] for r in results)
    budgets = set(r['initial_budget'] for r in results)
    
    print(f"Scenarios: {scenarios}")
    print(f"Robot counts: {sorted(robot_counts)}")
    print(f"Budgets: {sorted(budgets)}")
    
    # 1. Overall comparison (all data)
    print("\n1. Creating overall comparison plots...")
    plot_final_metrics_comparison(
        results,
        str(output_dir / 'overall_comparison.png'),
        title='Overall Baseline Comparison'
    )
    
    # 2. Boxplots for variability
    print("2. Creating metric distribution boxplots...")
    plot_metric_boxplots(
        results,
        str(output_dir / 'metric_distributions.png'),
        title='Metric Distributions Across Repetitions'
    )
    
    # 3. Per-scenario comparisons
    print("3. Creating per-scenario comparisons...")
    for scenario in scenarios:
        scenario_results = filter_results(results, scenario_type=scenario)
        if scenario_results:
            plot_final_metrics_comparison(
                scenario_results,
                str(output_dir / f'{scenario}_comparison.png'),
                title=f'{scenario.replace("_", " ").title()} Scenario'
            )
            
            # SAR-specific plots
            if scenario == 'gaussian_hotspot':
                plot_sar_metrics(
                    scenario_results,
                    str(output_dir / f'{scenario}_sar_metrics.png'),
                    title='Search and Rescue Performance'
                )
    
    # 4. Scalability analysis (if multiple robot counts)
    if len(robot_counts) > 1:
        print("4. Creating scalability plots...")
        plot_scalability(
            results,
            str(output_dir / 'scalability.png'),
            title='Planning Time vs Number of Robots'
        )
    
    # 5. Budget analysis (if multiple budgets)
    if len(budgets) > 1:
        print("5. Creating budget analysis plots...")
        plot_coverage_vs_time(
            results,
            str(output_dir / 'coverage_vs_budget.png'),
            title='Coverage vs Time Budget'
        )
    
    # 6. Trajectory visualizations
    print("6. Creating trajectory visualizations...")
    
    # Comparison plot for all planners (use first repetition of each)
    for scenario in scenarios:
        scenario_results = filter_results(results, scenario_type=scenario)
        if scenario_results:
            # Get one result per planner
            planner_sample_results = []
            for planner in set(r['planner_name'] for r in scenario_results):
                planner_results = [r for r in scenario_results if r['planner_name'] == planner]
                if planner_results:
                    planner_sample_results.append(planner_results[0])
            
            plot_trajectories_comparison(
                planner_sample_results,
                str(output_dir / f'{scenario}_trajectories_comparison.png'),
                title=f'{scenario.replace("_", " ").title()} - Robot Trajectories'
            )
    
    # 7. Detailed individual planner plots (first result of each planner)
    print("7. Creating detailed planner visualizations...")
    detailed_dir = output_dir / 'detailed'
    detailed_dir.mkdir(exist_ok=True)
    
    planners_seen = set()
    for result in results:
        planner = result['planner_name']
        if planner not in planners_seen:
            planners_seen.add(planner)
            plot_single_planner_detailed(
                result,
                str(detailed_dir / f'{planner}_detailed.png')
            )
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    """Test plotting on existing experiment results."""
    import sys
    
    results_dir = 'results/experiments'
    output_dir = 'results/plots'
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Loading results from: {results_dir}")
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Loaded {len(results)} results")
    create_results_summary(results, output_dir)
