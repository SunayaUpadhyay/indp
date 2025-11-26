"""Generate comprehensive comparison plots for baseline planners."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plotting import setup_plot_style, plot_trajectories_comparison

# Setup
setup_plot_style()
results_dir = Path('results/experiments')
plots_dir = Path('results/plots')
plots_dir.mkdir(parents=True, exist_ok=True)

# Load all results
results = []
for json_file in sorted(results_dir.glob('*.json')):
    with open(json_file, 'r') as f:
        result = json.load(f)
        results.append(result)

print(f"Loaded {len(results)} results\n")

# Group by planner
planner_results = {}
for result in results:
    planner = result['planner_name']
    if planner not in planner_results:
        planner_results[planner] = []
    planner_results[planner].append(result)

planners = ['Random', 'Lawnmower', 'SequentialGreedy', 'IndependentGreedy', 'Auction']
colors = {
    'Random': '#e74c3c',
    'Lawnmower': '#3498db', 
    'SequentialGreedy': '#2ecc71',
    'IndependentGreedy': '#f39c12',
    'Auction': '#9b59b6'
}

# ============================================================================
# 1. TRAJECTORY COMPARISON (3D surfaces)
# ============================================================================
print("Generating trajectory comparison plot...")
plot_trajectories_comparison(
    results,
    'results/plots/1_trajectories_comparison.png',
    title="Robot Trajectories - All 5 Baselines (Event-Driven Parallel Execution)"
)
print("  ✓ Saved: results/plots/1_trajectories_comparison.png\n")

# ============================================================================
# 2. COVERAGE & MEASUREMENTS BAR CHART
# ============================================================================
print("Generating coverage & measurements bar chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Coverage
coverage_data = {p: [r['coverage_fraction']*100 for r in planner_results[p]] for p in planners}
positions = np.arange(len(planners))
means = [np.mean(coverage_data[p]) for p in planners]
stds = [np.std(coverage_data[p]) for p in planners]

bars = ax1.bar(positions, means, yerr=stds, capsize=5, 
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax1.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Area Coverage Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(positions)
ax1.set_xticklabels(planners, rotation=15, ha='right')
ax1.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Measurements
meas_data = {p: [r['total_measurements'] for r in planner_results[p]] for p in planners}
means = [np.mean(meas_data[p]) for p in planners]
stds = [np.std(meas_data[p]) for p in planners]

bars = ax2.bar(positions, means, yerr=stds, capsize=5,
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Measurements', fontsize=12, fontweight='bold')
ax2.set_title('Number of Measurements Taken', fontsize=14, fontweight='bold')
ax2.set_xticks(positions)
ax2.set_xticklabels(planners, rotation=15, ha='right')
ax2.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(mean)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Coverage & Measurement Efficiency (2 Robots, 150s Budget)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/plots/2_coverage_measurements.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/plots/2_coverage_measurements.png\n")
plt.close()

# ============================================================================
# 3. RMSE & PLANNING TIME
# ============================================================================
print("Generating RMSE & planning time comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# RMSE
rmse_data = {p: [r['rmse_final'] for r in planner_results[p]] for p in planners}
means = [np.mean(rmse_data[p]) for p in planners]
stds = [np.std(rmse_data[p]) for p in planners]

bars = ax1.bar(positions, means, yerr=stds, capsize=5,
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax1.set_ylabel('Final RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Accuracy (Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_xticks(positions)
ax1.set_xticklabels(planners, rotation=15, ha='right')
ax1.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Planning Time (log scale)
time_data = {p: [r['total_planning_time'] for r in planner_results[p]] for p in planners}
means = [np.mean(time_data[p]) for p in planners]
stds = [np.std(time_data[p]) for p in planners]

bars = ax2.bar(positions, means, yerr=stds, capsize=5,
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax2.set_ylabel('Planning Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Computational Cost', fontsize=14, fontweight='bold')
ax2.set_xticks(positions)
ax2.set_xticklabels(planners, rotation=15, ha='right')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3, which='both')

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Accuracy & Computational Efficiency', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/plots/3_rmse_planning_time.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/plots/3_rmse_planning_time.png\n")
plt.close()

# ============================================================================
# 4. SAR METRICS (Hotspot Recall & Probability Mass)
# ============================================================================
print("Generating SAR metrics comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Hotspot Recall
recall_data = {p: [r['hotspot_recall']*100 for r in planner_results[p]] for p in planners}
means = [np.mean(recall_data[p]) for p in planners]
stds = [np.std(recall_data[p]) for p in planners]

bars = ax1.bar(positions, means, yerr=stds, capsize=5,
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hotspot Recall (%)', fontsize=12, fontweight='bold')
ax1.set_title('Search & Rescue: Hotspot Detection', fontsize=14, fontweight='bold')
ax1.set_xticks(positions)
ax1.set_xticklabels(planners, rotation=15, ha='right')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Probability Mass
prob_data = {p: [r['prob_mass_covered']*100 for r in planner_results[p]] for p in planners}
means = [np.mean(prob_data[p]) for p in planners]
stds = [np.std(prob_data[p]) for p in planners]

bars = ax2.bar(positions, means, yerr=stds, capsize=5,
               color=[colors[p] for p in planners], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Planner', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability Mass Covered (%)', fontsize=12, fontweight='bold')
ax2.set_title('Search & Rescue: Area Prioritization', fontsize=14, fontweight='bold')
ax2.set_xticks(positions)
ax2.set_xticklabels(planners, rotation=15, ha='right')
ax2.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Search & Rescue Performance', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/plots/4_sar_metrics.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/plots/4_sar_metrics.png\n")
plt.close()

# ============================================================================
# 5. SUMMARY STATISTICS TABLE
# ============================================================================
print("Generating summary statistics...")

# Create summary table
summary = []
for planner in planners:
    results_p = planner_results[planner]
    
    summary.append({
        'Planner': planner,
        'Coverage (%)': f"{np.mean([r['coverage_fraction']*100 for r in results_p]):.1f} ± {np.std([r['coverage_fraction']*100 for r in results_p]):.1f}",
        'Measurements': f"{np.mean([r['total_measurements'] for r in results_p]):.0f} ± {np.std([r['total_measurements'] for r in results_p]):.0f}",
        'RMSE': f"{np.mean([r['rmse_final'] for r in results_p]):.4f} ± {np.std([r['rmse_final'] for r in results_p]):.4f}",
        'Planning Time (s)': f"{np.mean([r['total_planning_time'] for r in results_p]):.2f} ± {np.std([r['total_planning_time'] for r in results_p]):.2f}",
        'Hotspot Recall (%)': f"{np.mean([r['hotspot_recall']*100 for r in results_p]):.0f} ± {np.std([r['hotspot_recall']*100 for r in results_p]):.0f}",
        'Prob Mass (%)': f"{np.mean([r['prob_mass_covered']*100 for r in results_p]):.1f} ± {np.std([r['prob_mass_covered']*100 for r in results_p]):.1f}",
    })

# Print table
print("\n" + "="*120)
print("BASELINE COMPARISON SUMMARY (2 Robots, 150s Budget, Event-Driven Parallel Execution)")
print("="*120)
print(f"{'Planner':<20} {'Coverage':<18} {'Measurements':<18} {'RMSE':<25} {'Planning Time':<22} {'Hotspot Recall':<18} {'Prob Mass':<15}")
print("-"*120)
for row in summary:
    print(f"{row['Planner']:<20} {row['Coverage (%)']:<18} {row['Measurements']:<18} {row['RMSE']:<25} {row['Planning Time (s)']:<22} {row['Hotspot Recall (%)']:<18} {row['Prob Mass (%)']:<15}")
print("="*120)

print("\n✅ All plots generated successfully!")
print("\nGenerated files:")
print("  1. results/plots/1_trajectories_comparison.png - 3D trajectory visualization")
print("  2. results/plots/2_coverage_measurements.png - Coverage and measurements")
print("  3. results/plots/3_rmse_planning_time.png - Accuracy and computational cost")
print("  4. results/plots/4_sar_metrics.png - Search & rescue performance")
