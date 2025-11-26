"""Generate trajectory visualizations for all planners."""

import json
from pathlib import Path
from plotting import plot_trajectories_comparison

# Load all results
results_dir = Path('results/experiments')
results = []

for json_file in results_dir.glob('*.json'):
    with open(json_file, 'r') as f:
        result = json.load(f)
        results.append(result)

print(f"Loaded {len(results)} results")

# Generate trajectory comparison plot
output_path = 'results/plots/all_planners_trajectories_fixed.png'
Path('results/plots').mkdir(parents=True, exist_ok=True)

plot_trajectories_comparison(
    results,
    output_path,
    title="Robot Trajectories - All 5 Baselines (After Bug Fixes)"
)

print(f"\n✓ Plot saved to: {output_path}")
