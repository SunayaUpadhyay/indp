# Multi-Robot Informative Path Planning Framework

A modular research framework for multi-robot informative path planning using Gaussian Processes, MCTS planning, and Kriging Believer coordination.

## Overview

This framework implements a sophisticated IPP algorithm with the following key features:

- **Adaptive Candidate Generation**: Quadtree-based spatial refinement focusing on high-uncertainty regions
- **Decentralized Coordination**: Kriging Believer approach for conflict-free target assignment
- **MCTS Planning**: Monte Carlo Tree Search with receding horizon control
- **Modular Design**: Easy to swap components (GP backends, acquisition functions, planners)
- **Flexible Configuration**: YAML-based configuration for reproducible experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/SunayaUpadhyay/indp.git
cd indp

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.core import Robot, GaussianProcessBelief, Environment
from src.core.belief import create_gp_belief
from src.core.environment import create_environment
import numpy as np

# Create environment
env = create_environment(
    bounds=np.array([[0, 100], [0, 100]]),
    env_type='synthetic',
    function_name='peaks',
    observation_noise=0.1,
    seed=42
)

# Create GP belief
gp = create_gp_belief(
    bounds=np.array([[0, 100], [0, 100]]),
    backend='sklearn',
    kernel_type='rbf',
    length_scale=10.0,
    variance=1.0,
    noise=0.1
)

# Create robots
robots = [
    Robot(
        robot_id=i,
        initial_position=np.random.uniform([0, 0], [100, 100]),
        budget_type='distance',
        initial_budget=500.0,
        max_speed=1.0
    )
    for i in range(3)
]

# TODO: Main planning loop will be implemented
```

## Project Structure

```
ipp_framework/
├── config/                 # Configuration files
│   ├── default_config.yaml # Default parameters
│   └── experiment_configs/ # Experiment-specific configs
├── src/
│   ├── core/              # Core data structures
│   │   ├── robot.py       # Robot state and dynamics
│   │   ├── belief.py      # GP belief representation
│   │   └── environment.py # Environment/ground truth
│   ├── planning/          # Planning algorithms (TBD)
│   │   ├── candidates/    # Candidate generation
│   │   ├── assignment/    # Kriging Believer assignment
│   │   ├── mcts/          # MCTS planner
│   │   └── acquisition/   # Acquisition functions
│   ├── execution/         # Execution and GP updates (TBD)
│   ├── utils/             # Utilities
│   └── main.py            # Main orchestrator (TBD)
├── examples/              # Example scripts
├── experiments/           # Experiment scripts
├── tests/                 # Unit tests
└── results/               # Output data
```

## Algorithm Overview

The algorithm follows a receding horizon approach with six main steps per cycle:

- **A) Candidate Generation**: Quadtree adaptive refinement based on GP variance
- **B) Assignment**: Kriging Believer for conflict-free target selection
- **C) MCTS Planning**: Tree search within planning window
- **D) Segment Selection**: Extract immediate actions from plan
- **E) Execution**: Execute segment and collect measurements
- **F) GP Update**: Update belief with real observations

## Current Status

- ✅ Core data structures (Robot, GP, Environment)
- ✅ Configuration schema
- ✅ Benchmark environments and documentation
- 🔲 Candidate generation (Step A)
- 🔲 Kriging Believer assignment (Step B)
- 🔲 MCTS planner (Step C)
- 🔲 Main orchestrator loop
- 🔲 Visualization tools
- 🔲 Baseline implementations

## Features

### Environments

**Synthetic Functions** (for controlled experiments):

- Peaks, Ackley, Rastrigin, Rosenbrock, Sphere, Branin, Forrester
- Townsend (local minima testing)
- Gaussian Mixture (Search & Rescue scenarios)

**Real-World Data** (for validation):

- ROMS Oregon Coast ocean simulations
- LAMP lunar crater hydration data
- Lake Haviland field measurements
- Custom interpolated datasets

See [`BENCHMARKS.md`](BENCHMARKS.md) for full documentation.

## Configuration

The framework uses YAML configuration files. See `config/default_config.yaml` for all available options.

Key parameters:

- `robots.n_robots`: Number of robots
- `planning.window_length`: Planning horizon
- `planning.execution_step`: Replanning frequency
- `planning.mcts.n_iterations`: MCTS budget
- `gp.backend`: GP implementation ('sklearn', 'gpy', 'gpytorch')

## Advanced Assignment Options

`KrigingBelieverAssignment` supports an optional MCTS-backed acquisition policy and adaptive candidate refresh:

- Set `use_mcts_acquisition=True` (and optionally provide an `MCTSConfig`) to let a short-horizon MCTS search pick the next waypoint per robot. The planner normalizes variance gain by travel time, limits its candidate set via `mcts_candidate_limit`, and obeys its configured iteration/time caps to keep wall-clock cost predictable.
- Provide a `CandidateGenerator` when calling `assign_targets` and set `candidate_refresh_interval` to rebuild the quadtree/candidate sets every N new real samples. This keeps Step A aligned with the evolving GP while preserving already-targeted points.
- Use `candidate_budget_reserve` if you want feasibility checks during regeneration to keep a safety buffer on each robot's remaining budget.

## Running Examples

```bash
# Demonstrate different environments
python examples/environment_demo.py
```

This will create visualizations in `results/environment_examples.png`.

## Development

This is an active research project. Components are being developed incrementally with careful consideration of design choices.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Citation

[To be added upon publication]

## Contact

Sunaya Upadhyay - [GitHub](https://github.com/SunayaUpadhyay)

## Acknowledgments

This work is part of Independent Study research.
