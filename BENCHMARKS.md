"""
Benchmark Environments and Baselines

This document describes the environments and baseline algorithms available
for evaluating the IPP framework.

## Environments

### Synthetic Test Functions

These environments provide controlled testbeds with known ground truth.

1. **Peaks Function** (2D)

   - MATLAB peaks function with multiple peaks and valleys
   - Domain: Typically [-3, 3] × [-3, 3]
   - Characteristics: Smooth, multimodal
   - Use case: General testing, visualization

2. **Ackley Function** (N-D)

   - Highly multimodal with many local minima
   - Domain: Typically [-5, 5]^N
   - Characteristics: Nearly flat outer region, many local minima
   - Use case: Test exploration vs. exploitation

3. **Rastrigin Function** (N-D)

   - Regularly distributed local minima
   - Domain: Typically [-5.12, 5.12]^N
   - Characteristics: Highly multimodal, difficult to optimize
   - Use case: Test ability to escape local minima

4. **Rosenbrock Function** (N-D)

   - Banana-shaped valley
   - Domain: Typically [-2, 2]^N
   - Characteristics: Narrow valley, difficult to navigate
   - Use case: Test path optimization in constrained spaces

5. **Sphere Function** (N-D)

   - Simple convex bowl
   - Domain: Any
   - Characteristics: Smooth, unimodal, easy
   - Use case: Sanity check, debugging

6. **Branin Function** (2D)

   - Three global minima
   - Domain: x ∈ [-5, 10], y ∈ [0, 15]
   - Characteristics: Standard Bayesian optimization benchmark
   - Use case: Test multi-optimum discovery

7. **Forrester Function** (1D)

   - Simple 1D function with local structure
   - Domain: [0, 1]
   - Characteristics: Used in surrogate modeling examples
   - Use case: 1D debugging, quick tests

8. **Townsend Function** (2D)

   - Non-convex with multiple local minima
   - Domain: x ∈ [-2.25, 2.5], y ∈ [-2.5, 1.75]
   - Global minimum at approximately (-0.0299, -0.1151)
   - Reference: Common optimization benchmark
   - Use case: Test local minima escape capability
   - **Literature**: Used to evaluate if planners can escape local minima

9. **Gaussian Mixture Model** (N-D)
   - Field with multiple Gaussian "hotspots" or survivor clusters
   - Domain: Configurable
   - Parameters:
     - n_components: Number of Gaussian components
     - means: Cluster centers (random or specified)
     - covs: Covariances (isotropic or full)
     - weights: Mixture weights
   - Use case: Search & Rescue scenarios
   - **Literature**: Singh, Krause & Kaiser, IJCAI 2009 - Used for disaster
     response simulations with 500 survivors in 1000 candidate sites

### Real-World Environments

These environments use actual field data or simulations.

1. **Regional Ocean Modeling System (ROMS) - Oregon Coast**

   - Source: https://data.ioos.us/dataset/regional-ocean-modeling-system-roms-oregon-coast
   - Type: Spatiotemporal ocean simulation
   - Resolution: ~2 km, 4-hourly forecasts, 3-day window
   - Variables: Temperature, salinity, currents, etc.
   - Format: NetCDF
   - Use case: Real-world spatiotemporal path planning
   - Implementation: `NetCDFEnvironment` class
   - Example:
     ```python
     env = create_environment(
         bounds=np.array([[-125, -123], [43, 45]]),
         env_type='netcdf',
         netcdf_path='roms_oregon.nc',
         variable_name='temp',
         time_index=0
     )
     ```

2. **Lunar Crater Hydration Map (LAMP)**

   - Source: Lyman Alpha Mapping Project remote sensing
   - Type: Real remote-sensing dataset
   - Characteristics: Two noisy lunar swaths
     - 3 km swath: 169 samples
     - 6 km swath: 625 samples
     - Grid step: 0.25 (normalized)
   - Ground truth: None (real noisy data)
   - Evaluation: RMS vs. observations
   - Use case: Test convergence under realistic noise without ground truth
   - Implementation: `InterpolatedDataEnvironment` class
   - Note: Data needs to be obtained separately

3. **Lake Haviland, Colorado (Field Data)**

   - Source: Yoo et al., 2015 field experiments
   - Location: 37°31′55″N, 107°48′27″W
   - Type: Lake surface temperature field
   - Sensor: Platypus Lutra ASV with thermistor
   - Ground truth: Full coverage survey
   - Use case: Field validation of IPP algorithms
   - Implementation: `InterpolatedDataEnvironment` class
   - Note: Data from paper authors or similar field collection

4. **Yoo et al. Random Scalar Fields (2015)**
   - Type: 2960 randomly generated 2D scalar fields
   - Characteristics: 5-50 Gaussian reward peaks per field
   - Purpose: Statistical evaluation of J-Horizon IPP
   - Ground truth: Known analytically
   - Use case: Large-scale statistical benchmarking
   - Implementation: Use `gaussian_mixture` with varying n_components

## Baseline Algorithms

### 1. Information-Driven Planner (IDP)

- **Reference**: Ma et al.
- **Method**: Discrete optimization to maximize mutual information (MI)
- **Strengths**: Theoretical guarantees, near-optimal for discrete spaces
- **Complexity**: High computational cost
- **Implementation status**: TODO

### 2. Continuous-Space Informative Path Planner (CIPP)

- **Reference**: Hitz et al.
- **Method**: CMA-ES (genetic algorithm) for continuous MI maximization
- **Strengths**: Works in continuous space, good exploration
- **Complexity**: Moderate, depends on CMA-ES iterations
- **Implementation status**: TODO (use pycma library)

### 3. Brute Force (Optimal)

- **Method**: Enumerate all feasible paths
- **Strengths**: Guaranteed optimal
- **Limitations**: Only practical for very small graphs (< 10 nodes)
- **Use case**: Sanity check on toy problems
- **Implementation status**: TODO (low priority)

### 4. Recursive Greedy (RG)

- **Reference**: Singh et al., 2009
- **Method**: Select nodes with best marginal information gain per cost
- **Strengths**: Near-optimal on submodular objectives
- **Complexity**: Slow (requires many GP updates)
- **Use case**: Strong baseline for comparison
- **Implementation status**: TODO

### 5. Standard Greedy

- **Method**: Choose next vertex with highest info-gain-to-cost ratio
- **Strengths**: Very fast, simple
- **Weaknesses**: Short-sighted, can get stuck
- **Use case**: Minimum performance baseline
- **Implementation status**: Easy to implement, high priority

### 6. Genetic Algorithm (GA)

- **Reference**: Holland 1975, Goldberg 1989
- **Method**: Evolutionary search (crossover, mutation)
- **Strengths**: Good global search
- **Weaknesses**: Sensitive to hyperparameters
- **Use case**: Compare evolutionary vs. tree search
- **Implementation status**: TODO (use DEAP library)

### 7. Random Planner

- **Method**: Random walk or random waypoint selection
- **Strengths**: No computation needed
- **Use case**: Sanity check (we should beat this!)
- **Implementation status**: Trivial to implement

### 8. Lawnmower / Grid Coverage

- **Method**: Systematic grid coverage
- **Strengths**: Simple, predictable
- **Use case**: Baseline for spatially uniform sampling
- **Implementation status**: Easy to implement

### 9. UCB-Based Greedy (Single-Step)

- **Method**: Greedy selection using UCB acquisition function
- **Strengths**: Fast, theoretically motivated
- **Use case**: Ablation study (your MCTS without lookahead)
- **Implementation status**: Easy (reuse your acquisition functions)

### 10. Your Algorithm - Ablations

- **Kriging Believer Only**: Multi-robot coordination without MCTS
- **MCTS Only**: Planning without KB coordination (collision avoidance)
- **No Quadtree**: Uniform candidate sampling
- **Use case**: Understand contribution of each component

## Experimental Design Recommendations

### Phase 1: Synthetic Validation

1. Test on simple functions (peaks, sphere, gaussian_mixture)
2. Verify algorithm correctness
3. Compare against greedy and random baselines

### Phase 2: Benchmark Comparison

1. Run on standard benchmarks (Townsend, Ackley, Branin)
2. Compare against IDP, CIPP, Recursive Greedy
3. Statistical analysis over multiple seeds

### Phase 3: Ablation Studies

1. Remove Kriging Believer → measure coordination benefit
2. Remove MCTS → measure planning benefit
3. Remove quadtree → measure adaptive sampling benefit
4. Vary budget, number of robots, planning horizon

### Phase 4: Real-World Validation

1. Test on LAMP lunar data (noisy, no ground truth)
2. Test on ROMS ocean data (spatiotemporal)
3. Field experiment (if possible) like Lake Haviland

## Metrics to Track

For each experiment, record:

- **Information gain**: Total mutual information or variance reduction
- **Coverage**: Percentage of high-uncertainty areas visited
- **RMSE**: Root mean squared error vs. ground truth (when available)
- **Computational time**: Planning time per cycle
- **Trajectory length**: Total distance traveled
- **Budget efficiency**: Information gain per unit budget
- **Convergence rate**: How quickly uncertainty decreases
- **Scalability**: Performance vs. number of robots, budget, area size

## Configuration Templates

See `config/experiment_configs/` for pre-configured experiments:

- `synthetic_peaks.yaml`: Quick test on peaks function
- `townsend_benchmark.yaml`: Townsend function benchmark
- `search_rescue.yaml`: Gaussian mixture S&R scenario
- `roms_ocean.yaml`: ROMS spatiotemporal planning
- `ablation_no_kb.yaml`: Ablation without Kriging Believer
- etc.

## References

1. Singh, A., Krause, A., & Kaiser, W. (2009). "Nonmyopic adaptive informative
   path planning for multiple robots." IJCAI 2009.
2. Yoo, C., Stuntz, A., Hollinger, G.A., & Smith, R.N. (2015). "Informative
   planning for multi-robot deployment."
3. Hitz, G., et al. "Continuous-space informative path planning."

4. Ma, K.C., et al. "Information-Driven Planner (IDP)."

5. ROMS: Regional Ocean Modeling System - https://www.myroms.org/

6. LAMP: Lyman Alpha Mapping Project lunar data
   """

# This file is for documentation only - no executable code
