#!/usr/bin/env python
"""
Main script for running baseline comparison experiments.

Usage:
    python run_experiments.py --mode quick    # Quick test with 2 robots, 1 budget
    python run_experiments.py --mode full     # Full sweep over all configurations
    python run_experiments.py --mode scenario --scenario gaussian  # Single scenario
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import ExperimentRunner
from scenarios import *
from experimental_config import *


def run_quick_test():
    """Quick test with minimal configuration."""
    print("\n" + "="*70)
    print("QUICK TEST MODE - All 5 Baselines")
    print("="*70)
    
    runner = ExperimentRunner(use_drones=False)
    
    scenario = QUICK_TEST
    
    runner.run_scenario_suite(
        scenario_type=scenario['scenario_type'],
        env_function=scenario['env_function'],
        planners=ALL_PLANNERS,  # ALL 5 planners
        num_robots_list=[2],  # Just 2 robots
        budgets=[BUDGET_TIGHT],  # Just tight budget
        num_repetitions=2,  # Only 2 reps
        env_kwargs=scenario['env_kwargs'],
        max_iterations=100
    )
    
    print("\n✅ Quick test complete!")


def run_single_scenario(scenario_name: str):
    """Run experiments for a single scenario."""
    scenario_map = {
        'gaussian': GAUSSIAN_HOTSPOT_SPARSE,
        'gaussian_dense': GAUSSIAN_HOTSPOT_DENSE,
        'smooth': SMOOTH_MAPPING,
        'ackley': RUGGED_ACKLEY,
        'townsend': RUGGED_TOWNSEND
    }
    
    if scenario_name not in scenario_map:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {list(scenario_map.keys())}")
        return
    
    scenario = scenario_map[scenario_name]
    
    print("\n" + "="*70)
    print(f"SINGLE SCENARIO MODE: {scenario_name}")
    print("="*70)
    
    runner = ExperimentRunner(use_drones=False)
    
    runner.run_scenario_suite(
        scenario_type=scenario['scenario_type'],
        env_function=scenario['env_function'],
        planners=ALL_PLANNERS,
        num_robots_list=NUM_ROBOTS_LIST,
        budgets=[BUDGET_TIGHT, BUDGET_MEDIUM, BUDGET_LOOSE],
        num_repetitions=NUM_REPETITIONS,
        env_kwargs=scenario['env_kwargs'],
        max_iterations=200
    )
    
    print(f"\n✅ Scenario '{scenario_name}' complete!")


def run_full_suite():
    """Run complete experimental suite."""
    print("\n" + "="*70)
    print("FULL EXPERIMENTAL SUITE")
    print(f"  Total scenarios: {len(ALL_SCENARIOS)}")
    print(f"  Planners: {ALL_PLANNERS}")
    print(f"  Robot counts: {NUM_ROBOTS_LIST}")
    print(f"  Budgets: 3 levels")
    print(f"  Repetitions: {NUM_REPETITIONS}")
    total = len(ALL_SCENARIOS) * len(ALL_PLANNERS) * len(NUM_ROBOTS_LIST) * 3 * NUM_REPETITIONS
    print(f"  Total experiments: {total}")
    print("="*70)
    
    response = input("\nThis will take significant time. Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    runner = ExperimentRunner(use_drones=False)
    
    for scenario in ALL_SCENARIOS:
        runner.run_scenario_suite(
            scenario_type=scenario['scenario_type'],
            env_function=scenario['env_function'],
            planners=ALL_PLANNERS,
            num_robots_list=NUM_ROBOTS_LIST,
            budgets=[BUDGET_TIGHT, BUDGET_MEDIUM, BUDGET_LOOSE],
            num_repetitions=NUM_REPETITIONS,
            env_kwargs=scenario['env_kwargs'],
            max_iterations=200
        )
    
    print("\n" + "="*70)
    print("🎉 FULL SUITE COMPLETE! 🎉")
    print(f"  Total results: {len(runner.results)}")
    print("="*70)


def run_scalability_test():
    """Run scalability analysis (just Gaussian scenario, all robot counts)."""
    print("\n" + "="*70)
    print("SCALABILITY TEST (Planning Time vs N)")
    print("="*70)
    
    runner = ExperimentRunner(use_drones=False)
    
    scenario = GAUSSIAN_HOTSPOT_SPARSE
    
    runner.run_scenario_suite(
        scenario_type=scenario['scenario_type'],
        env_function=scenario['env_function'],
        planners=ALL_PLANNERS,
        num_robots_list=NUM_ROBOTS_LIST,  # All robot counts
        budgets=[BUDGET_MEDIUM],  # Just medium budget
        num_repetitions=NUM_REPETITIONS,
        env_kwargs=scenario['env_kwargs'],
        max_iterations=200
    )
    
    print("\n✅ Scalability test complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run IPP baseline comparison experiments"
    )
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'scenario', 'scalability'],
        default='quick',
        help='Experiment mode'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        help='Scenario name for scenario mode (gaussian, smooth, ackley, townsend)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    
    elif args.mode == 'full':
        run_full_suite()
    
    elif args.mode == 'scenario':
        if not args.scenario:
            print("Error: --scenario required for scenario mode")
            return
        run_single_scenario(args.scenario)
    
    elif args.mode == 'scalability':
        run_scalability_test()


if __name__ == '__main__':
    main()
