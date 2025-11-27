#!/usr/bin/env python
"""Generate supervised fine-tuning data using Kriging Believer + MCTS."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure repo root is importable when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.environment import create_environment
from src.core.belief import create_gp_belief
from src.core.robot import Robot, BudgetType
from src.planning.assignment.kriging_believer import KrigingBelieverAssignment
from src.planning.candidates.candidate_generator import CandidateGenerator
from src.planning.mcts.mcts_planner import MCTSConfig

from experiments.experimental_config import (
    START_POSITION,
    MAX_SPEED,
    SENSOR_RANGE,
)


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    'gaussian_mixture': {
        'bounds': np.array([[0.0, 100.0], [0.0, 100.0]]),
        'physical_scale': 5.0,
        'observation_noise': 0.08,
        'env_kwargs': {'n_components': 5, 'covs': 10.0},
        'gp_params': {
            'kernel_type': 'rbf',
            'length_scale': 0.06,
            'variance': 2.5,
            'noise': 0.09,
            'use_normalized_coords': True,
        },
        'quadtree_config': {
            'max_depth': 9,
            'min_cell_size': 0.6,
            'variance_threshold': 0.5,
        },
        'sampling_config': {
            'method': 'grid',
            'points_per_cell': 9,
            'min_spacing': 0.5,
        },
        'budget_reserve': 2.0,
    },
    'townsend': {
        'bounds': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
        'physical_scale': 20.0,
        'observation_noise': 0.05,
        'env_kwargs': {},
        'gp_params': {
            'kernel_type': 'matern',
            'length_scale': 0.2,
            'variance': 1.0,
            'noise': 0.05,
            'use_normalized_coords': True,
        },
        'quadtree_config': {
            'max_depth': 10,
            'min_cell_size': 0.8,
            'variance_threshold': 0.15,
        },
        'sampling_config': {
            'method': 'grid',
            'points_per_cell': 8,
            'min_spacing': 0.2,
        },
        'budget_reserve': 3.0,
    },
}


class ExpertDatasetBuilder:
    """Collect assignment snapshots, render tensors, and save HF-ready records."""

    def __init__(
        self,
        bounds: np.ndarray,
        grid_resolution: int,
        output_dir: Path,
        coord_metadata: Dict[str, Any],
        response_space: str,
    ):
        self.bounds = np.array(bounds, dtype=float)
        self.grid_resolution = grid_resolution
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.mean_dir = self.images_dir / 'mean'
        self.variance_dir = self.images_dir / 'variance'
        self.mean_dir.mkdir(parents=True, exist_ok=True)
        self.variance_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.sample_counter = 0
        self.grid_points = self._make_grid(bounds)
        self.current_mission_id: Optional[int] = None
        self.current_env_seed: Optional[int] = None
        self.current_run_seed: Optional[int] = None
        self.current_scenario: Optional[str] = None
        self.coord_metadata = coord_metadata
        self.response_space = response_space

    def _make_grid(self, bounds: np.ndarray) -> np.ndarray:
        x = np.linspace(bounds[0, 0], bounds[0, 1], self.grid_resolution)
        y = np.linspace(bounds[1, 0], bounds[1, 1], self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        return np.c_[X.ravel(), Y.ravel()]

    def start_mission(
        self,
        mission_id: int,
        env_seed: int,
        run_seed: int,
        scenario: str,
    ) -> None:
        self.current_mission_id = mission_id
        self.current_env_seed = env_seed
        self.current_run_seed = run_seed
        self.current_scenario = scenario

    def handle_decision(self, snapshot: Dict[str, Any]) -> None:
        gp = snapshot.get('gp_believer')
        if gp is None or self.current_mission_id is None:
            return
        mean, std = gp.predict(self.grid_points, return_std=True)
        mean_grid = mean.reshape(self.grid_resolution, self.grid_resolution)
        var_grid = (std ** 2).reshape(self.grid_resolution, self.grid_resolution)
        sample_idx = self.sample_counter
        mean_path = self._save_field_map(
            field=mean_grid,
            sample_idx=sample_idx,
            kind='mean',
            robot_states=snapshot['robot_states'],
        )
        variance_path = self._save_field_map(
            field=var_grid,
            sample_idx=sample_idx,
            kind='variance',
            robot_states=snapshot['robot_states'],
        )
        entry = self._build_entry(snapshot, mean_path, variance_path)
        self.records.append(entry)
        self.sample_counter += 1
    def _save_field_map(
        self,
        field: np.ndarray,
        sample_idx: int,
        kind: str,
        robot_states: Dict[int, Dict[str, Any]],
    ) -> Path:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
        extent = [self.bounds[0, 0], self.bounds[0, 1], self.bounds[1, 0], self.bounds[1, 1]]
        cmap = 'viridis' if kind == 'mean' else 'magma'
        title = 'GP Mean' if kind == 'mean' else 'GP Variance'
        im = ax.imshow(field, origin='lower', extent=extent, cmap=cmap, aspect='auto')
        for rid, info in robot_states.items():
            pos = np.asarray(info['position'], dtype=float)
            ax.scatter(pos[0], pos[1], c='white', edgecolors='black', s=18, linewidths=0.5)
            ax.text(pos[0], pos[1], f"R{rid}", color='white', fontsize=6, ha='center', va='center')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label(title)
        target_dir = self.mean_dir if kind == 'mean' else self.variance_dir
        path = target_dir / f"mission{self.current_mission_id:03d}_sample{sample_idx:05d}_{kind}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path

    def _build_entry(
        self,
        snapshot: Dict[str, Any],
        mean_image_path: Path,
        variance_image_path: Path,
    ) -> Dict[str, Any]:
        robot_id = snapshot['robot_id']
        time_stamp = float(snapshot['time'])
        target_env = self._clip_to_bounds(np.asarray(snapshot['selected_target'], dtype=float))
        target_response = self._convert_point(target_env)
        robot_states_response, robot_states_env = self._convert_robot_states(snapshot['robot_states'])
        prompt_lines = [
            (
                f"Robot {rid}: pos=({state['position'][0]:.2f}, {state['position'][1]:.2f}), "
                f"remaining={state['remaining_budget']:.1f}s, "
                f"assigned={len(state['assigned_targets'])}"
            )
            for rid, state in robot_states_response.items()
        ]
        bounds_text = self._format_bounds_text()
        prompt_text = (
            "Use the attached GP mean/variance maps to choose the next waypoint for "
            f"robot {robot_id}.\n"
            f"Mission time: {time_stamp:.1f}s.\n"
            "State summary:\n" + "\n".join(prompt_lines) +
            f"\nEnvironment bounds: {bounds_text}.\n"
            f"Coordinate space: {self._coordinate_label()}. Keep the answer inside bounds."
            "\nRespond with the target coordinates as 'x, y'."
        )
        response_text = f"{target_response[0]:.3f}, {target_response[1]:.3f}"
        mean_rel = mean_image_path.relative_to(self.output_dir).as_posix()
        var_rel = variance_image_path.relative_to(self.output_dir).as_posix()
        return {
            'id': f"mission{self.current_mission_id:03d}_sample{len(self.records):05d}",
            'mission_id': int(self.current_mission_id),
            'scenario': self.current_scenario,
            'env_seed': int(self.current_env_seed),
            'run_seed': int(self.current_run_seed),
            'robot_id': int(robot_id),
            'time': time_stamp,
            'mean_image_path': mean_rel,
            'variance_image_path': var_rel,
            'prompt': prompt_text,
            'response': response_text,
            'selected_target': target_response.round(4).tolist(),
            'selected_target_environment': target_env.round(4).tolist(),
            'robot_states': robot_states_response,
            'robot_states_environment': robot_states_env,
            'response_space': self.response_space,
            'coordinate_metadata': self.coord_metadata,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'path': mean_rel, 'modality': 'gp_mean'},
                        {'type': 'image', 'path': var_rel, 'modality': 'gp_variance'},
                        {'type': 'text', 'text': prompt_text},
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': response_text}],
                },
            ],
        }

    def _convert_robot_states(
        self,
        robot_states: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        converted: Dict[int, Dict[str, Any]] = {}
        raw: Dict[int, Dict[str, Any]] = {}
        for rid, info in robot_states.items():
            position_env = self._clip_to_bounds(np.asarray(info['position'], dtype=float))
            assigned_env = [
                self._clip_to_bounds(np.asarray(pt, dtype=float))
                for pt in info['assigned_targets']
            ]
            position_resp = self._convert_point(position_env)
            assigned_resp = [self._convert_point(pt) for pt in assigned_env]
            state_raw = {
                'position': position_env.round(4).tolist(),
                'remaining_budget': float(info['remaining_budget']),
                'assigned_targets': [pt.round(4).tolist() for pt in assigned_env],
            }
            state_conv = {
                'position': position_resp.round(4).tolist(),
                'remaining_budget': float(info['remaining_budget']),
                'assigned_targets': [pt.round(4).tolist() for pt in assigned_resp],
            }
            raw[int(rid)] = state_raw
            converted[int(rid)] = state_conv
        return converted, raw

    def _convert_point(self, point: np.ndarray) -> np.ndarray:
        if self.response_space == 'environment':
            return point
        if self.response_space == 'normalized':
            denom = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-9)
            normalized = (point - self.bounds[:, 0]) / denom
            return np.clip(normalized, 0.0, 1.0)
        if self.response_space == 'meters':
            scale = float(self.coord_metadata.get('physical_scale', 1.0))
            return point * scale
        raise ValueError(f"Unknown response space: {self.response_space}")

    def _clip_to_bounds(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.bounds[:, 0], self.bounds[:, 1])

    def _format_bounds_text(self) -> str:
        if self.response_space == 'normalized':
            return 'x ∈ [0, 1], y ∈ [0, 1]'
        if self.response_space == 'meters':
            scale = float(self.coord_metadata.get('physical_scale', 1.0))
            x_bounds = self.bounds[0] * scale
            y_bounds = self.bounds[1] * scale
            return (
                f"x ∈ [{x_bounds[0]:.2f}, {x_bounds[1]:.2f}] m, "
                f"y ∈ [{y_bounds[0]:.2f}, {y_bounds[1]:.2f}] m"
            )
        return (
            f"x ∈ [{self.bounds[0, 0]:.2f}, {self.bounds[0, 1]:.2f}], "
            f"y ∈ [{self.bounds[1, 0]:.2f}, {self.bounds[1, 1]:.2f}]"
        )

    def _coordinate_label(self) -> str:
        if self.response_space == 'normalized':
            return 'normalized [0, 1] coordinates'
        if self.response_space == 'meters':
            return 'meters'
        scale = float(self.coord_metadata.get('physical_scale', 1.0))
        if abs(scale - 1.0) < 1e-6:
            return 'meters'
        return f"environment units (1 unit = {scale:.2f} m)"

    def write_outputs(self, metadata: Dict[str, Any]) -> None:
        dataset_path = self.output_dir / 'expert_dataset.jsonl'
        with dataset_path.open('w', encoding='utf-8') as f:
            for record in self.records:
                json.dump(record, f)
                f.write('\n')
        meta = {
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'num_samples': len(self.records),
            'image_dir': self.images_dir.relative_to(self.output_dir).as_posix(),
            'mean_image_dir': self.mean_dir.relative_to(self.output_dir).as_posix(),
            'variance_image_dir': self.variance_dir.relative_to(self.output_dir).as_posix(),
            'response_space': self.response_space,
            'coordinate_metadata': self.coord_metadata,
            **metadata,
        }
        with (self.output_dir / 'metadata.json').open('w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate expert planner dataset')
    parser.add_argument('--scenario', default='gaussian_mixture', choices=SCENARIO_PRESETS.keys())
    parser.add_argument('--num-missions', type=int, default=2, help='How many missions to simulate')
    parser.add_argument('--num-robots', type=int, default=3)
    parser.add_argument('--robot-budget', type=float, default=150.0)
    parser.add_argument('--time-limit', type=float, default=150.0)
    parser.add_argument('--sensor-time', type=float, default=1.0)
    parser.add_argument('--n-initial-samples', type=int, default=5)
    parser.add_argument('--grid-resolution', type=int, default=64)
    parser.add_argument('--output-dir', type=Path, default=Path('results/expert_dataset'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--response-space',
        choices=['environment', 'meters', 'normalized'],
        default='environment',
        help='Coordinate space to express prompts/responses in',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mcts-iterations', type=int, default=600)
    parser.add_argument('--mcts-max-depth', type=int, default=10)
    parser.add_argument('--mcts-sim-depth', type=int, default=5)
    parser.add_argument('--mcts-exploration', type=float, default=1.2)
    parser.add_argument('--mcts-discount', type=float, default=0.9)
    parser.add_argument('--mcts-pw-alpha', type=float, default=0.7)
    parser.add_argument('--mcts-pw-constant', type=float, default=1.5)
    parser.add_argument('--mcts-time-limit', type=float, default=1.0)
    parser.add_argument('--mcts-candidate-limit', type=int, default=250)
    parser.add_argument('--candidate-refresh-interval', type=int, default=5)
    return parser.parse_args()


def run_mission(mission_id: int, args: argparse.Namespace, builder: ExpertDatasetBuilder) -> None:
    preset = SCENARIO_PRESETS[args.scenario]
    env_seed = args.seed + mission_id
    run_seed = args.seed * 97 + mission_id
    rng = np.random.default_rng(run_seed)

    bounds = preset['bounds']
    env = create_environment(
        bounds=bounds,
        env_type='synthetic',
        function_name=args.scenario,
        physical_scale=preset['physical_scale'],
        observation_noise=preset['observation_noise'],
        seed=env_seed,
        **preset.get('env_kwargs', {}),
    )

    gp = create_gp_belief(bounds, **preset['gp_params'])
    init_points = rng.uniform(
        low=[bounds[0, 0], bounds[1, 0]],
        high=[bounds[0, 1], bounds[1, 1]],
        size=(args.n_initial_samples, 2),
    )
    init_values = env.evaluate(init_points)
    gp.update(init_points, init_values)

    robots: List[Robot] = []
    for rid in range(args.num_robots):
        robots.append(
            Robot(
                robot_id=rid,
                initial_position=START_POSITION.copy(),
                budget_type=BudgetType.TIME,
                initial_budget=args.robot_budget,
                max_speed=MAX_SPEED,
                sensor_range=SENSOR_RANGE,
                environment=env,
            )
        )

    generator = CandidateGenerator(
        bounds=bounds,
        quadtree_config=preset['quadtree_config'],
        sampling_config=preset['sampling_config'],
    )
    candidate_sets = generator.generate_candidates(
        gp=gp.copy(),
        robots=robots,
        budget_reserve=preset.get('budget_reserve', 2.0),
    )

    mcts_config = MCTSConfig(
        iterations=args.mcts_iterations,
        exploration_constant=args.mcts_exploration,
        max_depth=args.mcts_max_depth,
        simulation_depth=args.mcts_sim_depth,
        discount_factor=args.mcts_discount,
        pw_alpha=args.mcts_pw_alpha,
        pw_constant=args.mcts_pw_constant,
        time_limit=args.mcts_time_limit,
        verbose=args.verbose,
    )

    planner = KrigingBelieverAssignment(
        time_limit=args.time_limit,
        environment=env,
        min_time_threshold=3.0,
        sensor_time=args.sensor_time,
        verbose=args.verbose,
        use_mcts_acquisition=True,
        mcts_config=mcts_config,
        mcts_candidate_limit=args.mcts_candidate_limit,
        candidate_refresh_interval=args.candidate_refresh_interval,
        candidate_budget_reserve=preset.get('budget_reserve', 2.0),
    )

    builder.start_mission(mission_id, env_seed, run_seed, args.scenario)
    env_sampler = lambda pos: env.evaluate(pos.reshape(1, -1))[0]

    planner.assign_targets(
        robots=robots,
        candidate_sets=candidate_sets,
        gp_belief=gp,
        environment_sampler=env_sampler,
        candidate_generator=generator,
        decision_logger=builder.handle_decision,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = SCENARIO_PRESETS[args.scenario]
    coord_metadata = {
        'bounds': preset['bounds'].astype(float).tolist(),
        'physical_scale': float(preset['physical_scale']),
        'use_normalized_coords': bool(preset.get('env_kwargs', {}).get('use_normalized_coords', False)),
    }
    builder = ExpertDatasetBuilder(
        bounds=preset['bounds'],
        grid_resolution=args.grid_resolution,
        output_dir=output_dir,
        coord_metadata=coord_metadata,
        response_space=args.response_space,
    )
    print(f"Generating {args.num_missions} missions for scenario '{args.scenario}'...")
    for mission_id in range(args.num_missions):
        print(f"  Mission {mission_id + 1}/{args.num_missions}")
        run_mission(mission_id, args, builder)
    builder.write_outputs({
        'scenario': args.scenario,
        'num_missions': args.num_missions,
        'num_robots': args.num_robots,
        'time_limit': args.time_limit,
        'sensor_time': args.sensor_time,
        'grid_resolution': args.grid_resolution,
    })
    print(f"Done. Dataset saved to {output_dir}.")


if __name__ == '__main__':
    main()
