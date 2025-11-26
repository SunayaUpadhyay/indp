"""
Kriging Believer assignment for decentralized multi-robot target selection.

The Kriging Believer approach enables conflict-free target assignment without
explicit coordination:

1. Each robot selects its next target from its candidate set
2. Selected targets are added to a "believer" GP before robots reach them
3. The believer GP acts as if these targets are already sampled
4. This implicitly prevents other robots from selecting nearby locations
5. The actual GP is only updated when robots physically reach targets

This approach eliminates the need for explicit radius-based candidate removal
and enables natural decentralized coordination.
"""

import numpy as np
import heapq
from typing import Any, List, Dict, Tuple, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from copy import deepcopy

from ...core.robot import Robot
from ...core.belief import GaussianProcessBelief
from ..candidates.candidate_generator import CandidateSet, CandidateGenerator
from ..mcts.mcts_planner import MCTSPlanner, MCTSConfig

if TYPE_CHECKING:
    from ...core.environment import Environment


@dataclass
class AssignmentEvent:
    """Event in the assignment simulation."""
    time: float
    robot_id: int
    event_type: str  # 'reach_target', 'return_home'
    position: Optional[np.ndarray] = None
    
    def __lt__(self, other):
        """Priority queue comparison (earlier times first)."""
        return self.time < other.time


@dataclass
class RobotAssignmentState:
    """Extended robot state for assignment tracking."""
    robot: Robot
    current_target: Optional[np.ndarray] = None
    time_to_target: Optional[float] = None
    assigned_targets: List[np.ndarray] = field(default_factory=list)
    samples_collected: List[Tuple[np.ndarray, float, float]] = field(default_factory=list)
    
    @property
    def position(self) -> np.ndarray:
        """Current robot position."""
        return self.robot.position
    
    @property
    def has_target(self) -> bool:
        """Whether robot currently has an assigned target."""
        return self.current_target is not None


class KrigingBelieverAssignment:
    """
    Kriging Believer target assignment for multi-robot exploration.
    
    This class manages the decentralized assignment of targets to robots:
    - Maintains a "believer" GP that includes planned but not-yet-reached targets
    - Schedules robot movements using an event queue
    - Updates the actual GP only when robots reach targets
    - Prevents conflicts through implicit coordination via the believer GP
    
    Design rationale:
    - Event-driven simulation for concurrent robot operations
    - Kriging believer for natural decentralized coordination
    - No explicit exclusion zones needed (implicit through GP variance reduction)
    - Time-based budget management for realistic mission scenarios
    """
    
    def __init__(
        self,
        time_limit: float,
        environment: 'Environment',
        min_time_threshold: float = 60.0,
        sensor_time: float = 5.0,
        acquisition_function: Optional[Callable] = None,
        verbose: bool = True,
        use_mcts_acquisition: bool = False,
        mcts_config: Optional[MCTSConfig] = None,
        mcts_candidate_limit: int = 25,
        candidate_refresh_interval: Optional[int] = None,
        candidate_budget_reserve: float = 0.0
    ):
        """
        Initialize kriging believer assignment.
        
        Args:
            time_limit: Maximum mission duration (seconds)
            environment: Environment for coordinate/distance conversions
            min_time_threshold: Minimum remaining time to assign new target (seconds)
            sensor_time: Time required to take a measurement (seconds)
            acquisition_function: Function to select next target (if None, uses default MI)
            verbose: Whether to print progress messages
            use_mcts_acquisition: If True, use MCTS planner instead of greedy acquisition
            mcts_config: Optional configuration override for the MCTS planner
            mcts_candidate_limit: Max candidates passed into MCTS to keep branching bounded
            candidate_refresh_interval: Rebuild quadtree/candidates after this many new samples
            candidate_budget_reserve: Reserve when regenerating feasibility checks
        """
        self.time_limit = time_limit
        self.environment = environment
        self.min_time_threshold = min_time_threshold
        self.sensor_time = sensor_time
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        self.use_mcts_acquisition = use_mcts_acquisition
        self.mcts_candidate_limit = max(1, mcts_candidate_limit)
        self.mcts_config = deepcopy(mcts_config) if mcts_config is not None else MCTSConfig(
            iterations=250,
            max_depth=6,
            simulation_depth=4,
            time_limit=0.2
        )
        self._mcts_planner: Optional[MCTSPlanner] = None
        self.candidate_refresh_interval = candidate_refresh_interval
        self.candidate_budget_reserve = candidate_budget_reserve
        self._candidate_generator: Optional[CandidateGenerator] = None
        self._samples_since_candidate_refresh: int = 0
        self._decision_logger: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Assignment state
        self.robot_states: Dict[int, RobotAssignmentState] = {}
        self.event_queue: List[AssignmentEvent] = []
        self.simulation_clock: float = 0.0
        
        # Global data tracking
        self.all_target_points: List[Tuple[np.ndarray, int]] = []  # (position, robot_id)
        self.global_samples: List[Tuple[np.ndarray, float, float, int]] = []  # (pos, value, time, robot_id)
        
        # GP tracking
        self.gp_believer: Optional[GaussianProcessBelief] = None
        self.gp_actual: Optional[GaussianProcessBelief] = None
        
    def assign_targets(
        self,
        robots: List[Robot],
        candidate_sets: Dict[int, CandidateSet],
        gp_belief: GaussianProcessBelief,
        environment_sampler: Callable[[np.ndarray], float],
        candidate_generator: Optional[CandidateGenerator] = None,
        decision_logger: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[Tuple[np.ndarray, float, float]]]]:
        """
        Assign targets to robots using kriging believer approach.
        
        This is the main entry point for Step B. It simulates the full mission:
        - Robots start at their current positions
        - Each robot selects targets from its candidate set
        - Targets are added to believer GP before being reached
        - Actual GP is updated only when targets are physically reached
        - Simulation continues until time limit or all robots are idle
        
        Args:
            robots: List of robot agents
            candidate_sets: Candidate sets for each robot (from Step A)
            gp_belief: Initial GP belief state
            environment_sampler: Function to get measurement at position
            candidate_generator: Optional generator to rebuild candidates during mission
            decision_logger: Optional callback receiving state snapshots whenever
                a new target assignment decision is made. The callback receives a
                dictionary with GP copies, robot states, and selected targets.
            
        Returns:
            Tuple of:
                - Dict mapping robot_id to list of assigned target positions
                - Dict mapping robot_id to list of samples (position, value, timestamp)
        """
        # Initialize assignment state
        self._candidate_generator = candidate_generator
        self._samples_since_candidate_refresh = 0
        self._mcts_planner = None
        self._decision_logger = decision_logger
        self._initialize_assignment(robots, gp_belief, environment_sampler)
        
        # Assign initial targets to all robots
        for robot_id in self.robot_states.keys():
            self._assign_next_target(robot_id, candidate_sets[robot_id])
        
        # Run event-driven simulation
        self._run_simulation(candidate_sets, environment_sampler)
        
        # Extract results
        assignments = {
            robot_id: state.assigned_targets 
            for robot_id, state in self.robot_states.items()
        }
        
        samples = {
            robot_id: state.samples_collected
            for robot_id, state in self.robot_states.items()
        }
        
        return assignments, samples
    
    def _initialize_assignment(
        self,
        robots: List[Robot],
        gp_belief: GaussianProcessBelief,
        environment_sampler: Callable[[np.ndarray], float]
    ) -> None:
        """Initialize assignment state for all robots."""
        self.simulation_clock = 0.0
        self.event_queue = []
        self.all_target_points = []
        self.global_samples = []
        
        # Create copies of GP for actual and believer
        self.gp_actual = deepcopy(gp_belief)
        self.gp_believer = deepcopy(gp_belief)
        
        # Initialize robot states and take initial samples
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"INITIALIZATION - ROBOTS TAKING INITIAL SAMPLES")
            print(f"{'='*70}")
        
        for robot in robots:
            # Create assignment state
            state = RobotAssignmentState(robot=robot)
            self.robot_states[robot.id] = state
            
            # Take initial sample at starting position
            initial_value = environment_sampler(robot.position)
            initial_time = self.simulation_clock
            
            # Update robot state
            state.samples_collected.append((robot.position.copy(), initial_value, initial_time))
            robot.consume_budget(self.sensor_time)
            
            # Add to global samples
            self.global_samples.append((robot.position.copy(), initial_value, initial_time, robot.id))
            
            if self.verbose:
                print(f"\n  Robot {robot.id}:")
                print(f"    Starting Position: {robot.position}")
                print(f"    Initial Measurement: {initial_value:.3f}")
                print(f"    Sensor Time: {self.sensor_time:.1f}s")
                print(f"    Budget Remaining: {robot.remaining_budget:.1f}s")
            
        # Update actual GP with initial samples
        if self.global_samples:
            positions = np.array([s[0] for s in self.global_samples])
            values = np.array([s[1] for s in self.global_samples])
            self.gp_actual.update(positions, values)
            self.gp_believer = deepcopy(self.gp_actual)
            
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STARTING KRIGING BELIEVER ASSIGNMENT")
            print(f"{'='*70}")
            print(f"Time Limit:     {self.time_limit:.1f}s ({self.time_limit/60:.1f} minutes)")
            print(f"Min Threshold:  {self.min_time_threshold:.1f}s")
            print(f"Sensor Time:    {self.sensor_time:.1f}s")
            print(f"Initial Samples: {len(self.global_samples)}")
            print(f"\nNow assigning initial targets to all robots...")
    
    def _assign_next_target(
        self,
        robot_id: int,
        candidate_set: CandidateSet
    ) -> bool:
        """
        Assign next target to a robot from its candidate set.
        
        Returns:
            True if target was assigned, False otherwise
        """
        state = self.robot_states[robot_id]
        robot = state.robot
        
        # Check if robot has sufficient budget
        if robot.remaining_budget <= self.min_time_threshold:
            if self.verbose:
                print(f"\n  [ROBOT {robot_id}] Cannot assign new target - Insufficient budget")
                print(f"    Budget Remaining: {robot.remaining_budget:.1f}s (need >{self.min_time_threshold:.1f}s)")
                print(f"    Robot is DONE exploring")
            return False
        
        # Get feasible candidates
        feasible = candidate_set.get_feasible_points()
        if len(feasible) == 0:
            if self.verbose:
                print(f"\n  [ROBOT {robot_id}] Cannot assign new target - No feasible candidates")
                print(f"    Robot is DONE exploring")
            return False
        
        # Remove already-targeted points
        targeted_positions = {tuple(pos) for pos, _ in self.all_target_points}
        available = np.array([
            p for p in feasible 
            if tuple(p) not in targeted_positions
        ])
        
        if len(available) == 0:
            if self.verbose:
                print(f"\n  [ROBOT {robot_id}] Cannot assign new target - All candidates already targeted")
                print(f"    Feasible: {len(feasible)}, Already Targeted: {len(feasible)}")
                print(f"    Robot is DONE exploring")
            return False

        available = self._filter_budget_viable(available, robot)
        if len(available) == 0:
            if self.verbose:
                print(f"\n  [ROBOT {robot_id}] Cannot assign new target - No budget-viable candidates")
            return False
        
        # Select best target using configured strategy (MCTS preferred)
        target = self._select_target(available, robot)
        
        # Calculate distance in coordinate units
        distance_coords = np.linalg.norm(target - robot.position)
        
        # Convert to physical meters using environment
        if robot.environment is not None:
            distance_meters = robot.environment.coord_to_meters(distance_coords)
        else:
            # Fallback: assume coordinates are in meters
            distance_meters = distance_coords
        
        # Calculate travel time using PHYSICAL distance
        travel_time = distance_meters / robot.max_speed
        
        # Assign target to robot
        state.current_target = target.copy()
        state.time_to_target = travel_time
        state.assigned_targets.append(target.copy())
        
        # Add to global target list
        self.all_target_points.append((target.copy(), robot_id))
        
        # Schedule event for when robot reaches target
        event_time = self.simulation_clock + travel_time
        heapq.heappush(
            self.event_queue,
            AssignmentEvent(event_time, robot_id, 'reach_target', target.copy())
        )
        
        # Update kriging believer GP (pretend target is already sampled)
        self._update_kriging_believer()
        
        if self.verbose:
            print(f"\n  [Time {self.simulation_clock:.1f}s] ROBOT {robot_id} - NEW TARGET ASSIGNED")
            print(f"    Current Position:     {robot.position}")
            print(f"    Target Position:      {target}")
            print(f"    Distance (coords):    {distance_coords:.2f} units")
            print(f"    Distance (physical):  {distance_meters:.2f}m")
            print(f"    Robot Speed:          {robot.max_speed:.1f} m/s")
            print(f"    Travel Time:          {travel_time:.1f}s")
            print(f"    Arrival Time:         {event_time:.1f}s")
            print(f"    Budget Remaining:     {robot.remaining_budget:.1f}s")
            print(f"    Total Targets:        {len(state.assigned_targets)}")
        self._log_assignment_decision(
            robot_id=robot_id,
            selected_target=target,
            available_candidates=available
        )
        
        return True

    def _log_assignment_decision(
        self,
        robot_id: int,
        selected_target: np.ndarray,
        available_candidates: np.ndarray
    ) -> None:
        """Emit structured snapshot for downstream dataset collection."""
        if self._decision_logger is None:
            return
        robot_state_snapshot: Dict[int, Dict[str, Any]] = {}
        for rid, state in self.robot_states.items():
            robot_state_snapshot[rid] = {
                'position': state.robot.position.copy(),
                'remaining_budget': state.robot.remaining_budget,
                'current_target': None if state.current_target is None else state.current_target.copy(),
                'assigned_targets': [target.copy() for target in state.assigned_targets],
            }
        snapshot = {
            'time': self.simulation_clock,
            'robot_id': robot_id,
            'selected_target': selected_target.copy(),
            'available_candidates': available_candidates.copy(),
            'robot_states': robot_state_snapshot,
            'global_targets': [
                {'position': pos.copy(), 'robot_id': rid}
                for pos, rid in self.all_target_points
            ],
            'global_samples': [
                {'position': sample[0].copy(), 'value': sample[1], 'time': sample[2], 'robot_id': sample[3]}
                for sample in self.global_samples
            ],
            'gp_believer': self.gp_believer.copy() if self.gp_believer is not None else None,
            'gp_actual': self.gp_actual.copy() if self.gp_actual is not None else None,
        }
        self._decision_logger(snapshot)
    
    def _default_acquisition(
        self,
        candidates: np.ndarray,
        current_position: np.ndarray
    ) -> np.ndarray:
        """
        Default acquisition: select candidate with highest variance/distance ratio.
        
        This balances exploration (high variance) with efficiency (low travel distance).
        Uses PHYSICAL distances for fair comparison across different coordinate systems.
        """
        # Get GP variance predictions
        _, variances = self.gp_believer.predict(candidates, return_std=True)
        variances = variances ** 2  # Convert std to variance
        
        # Calculate distances in coordinate units
        distances_coords = np.linalg.norm(candidates - current_position, axis=1)
        
        # Convert to physical meters
        if self.environment is not None:
            distances_meters = np.array([
                self.environment.coord_to_meters(d) for d in distances_coords
            ])
        else:
            # Fallback: assume coordinates are in meters
            distances_meters = distances_coords
        
        # Avoid division by zero
        distances_meters = np.maximum(distances_meters, 1e-6)
        
        # Score: variance per meter traveled
        scores = variances / distances_meters
        best_idx = np.argmax(scores)
        
        return candidates[best_idx]
    
    def _select_target(self, candidates: np.ndarray, robot: Robot) -> np.ndarray:
        """Select next target using MCTS (if enabled) with graceful fallback."""
        if self.use_mcts_acquisition:
            mcts_target = self._select_target_with_mcts(candidates, robot)
            if mcts_target is not None:
                return mcts_target
        
        if self.acquisition_function is not None:
            return self.acquisition_function(
                candidates,
                robot.position,
                self.gp_believer,
                robot.remaining_budget
            )
        
        return self._default_acquisition(candidates, robot.position)
    
    def _select_target_with_mcts(self, candidates: np.ndarray, robot: Robot) -> Optional[np.ndarray]:
        """Run a lightweight MCTS planner to pick the next best candidate."""
        if len(candidates) == 0:
            return None
        subset = self._prepare_mcts_candidates(candidates, robot.position)
        if subset.size == 0:
            return None
        feasible_mask = np.ones(len(subset), dtype=bool)
        candidate_set = CandidateSet(
            robot_id=robot.id,
            points=subset,
            feasible=feasible_mask
        )
        planner = self._get_mcts_planner()
        path = planner.plan(
            robot=robot,
            candidates=candidate_set,
            gp_belief=self.gp_believer,
            sensor_time=self.sensor_time,
            environment=self.environment
        )
        if len(path) > 0:
            return path[0]
        return subset[0]
    
    def _prepare_mcts_candidates(
        self,
        candidates: np.ndarray,
        current_position: np.ndarray
    ) -> np.ndarray:
        """Trim candidate list so MCTS search stays focused."""
        if len(candidates) <= self.mcts_candidate_limit:
            return candidates
        
        _, std = self.gp_believer.predict(candidates, return_std=True)
        variances = std ** 2
        distances = np.linalg.norm(candidates - current_position, axis=1)
        if self.environment is not None:
            distances = np.array([
                self.environment.coord_to_meters(d) for d in distances
            ])
        distances = np.maximum(distances, 1e-3)
        scores = variances / distances
        nearest_k = max(1, self.mcts_candidate_limit // 2)
        best_by_distance = np.argsort(distances)[:nearest_k]
        best_by_score = np.argsort(scores)[-self.mcts_candidate_limit:]
        combined_indices = []
        seen = set()
        for idx in list(best_by_distance) + list(reversed(best_by_score)):
            if idx in seen:
                continue
            combined_indices.append(idx)
            seen.add(idx)
            if len(combined_indices) >= self.mcts_candidate_limit:
                break
        return candidates[combined_indices]

    def _filter_budget_viable(self, candidates: np.ndarray, robot: Robot) -> np.ndarray:
        """Remove candidates that cannot be reached with remaining budget margin."""
        if len(candidates) == 0:
            return candidates
        budget_margin = robot.remaining_budget - self.min_time_threshold
        if budget_margin <= self.sensor_time:
            return candidates
        viable: List[np.ndarray] = []
        for point in candidates:
            distance = np.linalg.norm(point - robot.position)
            if self.environment is not None:
                distance = self.environment.coord_to_meters(distance)
            travel_time = distance / max(robot.max_speed, 1e-6)
            total_time = travel_time + self.sensor_time
            if total_time <= budget_margin:
                viable.append(point)
        if viable:
            return np.array(viable)
        return candidates
    
    def _get_mcts_planner(self) -> MCTSPlanner:
        """Lazily instantiate the MCTS planner."""
        if self._mcts_planner is None:
            self._mcts_planner = MCTSPlanner(self.mcts_config)
        return self._mcts_planner
    
    def _update_kriging_believer(self) -> None:
        """Update kriging believer GP with all targeted (but not yet reached) points."""
        if not self.all_target_points:
            return
        
        # Get positions of all targets
        target_positions = np.array([pos for pos, _ in self.all_target_points])
        
        # Predict what values we expect at these targets using actual GP
        predicted_values, _ = self.gp_actual.predict(target_positions)
        
        # Create believer GP that includes these "pretend" observations
        self.gp_believer = deepcopy(self.gp_actual)
        self.gp_believer.update(target_positions, predicted_values)
    
    def _run_simulation(
        self,
        candidate_sets: Dict[int, CandidateSet],
        environment_sampler: Callable[[np.ndarray], float]
    ) -> None:
        """Run event-driven simulation until time limit or no more events."""
        while self.event_queue and self.simulation_clock < self.time_limit:
            # Get next event
            event = heapq.heappop(self.event_queue)
            
            # Check if event exceeds time limit
            if event.time > self.time_limit:
                if self.verbose:
                    print(f"\nTime limit reached at {self.simulation_clock:.1f}s")
                break
            
            # Update simulation clock
            self.simulation_clock = event.time
            
            # Process event
            if event.event_type == 'reach_target':
                self._process_reach_target(event, candidate_sets, environment_sampler)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ASSIGNMENT COMPLETE")
            print(f"{'='*60}")
            print(f"Final time: {self.simulation_clock:.1f}s")
            print(f"Total samples: {len(self.global_samples)}")
            for robot_id, state in self.robot_states.items():
                print(f"  Robot {robot_id}: {len(state.samples_collected)} samples, "
                      f"{len(state.assigned_targets)} targets assigned")
    
    def _process_reach_target(
        self,
        event: AssignmentEvent,
        candidate_sets: Dict[int, CandidateSet],
        environment_sampler: Callable[[np.ndarray], float]
    ) -> None:
        """Process robot reaching its target."""
        state = self.robot_states[event.robot_id]
        robot = state.robot
        
        # Calculate distance traveled (for logging)
        if len(robot.trajectory) >= 2:
            distance_coords = np.linalg.norm(event.position - robot.trajectory[-2].position)
            if robot.environment is not None:
                distance_meters = robot.environment.coord_to_meters(distance_coords)
            else:
                distance_meters = distance_coords
        else:
            distance_coords = 0.0
            distance_meters = 0.0
        
        # Move robot to target
        travel_time = state.time_to_target
        robot.consume_budget(travel_time)
        robot.state.position = event.position.copy()
        robot.state.timestamp = event.time
        
        # Take measurement
        measurement_value = environment_sampler(event.position)
        robot.consume_budget(self.sensor_time)
        
        # Update state
        state.samples_collected.append((event.position.copy(), measurement_value, event.time))
        state.current_target = None
        state.time_to_target = None
        
        # Add to global samples
        self.global_samples.append((event.position.copy(), measurement_value, event.time, robot.id))
        
        # Update actual GP with new sample
        positions = np.array([s[0] for s in self.global_samples])
        values = np.array([s[1] for s in self.global_samples])
        self.gp_actual.update(positions, values)
        self._handle_post_sample_updates(candidate_sets)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Time {event.time:.1f}s] ROBOT {robot.id} - ARRIVED AT TARGET")
            print(f"{'='*70}")
            print(f"  Position:             {event.position}")
            print(f"  Measurement:          {measurement_value:.3f}")
            print(f"  Distance Traveled:    {distance_meters:.2f}m")
            print(f"  Travel Time:          {travel_time:.1f}s")
            print(f"  Sensor Time:          {self.sensor_time:.1f}s")
            print(f"  Budget Used:          {travel_time + self.sensor_time:.1f}s")
            print(f"  Budget Remaining:     {robot.remaining_budget:.1f}s")
            print(f"  Samples So Far:       {len(state.samples_collected)}")
            print(f"  Total Samples:        {len(self.global_samples)}")
        
        # Assign next target if robot still has budget
        self._assign_next_target(robot.id, candidate_sets[robot.id])
    
    def _handle_post_sample_updates(self, candidate_sets: Dict[int, CandidateSet]) -> None:
        """Handle bookkeeping after a real sample is incorporated."""
        if self.candidate_refresh_interval is None or self._candidate_generator is None:
            return
        
        self._samples_since_candidate_refresh += 1
        if self._samples_since_candidate_refresh < self.candidate_refresh_interval:
            return
        
        self._samples_since_candidate_refresh = 0
        self._refresh_candidate_sets(candidate_sets)
    
    def _refresh_candidate_sets(self, candidate_sets: Dict[int, CandidateSet]) -> None:
        """Regenerate candidate sets using the latest GP belief."""
        if self._candidate_generator is None:
            return
        robots = [state.robot for state in self.robot_states.values()]
        regenerated = self._candidate_generator.generate_candidates(
            gp=self.gp_actual,
            robots=robots,
            budget_reserve=self.candidate_budget_reserve
        )
        updated = 0
        for robot_id in self.robot_states.keys():
            if robot_id in regenerated:
                candidate_sets[robot_id] = regenerated[robot_id]
                updated += 1
        if self.verbose:
            print(f"\n  [REFRESH] Candidate sets regenerated for {updated} robots")
    
    def get_final_gp(self) -> GaussianProcessBelief:
        """Get the final GP belief after all assignments."""
        return self.gp_actual
    
    def get_statistics(self) -> Dict:
        """Get assignment statistics."""
        return {
            'total_time': self.simulation_clock,
            'total_samples': len(self.global_samples),
            'robot_stats': {
                robot_id: {
                    'samples_collected': len(state.samples_collected),
                    'targets_assigned': len(state.assigned_targets),
                    'final_budget': state.robot.remaining_budget,
                    'budget_used': state.robot.initial_budget - state.robot.remaining_budget
                }
                for robot_id, state in self.robot_states.items()
            }
        }
