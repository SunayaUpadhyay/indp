"""
Baseline planners for multi-robot informative path planning.

Provides simple baseline algorithms that serve as:
- Sanity checks (random)
- Non-informative coverage baselines (lawnmower)
- Information-driven baselines (greedy IG with/without coordination)
- Auction-based coordination
- Performance lower bounds for comparison
"""

from .base_planner import BaseMultiRobotPlanner
from .random_planner import RandomMultiRobotPlanner
from .lawnmower_planner import LawnmowerPlanner
from .sequential_greedy_planner import SequentialGreedyIGPlanner
from .independent_greedy_planner import IndependentGreedyIGPlanner
from .auction_planner import AuctionVariancePlanner

__all__ = [
    'BaseMultiRobotPlanner',
    'RandomMultiRobotPlanner',
    'LawnmowerPlanner',
    'SequentialGreedyIGPlanner',
    'IndependentGreedyIGPlanner',
    'AuctionVariancePlanner',
]
