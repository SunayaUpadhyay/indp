"""
Target assignment using Kriging Believer approach.

Step B: Assignment - Decentralized target selection where robots independently
select targets from their candidate sets while avoiding conflicts through
kriging believer coordination.
"""

from .kriging_believer import KrigingBelieverAssignment

__all__ = ['KrigingBelieverAssignment']
