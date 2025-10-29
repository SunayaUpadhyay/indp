"""
Candidate generation module for IPP.

This module implements Step A of the algorithm:
Adaptive generation of candidate sampling points using quadtree refinement
based on Gaussian Process uncertainty.
"""

from .quadtree import QuadTree, QuadTreeNode
from .candidate_generator import CandidateGenerator, CandidateSet

__all__ = ['QuadTree', 'QuadTreeNode', 'CandidateGenerator', 'CandidateSet']
