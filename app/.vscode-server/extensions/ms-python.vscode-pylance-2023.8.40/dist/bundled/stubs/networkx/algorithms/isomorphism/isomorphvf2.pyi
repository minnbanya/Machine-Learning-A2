# This work was originally coded by Christopher Ellison
# as part of the Computational Mechanics Python (CMPy) project.
# James P. Crutchfield, principal investigator.
# Complexity Sciences Center and Physics Department, UC Davis.

import sys
from ...classes.graph import Graph

__all__ = ["GraphMatcher", "DiGraphMatcher"]

class GraphMatcher:
    def __init__(self, G1: Graph, G2: Graph): ...
    def reset_recursion_limit(self): ...
    def candidate_pairs_iter(self): ...
    def initialize(self): ...
    def is_isomorphic(self): ...
    def isomorphisms_iter(self): ...
    def match(self): ...
    def semantic_feasibility(self, G1_node, G2_node): ...
    def subgraph_is_isomorphic(self): ...
    def subgraph_is_monomorphic(self): ...

    #    subgraph_is_isomorphic.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def subgraph_isomorphisms_iter(self): ...
    def subgraph_monomorphisms_iter(self): ...

    #    subgraph_isomorphisms_iter.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def syntactic_feasibility(self, G1_node, G2_node): ...

class DiGraphMatcher(GraphMatcher):
    def __init__(self, G1: Graph, G2: Graph): ...
    def candidate_pairs_iter(self): ...
    def initialize(self): ...
    def syntactic_feasibility(self, G1_node, G2_node): ...

class GMState:
    def __init__(self, GM, G1_node=None, G2_node=None): ...
    def restore(self): ...

class DiGMState:
    def __init__(self, GM, G1_node=None, G2_node=None): ...
    def restore(self): ...
