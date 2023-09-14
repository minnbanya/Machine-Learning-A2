from ..classes.digraph import DiGraph
from numpy.typing import ArrayLike

import heapq
from collections import deque
from functools import partial
from itertools import chain, product, starmap
from math import gcd

from ..classes.graph import Graph
from ..utils import arbitrary_element, not_implemented_for, pairwise

__all__ = [
    "descendants",
    "ancestors",
    "topological_sort",
    "lexicographical_topological_sort",
    "all_topological_sorts",
    "topological_generations",
    "is_directed_acyclic_graph",
    "is_aperiodic",
    "transitive_closure",
    "transitive_closure_dag",
    "transitive_reduction",
    "antichains",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_to_branching",
]

chaini = ...

def descendants(G: Graph, source): ...
def ancestors(G: Graph, source): ...
def has_cycle(G: Graph): ...
def is_directed_acyclic_graph(G: Graph) -> bool: ...
def topological_generations(G: Graph): ...
def topological_sort(G: Graph): ...
def lexicographical_topological_sort(G: Graph, key=None): ...
def all_topological_sorts(G: Graph): ...
def is_aperiodic(G: Graph) -> bool: ...
def transitive_closure(G: Graph, reflexive=False): ...
def transitive_closure_dag(G: Graph, topo_order: ArrayLike | tuple | None = None): ...
def transitive_reduction(G: Graph): ...
def antichains(G: Graph, topo_order: ArrayLike | tuple | None = None): ...
def dag_longest_path(
    G: Graph,
    weight: str = "weight",
    default_weight: int = 1,
    topo_order: ArrayLike | tuple | None = None,
) -> ArrayLike: ...
def dag_longest_path_length(
    G: Graph, weight: str = "weight", default_weight: int = 1
) -> int: ...
def root_to_leaf_paths(G: Graph): ...
def dag_to_branching(G: Graph) -> DiGraph: ...
