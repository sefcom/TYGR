from typing import *

from torch import Tensor

from ...analysis.types import Type
from ...analysis.dwarf import LocationList
from ...analysis.angr.ast_graph import AstGraph

class PreprocGlowTHInput:
  def __init__(
    self,
    node_labels: Tensor,
    edge_labels: Tensor,
    edges: Tensor,
    var_nodes: List[List[int]],
  ):
    self.node_labels = node_labels
    self.edge_labels = edge_labels
    self.edges = edges
    self.var_nodes = var_nodes

  def print(self):
    print("Node labels")
    print(self.node_labels)
    print("Edge labels")
    print(self.edge_labels)
    print("Edges")
    print(self.edges)
    print("Variable nodes")
    print(self.var_nodes)
