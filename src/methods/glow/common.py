from typing import *

from torch import Tensor

from ...analysis.types import Type
from ...analysis.dwarf import LocationList
from ...analysis.angr.ast_graph import AstGraph

class GlowVar:
  def __init__(
    self,
    name: Optional[str],
    locs: LocationList,
    nodes: List[int],
  ):
    self.name = name
    self.locs = locs
    self.nodes = nodes

class GlowInput:
  def __init__(
    self,
    input_file_name: str,
    directory: Optional[str],
    file_name: Optional[str],
    function_name: Optional[str],
    low_high_pc: Tuple[int, int],
    ast_graph: AstGraph,
    vars: List[GlowVar],
    arch: str,
  ):
    self.input_file_name = input_file_name
    self.directory = directory
    self.file_name = file_name
    self.function_name = function_name
    self.low_high_pc = low_high_pc
    self.ast_graph = ast_graph
    self.vars = vars
    self.arch = arch

class GlowOutput:
  def __init__(
    self,
    types: List[Type],
  ):
    self.types = types

class PreprocGlowInput:
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
