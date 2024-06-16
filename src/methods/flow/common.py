from typing import *

from ...analysis.types import *
from ...analysis.dwarf import VarLocation
from .flow_graph import FlowGraph

class FlowInput:
  def __init__(
    self,
    directory: Optional[str],
    file_name: Optional[str],
    func_name: Optional[str],
    low_high_pc: Tuple[int, int],
    var_name: Optional[str],
    var_loc: VarLocation,
    flow_graphs: List[FlowGraph],
  ):
    self.directory = directory
    self.file_name = file_name
    self.function_name = func_name
    self.variable_name = var_name
    self.low_high_pc = low_high_pc
    self.variable_location = var_loc
    self.flow_graphs = flow_graphs

class PreprocFlowInput:
  def __init__(self):
    self.nodes = None
    self.edges = None
    self.edge_indices = None
