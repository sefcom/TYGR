from typing import *

from pyvex.block import IRSB

class FGNode:
  def __init__(self):
    pass

class FGNodeRegister(FGNode):
  def __init__(self, reg_id: int):
    self.reg_id = reg_id

class FGNodeTmpVar(FGNode):
  def __init__(self, tmp_var: int):
    self.tmp_var = tmp_var

class FGEdge:
  def __init__(self):
    pass

class FGEdgeGet(FGEdge):
  pass

class FGEdgePut(FGEdge):
  pass

class FGEdgeIdentity(FGEdge):
  pass

class FGEdgeLoad(FGEdge):
  pass

class FGEdgeStore(FGEdge):
  pass

class FGEdgeIOP(FGEdge):
  pass

class FlowGraph:
  def __init__(self):
    pass

class JoinedFlowGraph:
  def __init__(self):
    pass

def generate_tmp_var_flow_graph(irsb: IRSB, tmp_var: int) -> FlowGraph:
  pass

def join_tmp_var_flow_graphs(graphs: List[FlowGraph]) -> JoinedFlowGraph:
  pass
