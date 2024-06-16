from typing import *

from .common import FlowInput
from .flow_graph import generate_tmp_var_flow_graph
from ...analysis.types import Type
from ...analysis.dwarf import get_dwarf_info, dwarf_info_to_vars
from ...analysis.vex import get_vex_irsbs, find_tmp_var_within_range

class Stats:
  def __init__(self):
    self.num_dwarf_vars = 0
    self.num_tmp_vars = 0

  def print(self):
    print(f"#Dwarf Vars: {self.num_dwarf_vars}")
    print(f"#Tmp Vars: {self.num_tmp_vars}")

class Options:
  def __init__(self, verbose=False):
    self.verbose = verbose

def generate_flow_dataset(input_file_name: str, options: Options) -> Iterator[Tuple[FlowInput, Type]]:
  stats = Stats()
  dwarf_info = get_dwarf_info(input_file_name)
  irsbs = get_vex_irsbs(input_file_name)
  for dwarf_var in dwarf_info_to_vars(dwarf_info):
    # First extract dwarf variable information
    stats.num_dwarf_vars += 1
    if options.verbose:
      print("Dwarf Var", dwarf_var)
    directory, file_name, func_name, low_high_pc, var_name, var_loc, var_type = dwarf_var

    # Then find the corresponding variable information in VexIR
    var_flows = []
    tmp_var_expr_cache = {}
    for irsb, tmp_var in find_tmp_var_within_range(irsbs, tmp_var_expr_cache, low_high_pc, var_loc):
      stats.num_tmp_vars += 1
      if options.verbose:
        print(f"Tmp Var t{tmp_var} Found")
      flow_graph = generate_tmp_var_flow_graph(irsb, tmp_var)
      var_flows.append(flow_graph)

    # Combine them to generate datapoint
    flow_input = FlowInput(directory, file_name, func_name, low_high_pc, var_name, var_loc, var_flows)
    yield (flow_input, var_type)

  if options.verbose:
    stats.print()
