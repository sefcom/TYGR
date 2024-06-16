from typing import *
import json
import datetime
import traceback

import angr

from archinfo.arch_amd64 import ArchAMD64
from archinfo.arch_x86 import ArchX86
from archinfo.arch_aarch64 import ArchAArch64
from archinfo.arch_arm import ArchARM
from archinfo.arch_mips32 import ArchMIPS32

from multiprocessing import Pool, Lock

from sympy import fu

from .common import GlowInput, GlowVar, GlowOutput
from ...analysis.angr.sim_exec import SimExecResult, \
                                      SimExecStrategy, \
                                      SimpleDominatorStrategy, \
                                      LessSimpleDominatorStrategy
from ...analysis.angr.ast_graph import AstGraph, NodeLabel
from ...analysis.dwarf.location import *
from ...analysis.dwarf import get_dwarf_info, \
                              dwarf_info_to_subprograms, \
                              dwarf_info_to_context, \
                              dwarf_subprogram_to_vars, \
                              DwarfVariable2
import networkx as nx
import pickle
import os

class Stats:
  def __init__(self):
    self.num_functions = 0
    self.num_dwarf_vars = 0

  def print(self):
    print(f"#Functions: {self.num_functions}")
    print(f"#Dwarf Vars: {self.num_dwarf_vars}")

class Options:
  def __init__(
    self,
    verbose=False,
    parallel=True,
    no_splice=False,
    ignore_functions=True,
    ignore_functions_file="data/ignore_functions.json",
    output_functions=None,
    predict_phase=False,
    pool_size=10, # Between, 16, 32, 64, this seems to be a "sweet spot"
    skip_function_with_no_vars=True,
    skip_vars_with_no_nodes=True,
  ):
    self.verbose = verbose
    self.parallel = parallel
    self.no_splice = no_splice
    self.ignore_functions = ignore_functions
    self.ignore_functions_file = ignore_functions_file
    self.output_functions = output_functions
    self.pool_size = pool_size
    self.predict_phase = predict_phase
    self.skip_function_with_no_vars = skip_function_with_no_vars
    self.skip_vars_with_no_nodes = skip_vars_with_no_nodes

    self.explored_functions = []

  def add_explored_function(self, directory, file_name, function_name):
    if self.output_functions is not None:
      self.explored_functions.append([directory, file_name, function_name])

  def save_explored_function(self):
    if self.output_functions is not None:
      json.dump(self.explored_functions, open(self.output_functions, "w"))

LOAD_OPTIONS = {
  "main_opts": {
    "base_addr": 0x0
  },
  "auto_load_libs": False
}

DwarfVars = List[DwarfVariable2]
func_name = "XX"
Task = Tuple[
  Tuple[int, int], # Low High PC of the subprogram
  DwarfVars,
  SimExecStrategy,
  Options,
]

# in_file = "XX"
# out_dir = "XX"
# dir_name = "XX"
# file_name = "XX"
# Task = Tuple[
#   in_file, 
#   out_dir, 
#   dir_name, 
#   file_name, 
#   Tuple[int, int], # Low High PC of the subprogram
#   DwarfVars,
#   SimExecStrategy,
#   Options,
#   func_name
# ]


def ignore_functions(options: Options) -> Set[Union[str, Tuple[str, str, str]]]:
  if options.ignore_functions:
    ign_fns = json.load(open(options.ignore_functions_file, "r"))
    proc_ign_fns = []
    for ign_fn in ign_fns:
      if isinstance(ign_fn, str):
        proc_ign_fns.append(ign_fn)
      else:
        proc_ign_fns.append((ign_fn[0], ign_fn[1], ign_fn[2]))
    return proc_ign_fns
  else:
    return []

def addr_to_cfa_offset(addr, exec_res : SimExecResult):
  arch = exec_res.proj.arch
  if isinstance(arch, ArchAMD64) or isinstance(arch, ArchX86):
    # cfa_off = exec_res.config["init_sp"] + loc.arg + arch.bytes
    cfa_off = addr - exec_res.config["init_sp"] - arch.bytes
  elif isinstance(arch, ArchAArch64) or isinstance(arch, ArchARM) or isinstance(arch, ArchMIPS32):
    cfa_off = exec_res.config["init_sp"] - addr
    # cfa_off = exec_res.config["init_sp"] + loc.arg
  # Unsupported
  else:
    print(f"Unsupported arch {arch}")
    return None

  if abs(cfa_off) < 0x10000:
    return cfa_off
  else:
    return None

# Translate these into something that can be processed by the AST graph,
# which we don't to have to import dwarf features
def prepare_var_location_list(var_locs : LocationList, exec_res : SimExecResult):
  if not isinstance(var_locs, list):
    return None

  arch = exec_res.proj.arch
  locs = []
  for (low_addr, high_addr, loc) in var_locs:
    range_pair = (low_addr, high_addr)
    if isinstance(loc, CfaLocation):
      if isinstance(arch, ArchAMD64) or isinstance(arch, ArchX86):
        addr = exec_res.config["init_sp"] + loc.arg + arch.bytes
      elif isinstance(arch, ArchAArch64) or isinstance(arch, ArchARM) or isinstance(arch, ArchMIPS32):
        addr = exec_res.config["init_sp"] + loc.arg
      # Unsupported
      else:
        print(f"Unsupported arch {arch}")
        return None

      locs.append((range_pair, ("addr", addr)))

    elif isinstance(loc, RegLocation):
      reg_info = arch.registers.get(loc.reg_name)
      if isinstance(reg_info, tuple) and len(reg_info) == 2:
        reg_offset = reg_info[0]
        locs.append((range_pair, ("reg", reg_offset)))

    elif isinstance(loc, AddrLocation):
      locs.append((range_pair, ("addr", loc.arg)))

    # breg means "the value in this address", which might be challenging to compute statically
    elif isinstance(loc, BregLocation):
      pass

  return locs

def sym_exec_one(task) -> Optional[Tuple[AstGraph, List[GlowVar], List[Type]]]:
  now = datetime.datetime.now().timestamp()
  (low_pc, dwarf_vars, strategy, options)= task
  num_var_count = len(dwarf_vars)

  if options.verbose:
    print(f"\t{now} | attempting to run function at {hex(low_pc)}")

  # Then symexec the function to obtain the ast graph
  try:
    sym_exec_result = strategy.sim_exec_function(low_pc)
  except Exception as err:
    if options.verbose:
      print(f"\tError when doing symbolic execution: {err}")
      traceback.print_exc()
    return None
  ast_graph = AstGraph()

  # get function parameter ast value
  ## if an argument register has been written to, we can no longer assign the function parameter name to it
  func_param_locs = []
  for dwarf_var in dwarf_vars:
    (var_name, var_locs, var_type, func_param) = dwarf_var 
    if func_param:
      for i, (_, regs) in enumerate(prepare_var_location_list(var_locs, sym_exec_result)):
        if regs[1] not in func_param_locs: func_param_locs.append(regs[1])

  for i, (_, _, state) in enumerate(sym_exec_result.tups):
    if options.verbose:
      print(f"\t[{i+1} / {len(sym_exec_result.tups)}]: ast graph processing {state}")
      
    ast_graph.process_state(state, func_param_locs)
    
    
  # Then iterate through the dwarf variables
  all_var_nodes = set()
  vars = []
  tys = []

  if options.predict_phase:
    ## seperate read/write
    print("Predict_phase")
    print("This part need to be implemented")
    for node in ast_graph.graph.nodes:
      nlabel = ast_graph.node_to_label.get(node)
      if isinstance(nlabel, NodeLabel):
        for pred in ast_graph.graph.predecessors(node):
          if ast_graph.edge_exists(pred, node, "mem_loc"):
            plabel = ast_graph.node_to_label.get(pred)
            if isinstance(plabel, NodeLabel):
              cfa_off = addr_to_cfa_offset(plabel.ast_val, sym_exec_result)
              if cfa_off is not None:
                loc = CfaLocation(cfa_off)
                loc_list = [(nlabel.addr, nlabel.addr + 1, loc)] # singleton location for prediction
                glow_var = GlowVar(None, loc_list, [node])
                vars.append(glow_var)
                all_var_nodes.add(node)

          if ast_graph.edge_exists(pred, node, "reg_loc"):
            plabel = ast_graph.node_to_label.get(pred)
            if isinstance(plabel, NodeLabel):
              reg_num = plabel.ast_val
              loc = RegLocation(reg_num)
              loc_list= [(nlabel.addr, nlabel.addr + 1, loc)] # singleton location for prediction
              glow_var = GlowVar(None, loc_list, [node])
              all_var_nodes.add(node)

  else:
    for dwarf_var in dwarf_vars:
      (var_name, var_locs, var_type, func_param) = dwarf_var 
      prepd_locs = prepare_var_location_list(var_locs, sym_exec_result)
      nodes = ast_graph.find_nodes_from_locs(prepd_locs)
      all_var_nodes |= nodes
      # Push the data to the outputs
      vars.append(GlowVar(var_name, var_locs, nodes))
      tys.append(var_type)

  #Splice the ast_graph
  if not options.no_splice:
    # Splice the graph to obtain the new graph and the mapping
    (ast_graph, mapping) = ast_graph.directed_splice_and_normalize(all_var_nodes)

    # Updated the variable nodes by the mapping
    normed_vars = []
    for var in vars:
      normed_nodes = [mapping[n] for n in var.nodes]
      normed_vars.append(GlowVar(var.name, var.locs, normed_nodes))
      
    vars = normed_vars

  return (ast_graph, vars, tys) 

def sym_exec_one_return(task):
  now = datetime.datetime.now().timestamp()
  
  (input_file_name, out_file_dir, directory, file_name, low_high_pc, dwarf_vars, strategy, options, func_name) = task
  # (directory, file_name, func_name, low_high_pc, _, _) = subprogram
  
  num_var_count = len(dwarf_vars)

  low_pc = low_high_pc[0]
  if options.verbose:
    print(f"\t{now} | attempting to run function at {hex(low_pc)}")

  # Then symexec the function to obtain the ast graph
  try:
    sym_exec_result = strategy.sim_exec_function(low_pc)
  except Exception as err:
    if options.verbose:
      print(f"\tError when doing symbolic execution: {err}")
      traceback.print_exc()
    return None
  ast_graph = AstGraph()

  # get function parameter ast value
  ## if an argument register has been written to, we can no longer assign the function parameter name to it
  func_param_locs = []
  for dwarf_var in dwarf_vars:
    (var_name, var_locs, var_type, func_param) = dwarf_var 
    if func_param:
      for i, (_, regs) in enumerate(prepare_var_location_list(var_locs, sym_exec_result)):
        if regs[1] not in func_param_locs: func_param_locs.append(regs[1])

  for i, (_, _, state) in enumerate(sym_exec_result.tups):
    if options.verbose:
      print(f"\t[{i+1} / {len(sym_exec_result.tups)}]: ast graph processing {state}")
      
    ast_graph.process_state(state, func_param_locs)
    
    
  # Then iterate through the dwarf variables
  all_var_nodes = set()
  vars = []
  tys = []

  if options.predict_phase:
    ## seperate read/write
    print("Predict_phase")
    print("This part need to be implemented")
    for node in ast_graph.graph.nodes:
      nlabel = ast_graph.node_to_label.get(node)
      if isinstance(nlabel, NodeLabel):
        for pred in ast_graph.graph.predecessors(node):
          if ast_graph.edge_exists(pred, node, "mem_loc"):
            plabel = ast_graph.node_to_label.get(pred)
            if isinstance(plabel, NodeLabel):
              cfa_off = addr_to_cfa_offset(plabel.ast_val, sym_exec_result)
              if cfa_off is not None:
                loc = CfaLocation(cfa_off)
                loc_list = [(nlabel.addr, nlabel.addr + 1, loc)] # singleton location for prediction
                glow_var = GlowVar(None, loc_list, [node])
                vars.append(glow_var)
                all_var_nodes.add(node)

          if ast_graph.edge_exists(pred, node, "reg_loc"):
            plabel = ast_graph.node_to_label.get(pred)
            if isinstance(plabel, NodeLabel):
              reg_num = plabel.ast_val
              loc = RegLocation(reg_num)
              loc_list= [(nlabel.addr, nlabel.addr + 1, loc)] # singleton location for prediction
              glow_var = GlowVar(None, loc_list, [node])
              all_var_nodes.add(node)

  else:
    for dwarf_var in dwarf_vars:
      (var_name, var_locs, var_type, func_param) = dwarf_var 
      prepd_locs = prepare_var_location_list(var_locs, sym_exec_result)
      nodes = ast_graph.find_nodes_from_locs(prepd_locs)
      all_var_nodes |= nodes
      # Push the data to the outputs
      vars.append(GlowVar(var_name, var_locs, nodes))
      tys.append(var_type)

  #Splice the ast_graph
  if not options.no_splice:
    # Splice the graph to obtain the new graph and the mapping
    (ast_graph, mapping) = ast_graph.directed_splice_and_normalize(all_var_nodes)

    # Updated the variable nodes by the mapping
    normed_vars = []
    for var in vars:
      normed_nodes = [mapping[n] for n in var.nodes]
      normed_vars.append(GlowVar(var.name, var.locs, normed_nodes))
      
    vars = normed_vars
  # return (ast_graph, vars, tys) 

  ## ======================================== ##
  ## pick dump to final result
  if len(vars) > 0 or not options.skip_function_with_no_vars:
    if options.verbose and func_name:
      print(f"Dwarf Function {func_name}")
  else:
    print(f"[Info] Function {func_name} ({hex(low_pc)}) has no variable")
    return
  
  # Initialization
  stats = Stats()
  # Check if there are such nodes, proceed only if has node
  filtered_vars = []
  filtered_tys = []
  for (var, ty) in zip(vars, tys):

    if len(var.nodes) > 0 or not options.skip_vars_with_no_nodes:
      stats.num_dwarf_vars += 1
      filtered_vars.append(var)
      filtered_tys.append(ty)
      if options.verbose and func_name:
        print("|- Dwarf Variable", var.name)
    else:
      if options.verbose:
        print("|- [ERROR] No AST node found for variable", var.name)
      continue

  if len(filtered_vars) > 0 or not options.skip_function_with_no_vars:
    stats.num_functions += 1
  else:
    print("|- [Info] No variable in function", func_name)
    return

  options.add_explored_function(directory, file_name, func_name)
  # Construct the input and output
  glow_input = GlowInput(input_file_name, directory, file_name, func_name, low_high_pc, ast_graph, filtered_vars,"arch")
  glow_output = GlowOutput(filtered_tys)
  

  out_file_name = str(file_name) + "?" + str(func_name)
  out_file_path = os.path.join(out_file_dir, out_file_name)

  
  # one_function = filter_ill_formed((glow_input, glow_output))
  if one_function is None:
    return
  
  with open(out_file_path, "wb") as f:
    pickle.dump(one_function, f)
  return

def generate_glow_dataset_parallel(input_file_name: str, out_file_dir: str, options: Options) -> Iterator[Tuple[GlowInput, GlowOutput]]:
  import logging
  logging.getLogger('angr').setLevel('ERROR')
  ign_fns = ignore_functions(options)
  # Initialization
  stats = Stats()
  dwarf_info = get_dwarf_info(input_file_name)
  dwarf_ctx = dwarf_info_to_context(dwarf_info)
  proj = angr.Project(input_file_name, load_options=LOAD_OPTIONS)
  strat_config = None
  strategy = LessSimpleDominatorStrategy(proj, config=strat_config)

  # Iterate through all subprograms in the dwarf info
  tasks = []
  for subprogram in dwarf_info_to_subprograms(dwarf_info):
    # First find the functions
    (directory, file_name, func_name, low_high_pc, _, _) = subprogram
    if func_name in ign_fns or (directory, file_name, func_name) in ign_fns:
      if options.verbose:
        print("[Info] Skipping function", func_name)
      continue

    dwarf_vars = list(dwarf_subprogram_to_vars(subprogram, dwarf_ctx, dwarf_info))

    # First check if dwarf_vars are non-empty; if empty, directly jump to
    # the next function without doing symbolic execution  
    if len(dwarf_vars) == 0:
      continue
    
    # Add task
    # tasks.append((input_file_name, out_file_dir, directory, file_name, low_high_pc, dwarf_vars, strategy, options, func_name))
    tasks.append((subprogram, (low_high_pc[0], dwarf_vars, strategy, options)))
  print(options.pool_size)
  with Pool(options.pool_size) as pool:
    # pool.map(sym_exec_one_return, tasks) 
    sym_exec_result = pool.map(sym_exec_one, map(lambda t: t[1], tasks))    
    
  for ((subprogram, _), result) in zip(tasks, sym_exec_result):
    if result is None:
      continue

    (ast_graph, vars, tys) = result
    (directory, file_name, func_name, low_high_pc, _, _) = subprogram
    
    if len(vars) > 0 or not options.skip_function_with_no_vars:
      if options.verbose and func_name:
        print(f"Dwarf Function {func_name}")
    else:
      print(f"[Info] Function {func_name} ({hex(low_high_pc[0])}) has no variable")
      continue

    # Check if there are such nodes, proceed only if has node
    filtered_vars = []
    filtered_tys = []
    for (var, ty) in zip(vars, tys):

      if len(var.nodes) > 0 or not options.skip_vars_with_no_nodes:
        stats.num_dwarf_vars += 1
        filtered_vars.append(var)
        filtered_tys.append(ty)
        if options.verbose and func_name:
          print("|- Dwarf Variable", var.name)
      else:
        if options.verbose:
          print("|- [ERROR] No AST node found for variable", var.name)
        continue

    if len(filtered_vars) > 0 or not options.skip_function_with_no_vars:
      stats.num_functions += 1
    else:
      print("|- [Info] No variable in function", func_name)
      continue

    options.add_explored_function(directory, file_name, func_name)
    
    # Construct the input and output
    glow_input = GlowInput(input_file_name, directory, file_name, func_name, low_high_pc, ast_graph, filtered_vars,"arch")
    glow_output = GlowOutput(filtered_tys)
        
    yield (glow_input, glow_output)

  if options.verbose:
    stats.print()

  options.save_explored_function()


def generate_glow_dataset_no_parallel(input_file_name: str, out_file_dir: str, options: Options) -> Iterator[Tuple[GlowInput, GlowOutput]]:
  import logging
  logging.getLogger('angr').setLevel('ERROR')
  ign_fns = ignore_functions(options)

  # Initialization
  stats = Stats()
  dwarf_info = get_dwarf_info(input_file_name)
  dwarf_ctx = dwarf_info_to_context(dwarf_info)
  proj = angr.Project(input_file_name, load_options=LOAD_OPTIONS)
  strat_config = None
  strategy = LessSimpleDominatorStrategy(proj, config=strat_config)

  # Iterate through all subprograms in the dwarf info
  for subprogram in dwarf_info_to_subprograms(dwarf_info):
    # First find the functions
    (directory, file_name, func_name, low_high_pc, _, _) = subprogram
    if func_name in ign_fns or (directory, file_name, func_name) in ign_fns:
      if options.verbose:
        print("[Info] Skipping function", func_name)
      continue
    
    dwarf_vars = list(dwarf_subprogram_to_vars(subprogram, dwarf_ctx, dwarf_info))

    # First check if dwarf_vars are non-empty; if empty, directly jump to
    # the next function without doing symbolic execution
    if len(dwarf_vars) > 0 or not options.skip_function_with_no_vars:
      if options.verbose and func_name:
        print("Dwarf Function", func_name)
    else:
      print(f"[Info] Function {func_name} at {hex(low_high_pc[0])} has no variable")
      continue
    
    # Do symbolic execution, continue if fail
    try:
      task = (low_high_pc[0], dwarf_vars, strategy, options)
      sym_exec_result = sym_exec_one(task)
      # task = (input_file_name, out_file_dir, directory, file_name, low_high_pc, dwarf_vars, strategy, options, func_name)
      # sym_exec_one_return(task)  
    except Exception as exc:
      print(f"datagen: {exc}")
      continue

    if sym_exec_result is None:
      continue

    (ast_graph, vars, tys) = sym_exec_result
   
    # Get the variables that are valid
    filtered_vars = []
    filtered_tys = []
    for (var, ty) in zip(vars, tys):
      if len(var.nodes) > 0 or not options.skip_vars_with_no_nodes:
        stats.num_dwarf_vars += 1
        if options.verbose and func_name:
          print("|- Dwarf Variable", var.name)
        filtered_vars.append(var)
        filtered_tys.append(ty)
        
      else:
        if options.verbose:
          print("|- [ERROR] No AST node found for variable", var.name)
        continue
    
    if len(filtered_vars) > 0 or not options.skip_function_with_no_vars:
      stats.num_functions += 1
    else:
      print("|- [Info] No variable in function", func_name)
      continue

    # Store the explored function
    options.add_explored_function(directory, file_name, func_name)

    # Construct the input and output
    glow_input = GlowInput(input_file_name, directory, file_name, func_name, low_high_pc, ast_graph, filtered_vars,"arch")
    glow_output = GlowOutput(filtered_tys)

    yield (glow_input, glow_output)

  if options.verbose:
    stats.print()

  options.save_explored_function()


def generate_glow_dataset(input_file_name: str, out_file_dir: str, options: Options) -> Iterator[Tuple[GlowInput, GlowOutput]]:
  if options.parallel:
    print("enter parallel")
    return generate_glow_dataset_parallel(input_file_name, out_file_dir, options)
  else:
    return generate_glow_dataset_no_parallel(input_file_name, out_file_dir, options)


# def filter_ill_formed(list_sample):
#   # Stats
#   source_num_functions = len(list_sample)
#   source_num_vars = 0
#   preproc_num_vars = 0

#   result = []
#   (glow_input, glow_output) = list_sample

#   # Stats
#   source_num_vars += len(glow_input.vars)

#   if len(glow_input.vars) == 0:
#     return result
#   new_vars = [v for v in glow_input.vars if len(v.nodes) > 0]
#   new_types = [v for (i, v) in enumerate(glow_output.types) if len(glow_input.vars[i].nodes) > 0]

#   # Stats
#   preproc_num_vars += len(new_vars)

#   if len(new_vars) == 0:
#     return result
#   glow_input.vars = new_vars
#   glow_output.types = new_types
#   result.append(list_sample)

#   # Stats
#   print("Source #vars:", source_num_vars)
#   print("Well Formed #vars:", preproc_num_vars)

#   return result
