from typing import *
import json
import datetime
import torch

import angr

from multiprocessing import Pool

from .datagen import *

# The inputs to the Glow GNN
def collate_inputs(inputs):
  node_labels = []
  edges = []
  edge_labels = []
  node_offset = 0
  var_gather = []
  var_scatter = []
  num_vars = 0
  list_num_vars = []
  for x in inputs:
    node_labels.append(x.node_labels)
    cur_edges = x.edges + node_offset
    edges.append(cur_edges)
    edge_labels.append(x.edge_labels)

    for nodes in x.var_nodes:
      var_gather += [t + node_offset for t in nodes]
      var_scatter += [num_vars] * len(nodes)
      num_vars += 1
    list_num_vars.append(len(x.var_nodes))
    node_offset += x.node_labels.shape[0]

  node_labels = torch.cat(node_labels, dim=0)
  edge_labels = torch.cat(edge_labels, dim=0)
  edges = torch.cat(edges, dim=1)
  var_gather = torch.LongTensor(var_gather).to(node_labels.device)
  var_scatter = torch.LongTensor(var_scatter).to(var_gather.device)
  return list_num_vars, node_labels, edge_labels, edges, var_gather, var_scatter

# Insert
def add_to_set_valued(d, k, v):
  vals = d.get(k)
  new_vals = (vals if vals else set()) | {v}
  d[k] = new_vals

# Generating the inputs during prediction
def generate_glow_inputs_parallel(input_file_name : str, options: Options) -> Iterator[Tuple[GlowInput]]:
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

  func_addrs = list(strategy.cfg.functions)
  tasks = []
  for low_pc in func_addrs:
    tasks.append((low_pc, [], strategy, options))

  with Pool(options.pool_size) as pool:
    sym_exec_results = pool.map(sym_exec_one, tasks)

  for task, (ast_graph, vars, _) in zip(tasks, sym_exec_results):
    if len(ast_graph.graph.nodes) == 0:
      continue

    (low_pc, _, _, _) = task
    glow_input = GlowInput(input_file_name, None, None, None, (low_pc, None), ast_graph, vars,"x86")
    yield glow_input


def generate_glow_inputs(input_file_name: str, options: Options) -> Iterator[Tuple[GlowInput, GlowOutput]]:
  if options.parallel:
    return generate_glow_inputs_parallel(input_file_name, options)

  # For now, only parallel ... :)
  else:
    print("Too bad, we are forcing parallelization!")
    return generate_glow_inputs_parallel(input_file_name, options)

def generate_bickle(output: GlowOutput, bickle_file: str):
  print(f"generate_bickle: called for glow to write to {bickle_file}")



