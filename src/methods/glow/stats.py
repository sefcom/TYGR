from typing import *

from ...analysis.utils.histogram import Histogram

from .common import GlowInput, PreprocGlowInput, GlowOutput

def stats(dataset: List[Tuple[GlowInput, GlowOutput]]):
  num_funcs = len(dataset)

  num_vars_per_func_histo = Histogram("#Vars/Func")
  num_var_nodes_per_var_histo = Histogram("#Nodes/Var")
  num_var_nodes_per_func_histo = Histogram("#Nodes/Func")
  num_nodes_per_ast_graph_histo = Histogram("#Nodes/Ast Graph")
  num_edges_per_ast_graph_histo = Histogram("#Edges/Ast Graph")
  edge_label_histo = Histogram("Edge Label")
  type_histo = Histogram("Type")
  well_formed_funcs = 0

  bool_results = {}
  bool_edges = []
  charS_results = {}
  charS_edges = []
  structS_results = {}
  structS_edges = []
  for (i, o) in dataset:
    # for index in range(len(i.vars)):
    #   vType = str(i.vars[index].name)
    #   print(vType)
    #   if "array" in vType:
    #     print(type(o.types[index]))
    #     print(o.types[index].elem)
        
    # filter illegal formed
    if len(i.vars) == 0:
      continue
    new_vars = [v for v in i.vars if len(v.nodes) > 0]

    if len(new_vars) == 0:
      continue

    well_formed_funcs += 1
    # Vars
    num_vars_per_func_histo.add(len(i.vars))

    # Var nodes
    num_var_nodes_in_func = 0
    for v in i.vars:
      num_var_nodes_in_func += len(v.nodes)
      num_var_nodes_per_var_histo.add(len(v.nodes))
    num_var_nodes_per_func_histo.add(num_var_nodes_in_func)

    # AstGraph
    num_nodes_per_ast_graph_histo.add(len(i.ast_graph.graph.nodes))
    num_edges_per_ast_graph_histo.add(len(i.ast_graph.graph.edges))
    for edge in i.ast_graph.graph.edges:
      for edge_label in i.ast_graph.edge_to_labels[edge]:
        edge_label_histo.add(edge_label)

    # Types
    for ty in o.types:
      type_histo.add(ty)

  # print(bool_results)
  # print(charS_results)
  # print(structS_results)

  return {
    "num_funcs": num_funcs,
    "well_formed_funcs:": well_formed_funcs,
    "num_vars": num_vars_per_func_histo.total_count(),
    "avg_num_vars_per_func": num_vars_per_func_histo.average_count(),
    # "num_vars_per_func": num_vars_per_func_histo.to_json(),
    "num_var_nodes": num_var_nodes_per_var_histo.total_count(),
    "avg_num_var_nodes_per_var": num_var_nodes_per_var_histo.average_count(),
    # "num_var_nodes_per_var": num_var_nodes_per_var_histo.to_json(),
    # "num_var_nodes_per_func": num_var_nodes_per_func_histo.to_json(),
    "avg_num_nodes_per_ast_graph": num_nodes_per_ast_graph_histo.average_count(),
    #"num_nodes_per_ast_graph": num_nodes_per_ast_graph_histo.to_json(),
    "avg_num_edges_per_ast_graph": num_edges_per_ast_graph_histo.average_count(),
    # "num_edges_per_ast_graph": num_edges_per_ast_graph_histo.to_json(),
    # "edge_labels": edge_label_histo.to_json(),
    "types": type_histo.to_json(),
  }
