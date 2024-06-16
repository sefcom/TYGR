import ctypes
from curses import noecho
from operator import index
import sys
from typing import *
import networkx as nx
import matplotlib.pyplot as plt

import claripy as cp
import claripy.ast as ast
from claripy.ast.base import *

from angr.sim_state import *
from angr.state_plugins.sim_action import *
from networkx.classes.function import subgraph
from numpy import int64, uint64
from torch import long

from .bityr_annots import *

import collections
import itertools 
from networkx.drawing.nx_agraph import graphviz_layout

import math
import ctypes
import sys
# Some dummy things for the flow graph
class Label(object): pass

class NodeLabel(Label):
  def __init__(self, addr, ast, ast_val):
    self.addr = addr  # The PC value
    # self.ast = ast
    self.ast_val = ast_val
    # features for node
    # self.baseAddr_ast_value = baseAddr_ast_value
    # self.index_ast_value = index_ast_value 

    if isinstance(ast, cp.ast.Bool):
      self.bitsize = 1
    elif isinstance(ast, cp.ast.Base):
      self.bitsize = 0 if ast.length is None else ast.length
    else:
      raise Exception(f"NodeLabel: {ast} of type {type(ast)} does no have bitsize")
  
  def __str__(self):
    if self.addr is not None:
      return f"NodeLabel({hex(self.addr)}, {hex(self.ast_val)}, {self.bitsize})"
    else:
      return f"NodeLabel({self.addr}, {hex(self.ast_val)}, {self.bitsize})"

# Define a class for building up an AST graph
class AstGraph(object):
  def __init__(self):
    self.graph = nx.DiGraph()
    self.node_to_label = dict()
    self.edge_to_labels = dict()
    self.ast_to_node = dict() # Hashing by top-level AST
    self._next_node = 0

    # I think this number will literally scale exponentially
    # self.max_process_ast_depth = int(sys.getrecursionlimit() / 2)
    self.max_process_ast_depth = 200

  # Return the allocated node
  def alloc_node(self, nlabel : NodeLabel):
    node = self._next_node
    self._next_node += 1
    self.node_to_label[node] = nlabel
    # self.graph.add_node(node, label=nlabel)
    self.graph.add_node(node)
    return node

  # Edges may have a set of labels; adding an edge grows that set
  # Return the new labels that exist
  def add_edge(self, src, dst, elabel):
    old_labels = self.edge_to_labels.get((src, dst))
    new_labels = (old_labels if old_labels else set()) | {elabel}
    # self.graph.add_edge(src, dst, label=new_labels)
    self.graph.add_edge(src, dst)
    self.edge_to_labels[(src, dst)] = new_labels
    return (src, dst, new_labels)

  def edge_exists(self, src, dst, elabel):
    cur_labels = self.edge_to_labels.get((src, dst))
    if isinstance(cur_labels, set):
      return elabel in cur_labels
    else:
      return False

  def directed_splice_and_normalize(self, nodes: set) -> Tuple[Any, Dict[int, int]]:
    desc = set()
    ansc = set()
    #firstNEle = 7
   
    for node in nodes:
      
      desc = desc | nx.descendants(self.graph, node)
      ansc = ansc | nx.ancestors(self.graph, node)
    # Create the subgraph
    sub_nodes = nodes | desc | ansc
    sub_graph = nx.subgraph(self.graph, list(sub_nodes))

    # Normalize the subgraph
    mapping = { n: i for (i, n) in enumerate(sub_nodes) }
    normed_subgraph = nx.relabel_nodes(sub_graph, mapping)

    ## slicing has problem 
    new_node_to_label = { mapping[n]: l for (n, l) in self.node_to_label.items() if n in sub_nodes }
    new_edge_to_labels = { (mapping[s], mapping[d]): l for ((s, d), l) in self.edge_to_labels.items() if (s, d) in sub_graph.edges }
    new_ast_to_node = { ast: mapping[n] for (ast, n) in self.ast_to_node.items() if n in sub_nodes }

    new_ast = AstGraph()
    new_ast.graph = normed_subgraph
    new_ast.node_to_label = new_node_to_label
    new_ast.edge_to_labels = new_edge_to_labels
    new_ast.ast_to_node = new_ast_to_node
    new_ast._next_node = len(new_ast.graph.nodes)

    return (new_ast, mapping)

  def directed_splice(self, nodes: set):
    desc = set()
    ansc = set()
    for node in nodes:
      desc = desc | nx.descendants(self.graph, node)
      ansc = ansc | nx.ancestors(self.graph, node)

    sub_nodes = nodes | desc | ansc
    sub_graph = nx.subgraph(self.graph, list(sub_nodes)).copy()
    sub_edges = set(sub_graph.edges())

    sub_node_to_label = {n : l for (n, l) in self.node_to_label.items() if n in sub_nodes}
    sub_edge_to_labels = {e : l for (e, l) in self.edge_to_labels.items() if e in sub_edges}

    sub_ast = AstGraph()
    sub_ast.graph = sub_graph
    sub_ast.node_to_label = sub_node_to_label
    sub_ast.edge_to_labels = sub_edge_to_labels
    sub_ast.ast_to_node = {ast: n for (ast, n) in self.ast_to_node.items() if n in sub_nodes}
    sub_ast._next_node = self._next_node

    return sub_ast

  def undirected_splice(self, nodes : set):
    dir_splice = self.directed_splice(nodes)
    undir_splice = self.directed_splice(dir_splice.graph.nodes())
    return undir_splice

  def normalize_node_ids(self) -> Tuple[Any, Dict[int, int]]:
    mapping = { n: i for (i, n) in enumerate(self.graph.nodes) }
    normed_graph = nx.relabel_nodes(self.graph, mapping)

    normed_ast_graph = AstGraph()
    normed_ast_graph.graph = normed_graph
    normed_ast_graph.node_to_label = { mapping[n]: l for (n, l) in self.node_to_label.items() }
    normed_ast_graph.edge_to_labels = { (mapping[s], mapping[d]): l for ((s, d), l) in self.edge_to_labels.items() }
    normed_ast_graph.ast_to_node = { ast: mapping[n] for (ast, n) in self.ast_to_node.items() }
    normed_ast_graph._next_node = len(mapping)

    return (normed_ast_graph, mapping)

  # Delete some information right before a pickle call
  def ready_for_pickling(self):
    # Clear things that may still have bitvector ASTs
    self.ast_to_node = dict()

  
    # Get a string representation of a node
 
  def node_to_str(self, node):
    nlabel = self.node_to_label.get(node)
    if nlabel is not None:
      # This is the ast_val
      return f"[{node}] {hex(nlabel.ast_val)}"

    else:
      return f"[{node}]"

  # Iteratively builds up the current graph by processing a bitvector AST
  # Returns the graph node corresponding to the current AST root
  def process_ast(self, state, ast, func_param_locs:list, recursion_depth=0):
    assert isinstance(ast, cp.ast.base.Base)

    # Check if this AST already exists
    ## the following exception added due to new version of angr
    this_node = self.ast_to_node.get(ast)
    if this_node is not None:
      return this_node
  
    ast_val = state.solver.eval(ast)

    if isinstance(ast_val, int) and not isinstance(ast_val, bool):      
      if(len(ast) == 64):
        ast_val = ctypes.c_long(ast_val).value
      elif(len(ast) == 32):
        ast_val = ctypes.c_int(ast_val).value
      
    annot = ast.annotations[0] if len(ast.annotations) > 0 else None
    annot_addr = annot.addr if isinstance(annot, BityrAnnotation) else None

    nlabel = NodeLabel(annot_addr, ast, ast_val)
    this_node = self.alloc_node(nlabel)
    self.ast_to_node[ast] = this_node

    # If we are gonna be too deep, just quit
    if recursion_depth > self.max_process_ast_depth:
      return this_node

    # Recursively process annotations, if applicable
    for annot in ast.annotations:
      if isinstance(annot, BityrAnnotation):

        loc_node = self.process_ast(state, annot.loc, func_param_locs, recursion_depth + 1)
        data_node = self.process_ast(state, annot.data, func_param_locs, recursion_depth + 1)

        if isinstance(annot, MemoryReadAnnotation) or isinstance(annot, MemoryWriteAnnotation):
          loc_edge_str = "mem_loc"
          data_edge_str = "mem_data"

        elif isinstance(annot, RegisterReadAnnotation) or isinstance(annot, RegisterWriteAnnotation):
          loc_edge_str =  "reg_loc"
          data_edge_str = "reg_data"

        else:
          loc_edge_str = "loc"
          data_edge_str = "data"

        if not self.edge_exists(loc_node, this_node, loc_edge_str):
          self.add_edge(loc_node, this_node, loc_edge_str)

        if not self.edge_exists(data_node, this_node, data_edge_str):
          self.add_edge(data_node, this_node, data_edge_str)
  
    # Stop recursion at base case, otherwise investigate all the children
    if ast.depth > 1:
      for child_ast in ast.args:
        if isinstance(child_ast, cp.ast.base.Base):
          child_node = self.process_ast(state, child_ast, func_param_locs, recursion_depth + 1)
          # filter here for operation
          if not self.edge_exists(child_node, this_node, ast.op):
            self.add_edge(child_node, this_node, ast.op)  
    
    return this_node

  def process_state(self, state, func_param_locs:list):
    # original
    #(reg_writes, mem_writes) = get_state_writes(state)
    (reg_writes, mem_writes, reg_reads, mem_reads) = get_state_WR(state, func_param_locs)

    for i, (reg_offset, (addr, reg_expr)) in enumerate(reg_writes.items()):
      reg_bv = state.solver.BVV(reg_offset, reg_expr.length)
      annot = RegisterWriteAnnotation(addr, reg_bv, reg_expr)
      todo_bv = state.solver.BVS(f"r[{reg_offset}]", reg_expr.length)
      todo_bv = todo_bv.annotate(annot)
      self.process_ast(state, todo_bv, func_param_locs)

    
    for i, (mem_addr, (addr, mem_expr)) in enumerate(mem_writes.items()):
      annot = MemoryWriteAnnotation(addr, mem_addr, mem_expr)
      todo_bv = state.solver.BVS(f"m[{mem_addr}]", mem_expr.length)
      todo_bv = todo_bv.annotate(annot)
      self.process_ast(state, todo_bv, func_param_locs)


    for i, (reg_offset, (addr, reg_expr)) in enumerate(reg_reads.items()):
      reg_bv = state.solver.BVV(reg_offset, reg_expr.length)
      annot = RegisterReadAnnotation(addr, reg_bv, reg_expr)
      todo_bv = state.solver.BVS(f"r[{reg_offset}]", reg_expr.length)
      todo_bv = todo_bv.annotate(annot)   
      self.process_ast(state, todo_bv, func_param_locs)


    for i, (mem_addr, (addr, mem_expr)) in enumerate(mem_reads.items()):
      annot = MemoryReadAnnotation(addr, mem_addr, mem_expr)
      todo_bv = state.solver.BVS(f"m[{mem_addr}]", mem_expr.length)
      todo_bv = todo_bv.annotate(annot)
      self.process_ast(state, todo_bv, func_param_locs) 


    conds = get_state_exit_conds(state)
    for i, cond in enumerate(conds):
      self.process_ast(state, cond, func_param_locs)

  # Find the nodes whose ast_val equals what we're looking for
  def find_mem_loc_node(self, addr_val):
    nodes = set()
    for node in self.graph.nodes():
      nlabel = self.node_to_label.get(node)
      if isinstance(nlabel, NodeLabel):
        if nlabel.ast_val == addr_val:
          for succ in self.graph.successors(node):
            edge_labels = self.edge_to_labels.get((node, succ))
            if edge_labels is not None:
              for elabel in edge_labels:
                if elabel == "mem_loc":
                  nodes.add(succ)
    return nodes

  def find_nodes_from_locs(self, var_locs):
    if var_locs is None or len(var_locs) == 0:
      return set()

    found_nodes = set()
    for node in self.graph.nodes():
      nlabel = self.node_to_label.get(node)
      if isinstance(nlabel, NodeLabel):

        # Check if the node is a memory or register in our location list
        node_is_interesting = False
        for (_, (form, arg)) in var_locs:
          if form == "addr" and nlabel.ast_val == arg:
            node_is_interesting = True
            break
          elif form == "reg" and nlabel.ast_val == arg:
            node_is_interesting = True
            break

        # Not interesting? Keep going ...
        if not node_is_interesting:
          continue

        # Interesting? Let's see which successors to add
        for succ in self.graph.successors(node):
          edge_labels = self.edge_to_labels.get((node, succ))
          if isinstance(edge_labels, set):
            for elabel in edge_labels:
              # Check if the interesting node's mem_loc succ is in the appropriate addr range
              if elabel == "mem_loc":
                slabel = self.node_to_label.get(succ)
                if isinstance(slabel, NodeLabel):
                  for ((low_addr, high_addr), (form, arg)) in var_locs:
                    # All the criteria match
                    if (isinstance(slabel.addr, int) and
                        low_addr <= slabel.addr and slabel.addr < high_addr and
                        form == "addr" and arg == nlabel.ast_val):
                      found_nodes.add(succ)

              # Check if the interesting node's reg_loc succ is in the appropriate addr range
              if elabel == "reg_loc":
                slabel = self.node_to_label.get(succ)
                if isinstance(slabel, NodeLabel):
                  for ((low_addr, high_addr), (form, arg)) in var_locs:
                    # All the criteria match
                    if (isinstance(slabel.addr, int) and
                        low_addr <= slabel.addr and slabel.addr < high_addr and
                        form == "reg" and arg == nlabel.ast_val):
                      found_nodes.add(succ)

    return found_nodes

  # Determine if a node is possibly a stack offset
  def node_is_possibly_stack_offset(self, simres, node):
    # Check that:
    #   * It is within 0x1000 from the initial sp
    #   * There are mem_loc edges coming from it
    nlabel = self.node_to_label.get(node)
    if isinstance(nlabel, NodeLabel):
      init_sp_diff = nlabel.ast_val - simres.config["init_sp"]
      if abs(init_sp_diff) < 0x100000:
        for succ in self.graph.successors(node):
          if self.edge_exists(node, succ, "mem_loc"):
            return True

        return False
    else:
      return False

  # Plot the ast graph
  def plot(self, filename="hello2.png", save=True, show=True, verbose=False):
    if verbose:
      sorted_nodes = sorted(self.graph.nodes())
      for node in sorted_nodes:
        label = self.node_to_label.get(node)
        annot = label.data[2]
        if annot is not None:
          print(f"[{node}] {self.node_to_str(node)} has annot {annot}")
        else:
          print(f"[{node}] {self.node_to_str(node)}")

    pos = nx.spring_layout(self.graph, scale=3, k=5/math.sqrt(self.graph.order()))
    node_labels = {node : self.node_to_str(node) for node in self.graph.nodes()}
    edge_labels = {edge : str(label) for edge, label in self.edge_to_labels.items()}

    nx.draw(self.graph,
        pos,
        labels=node_labels,
        alpha=0.8,
        node_size=100,
        font_size=8)

    nx.draw_networkx_edge_labels(self.graph,
        pos,
        edge_labels=edge_labels,
        font_size=8)

    if save:
      plt.savefig(filename)

    if show:
      plt.show()

  def simple_edge_label_str(self, label: str) -> str:
    if label == "__add__": return "+"
    elif label == "__sub__": return "-"
    else: return label

  def save_dot(self, filename="graph.dot", mark_nodes={}):
    from networkx.drawing.nx_agraph import write_dot
    import networkx as nx

    s = ""
    s += "strict digraph \"\" {"

    for node_id in self.graph.nodes:
      mark = "fillcolor=blue, style=filled, fontcolor=white" if node_id in mark_nodes else ""
      nlabel = self.node_to_label.get(node_id)

      if isinstance(nlabel, NodeLabel):
        ast = nlabel.ast
        if ast.symbolic and len(ast.annotations) > 0:
          annot = ast.annotations[0]
          if isinstance(annot, RegisterReadAnnotation):
            label_str = '"RegRead"'
          elif isinstance(annot, RegisterWriteAnnotation):
            label_str = '"RegWrite"'

          elif isinstance(annot, MemoryReadAnnotation):
            label_str = '"MemRead"'

          elif isinstance(annot, MemoryWriteAnnotation):
            label_str = '"MemWrite"'
          
          else:
            label_str = f'"id[{node_id}]"'

        else:
          label_str = f'"{hex(nlabel.ast_val)}"'

      else:
        label_str = f'"id[{node_id}]"'

      # label_str = f'"{hex(nlabel.ast_val)}"' if nlabel.ast.concrete else f'"Node[{node_id}]"'

      s += f"{node_id} [ {mark} label={label_str} ];"

    for edge_id in self.graph.edges:
      label = str({self.simple_edge_label_str(label) for label in self.edge_to_labels[edge_id]})
      s += f"{edge_id[0]} -> {edge_id[1]} [ label=\"{label}\" ];"

    s += "}"

    with open(filename, "w") as f:
      f.write(s)

# Get the write-to destinations of a state
def get_state_writes(state):
  arch = state.arch
  reg_writes = dict()
  mem_writes = dict()
  acts = state.history.actions.hardcopy

  # Forward traverse the actions to gather the used registers and addresses
  for act in acts:
    if isinstance(act, SimActionData):
      #print(dir(act))
      if act.action == "write" and act.type == "reg":
        reg_offset = act.offset
        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.registers.load(reg_offset, size=bytesize, endness=arch.register_endness, inspect=False)
        reg_writes[reg_offset] = (act.bbl_addr, data)
        # print(act)
        # print(act.offset)
        # print(act.actual_addrs)
        # print(" ")
      elif act.action == "write" and act.type == "mem":
        mem_addr = act.addr.ast
        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.memory.load(mem_addr, size=bytesize, endness=arch.memory_endness, inspect=False)
        mem_writes[mem_addr] = (act.bbl_addr, data)
        # print(act)
        # print(act.addr.ast)
        # print(act.actual_addrs)
        # print(" ")
  return (reg_writes, mem_writes)

def get_state_WR(state, func_param_locs: list):
  arch = state.arch
  reg_writes = dict()
  reg_reads = dict()
  mem_writes = dict()
  mem_reads = dict()
  acts = state.history.actions.hardcopy
  # Forward traverse the actions to gather the used registers and addresses

  # if an argument register has been written to, we can no longer assign the function parameter name to it
  func_param_reg_written = []

  for act in acts:

    if isinstance(act, SimActionData):
      
      if act.action == "write" and act.type == "reg":
        reg_offset = act.offset

        #skip this node bec reg has been written to
        if reg_offset in func_param_reg_written:

          continue      

        if reg_offset in func_param_locs:
          # add written register and skip

          func_param_reg_written.append(reg_offset)
          continue

        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.registers.load(reg_offset, size=bytesize, endness=arch.register_endness, inspect=False)
        reg_writes[reg_offset] = (act.bbl_addr, data)

      elif act.action == "write" and act.type == "mem":
        mem_addr = act.addr.ast
        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.memory.load(mem_addr, size=bytesize, endness=arch.memory_endness, inspect=False)
        mem_writes[mem_addr] = (act.bbl_addr, data)

      elif act.action == "read" and act.type == "reg":
        reg_offset = act.offset

        # skip this node bec reg has been written to
        if reg_offset in func_param_reg_written:
          continue   
    
        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.registers.load(reg_offset, size=bytesize, endness=arch.register_endness, inspect=False)
        reg_reads[reg_offset] = (act.bbl_addr, data)
       
      elif act.action == "read" and act.type == "mem":
        mem_addr = act.addr.ast
        bytesize = int(act.data.ast.length / arch.byte_width)
        data = state.memory.load(mem_addr, size=bytesize, endness=arch.memory_endness, inspect=False)
        mem_reads[mem_addr] = (act.bbl_addr, data)

  return (reg_writes, mem_writes, reg_reads, mem_reads)

# Get the jumps (exits) of a state
def get_state_exit_conds(state):
  conds = []
  acts = state.history.actions.hardcopy
  for act in acts:
    if isinstance(act, SimActionExit):
      cond = act.condition
      if cond is not None:
        conds.append(cond.ast)

  return conds


# Use the state and the config that was used to initialize the state
# NOTE: this only works if we had symbolized the initial pointers
def get_stack_write_offset(state, config, addr):
  addr_val = state.solver.eval(addr)
  # Compare the address' concrete value against the config's initial SP
  # The offset for local variables should be negative
  init_sp_diff = addr_val - config["init_sp"]
  contains_name = config["init_sp_sym_name"] in str(addr)
  if contains_name:
    return init_sp_diff
  else:
    return None
