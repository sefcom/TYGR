from functools import cmp_to_key

import datetime

import angr
from angr import *
from angr.sim_state import *
from angr.state_plugins.inspect import *
from angr.state_plugins.sim_action import *

from archinfo.arch_arm import ArchARM
import claripy as cp
from claripy.ast.base import *
from claripy.ast.bv import *

from typing import *

from .bityr_annots import *

import networkx as nx

# The thing that is returned from symbolic execution
class SimExecResult(object):
  def __init__(self, proj, cfg, config, tups, rAddr):
    self.proj = proj
    self.cfg = cfg
    self.config = config
    self.tups = tups
    self.rAddr = rAddr

# The abstract execution strategy class provides some useful functionalities
# But the specific details of how to execute a function needs to be defined
# This class can be used repeatedly
class SimExecStrategy(object):
  def __init__(self, proj, config=None):
    self.proj = proj
    self.arch = self.proj.arch

    # Well, we don't really have a better reference point for these settings
    self.default_config = {
      "verbose" : False,
      "init_bp" : self.arch.initial_sp, # There is no arch.initial_bp, unfortuntely
      "init_sp" : self.arch.initial_sp,
      "init_bp_sym_name" : "bp0",
      "init_sp_sym_name" : "sp0",
      "symbolize_init_pointers" : True,
      "cfg" : None,
      "cfg_fast" : True,
      "expr_caching" : True,
      "solver_timeout" : 1000 # In milliseconds
    }

    # Manage some configurations
    if isinstance(config, dict):
      self.config = {**self.default_config, **config}
    else:
      self.config = self.default_config

    # Default options for our initial state
    self.default_sim_options = {
      sim_options.TRACK_MEMORY_ACTIONS,
      sim_options.TRACK_REGISTER_ACTIONS,
      sim_options.TRACK_JMP_ACTIONS,
      sim_options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
      sim_options.ZERO_FILL_UNCONSTRAINED_MEMORY,
      # sim_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
      # sim_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
      # sim_options.NO_SYMBOLIC_JUMP_RESOLUTION,
    }

    self.sim_options = self.default_sim_options

    # Set up the initial CFG
    if self.config["cfg"] is not None:
      self.cfg = self.config["cfg"]
    else:
      if self.config["cfg_fast"]:
        #force_complete_scan=True
        self.cfg = self.proj.analyses.CFGFast()
      else:
        self.cfg = self.proj.analyses.CFGEmulated()

    # The expression cache
    self._bv_to_expr = dict()

  # Make register BV
  def register_offset_bitvector(self, reg_offset : int):
    reg_name = self.arch.translate_register_name(reg_offset)
    reg_info = self.arch.registers.get(reg_name)
    # Lookup a success
    if isinstance(reg_info, tuple) and len(reg_info) == 2:
      return BVV(reg_offset, reg_info[1] * self.arch.byte_width)
    # Otherwise we guess the size
    else:
      return BVV(reg_offset, self.arch.bits)

  # Check if should ignore tracking on this register
  def is_ignored_register_offset(self, reg_offset : int):
    return (reg_offset == self.arch.bp_offset or
            reg_offset == self.arch.sp_offset or
            reg_offset == self.arch.ip_offset or
            reg_offset in self.arch.artificial_registers_offsets)

  # Intercept the to-be-read register data and symbolize + annotate if necessary
  def register_read_before_hook(self, state):
    reg_offset = state.inspect.reg_read_offset
    reg_offset_int = state.solver.eval(reg_offset)
    reg_end = state.inspect.reg_read_endness
    reg_len = state.inspect.reg_read_length
    reg_expr = state.registers.load(reg_offset, size=reg_len, endness=reg_end, inspect=False)

    # If the read does not come from an ignored register, then append an annotation
    if not self.is_ignored_register_offset(reg_offset_int):

      if self.config["expr_caching"]:
        if reg_expr in self._bv_to_expr:
          reg_expr_val = self._bv_to_expr.get(reg_expr)
        else:
          reg_expr_val = state.solver.eval(reg_expr)
          self._bv_to_expr[reg_expr] = reg_expr_val
      else:
        reg_expr_val = reg_expr

      annot = RegisterReadAnnotation(state.addr, reg_offset, reg_expr)
      sym_bv = state.solver.BVS(f"r[{reg_offset_int}]", reg_expr.length)
      sym_bv = sym_bv.annotate(annot)
      state.solver.add(sym_bv == reg_expr_val)
      state.inspect.register_read_expr = sym_bv

  # Intercept the to-be-read memory data and symbolize + annotate
  def memory_read_before_hook(self, state):
    mem_addr = state.inspect.mem_read_address
    mem_end = state.inspect.mem_read_endness
    mem_len = state.inspect.mem_read_length

    if self.config["expr_caching"]:
      if mem_addr in self._bv_to_expr:
        mem_addr_val = self._bv_to_expr.get(mem_addr)
      else:
        mem_addr_val = state.solver.eval(mem_addr)
        self._bv_to_expr[mem_addr] = mem_addr_val
    else:
      mem_addr_val = mem_addr

    mem_expr = state.memory.load(mem_addr_val, size=mem_len, endness=mem_end, inspect=False)

    if self.config["expr_caching"]:
      if mem_expr in self._bv_to_expr:
        mem_expr_val = self._bv_to_expr.get(mem_expr)
      else:
        mem_expr_val = state.solver.eval(mem_expr)
        self._bv_to_expr[mem_expr] = mem_expr_val
    else:
      mem_expr_val = mem_expr

    annot = MemoryReadAnnotation(state.addr, mem_addr, mem_expr)
    sym_bv = state.solver.BVS(f"m[{mem_addr}]", mem_expr.length)
    sym_bv = sym_bv.annotate(annot)
    state.solver.add(sym_bv == mem_expr_val)
    state.inspect.memory_read_expr = sym_bv

  # Intercept the to-be-written register data and symbolize + annotate
  def register_write_before_hook(self, state):
    reg_offset = state.inspect.reg_write_offset
    reg_offset_int = state.solver.eval(reg_offset)
    reg_end = state.inspect.reg_write_endness
    reg_expr = state.inspect.reg_write_expr

    if not self.is_ignored_register_offset(reg_offset_int):

      if self.config["expr_caching"]:
        if reg_expr in self._bv_to_expr:
          reg_expr_val = self._bv_to_expr.get(reg_expr)
        else:
          reg_expr_val = state.solver.eval(reg_expr)
          self._bv_to_expr[reg_expr] = reg_expr_val
      else:
        reg_expr_val = reg_expr

      annot = RegisterWriteAnnotation(state.addr, reg_offset, reg_expr)
      sym_bv = state.solver.BVS(f"r[{reg_offset_int}]", reg_expr.length)
      sym_bv = sym_bv.annotate(annot)
      state.solver.add(sym_bv == reg_expr_val)
      state.inspect.reg_write_expr = sym_bv

  # Intercept the top-be-writen memory data and symbolize + annotate
  def memory_write_before_hook(self, state):
    mem_addr = state.inspect.mem_write_address
    mem_end = state.inspect.mem_write_endness

    if self.config["expr_caching"]:
      if mem_addr not in self._bv_to_expr:
        mem_addr_val = state.solver.eval(mem_addr)
        self._bv_to_expr[mem_addr] = mem_addr_val

    mem_expr = state.inspect.mem_write_expr

    if self.config["expr_caching"]:
      if mem_expr in self._bv_to_expr:
        mem_expr_val = self._bv_to_expr.get(mem_expr)
      else:
        mem_expr_val = state.solver.eval(mem_expr)
        self._bv_to_expr[mem_expr] = mem_expr_val
    else:
      mem_expr_val = mem_expr

    annot = MemoryWriteAnnotation(state.addr, mem_addr, mem_expr)
    sym_bv = state.solver.BVS(f"mem[{mem_addr}]", mem_expr.length)
    sym_bv = sym_bv.annotate(annot)
    state.solver.add(sym_bv == mem_expr_val)
    state.inspect.mem_write_expr = sym_bv

  # Initialize the state for symbolic execution
  def init_symbolic_state(self, block_addr):

    state0 = self.proj.factory.blank_state(addr=block_addr, add_options=self.sim_options)
    state0.solver._solver.timeout = self.config["solver_timeout"]
    reg_end = self.arch.register_endness

    # Because we are ignoring the BP, SP, and IP during hooking, need to manually instatiate them
    # The SP and BP are symbolic since we want to track them, but IP is concrete

    init_bp = state0.solver.BVV(self.config["init_bp"], self.arch.bits)
    init_sp = state0.solver.BVV(self.config["init_sp"], self.arch.bits)

    if self.config["symbolize_init_pointers"]:

      bp_bv = self.register_offset_bitvector(self.arch.bp_offset)
      sp_bv = self.register_offset_bitvector(self.arch.sp_offset)

      sym_bp = state0.solver.BVS(self.config["init_bp_sym_name"], self.arch.bits)
      sym_sp = state0.solver.BVS(self.config["init_sp_sym_name"], self.arch.bits)

      sym_bp = sym_bp.annotate(RegisterWriteAnnotation(state0.addr, bp_bv, init_bp))
      sym_sp = sym_sp.annotate(RegisterWriteAnnotation(state0.addr, sp_bv, init_sp))

      state0.solver.add(sym_bp == init_bp)
      state0.solver.add(sym_sp == init_sp)

      state0.registers.store(self.arch.bp_offset, sym_bp, endness=reg_end, inspect=False)
      state0.registers.store(self.arch.sp_offset, sym_sp, endness=reg_end, inspect=False)
      state0.registers.store(self.arch.ip_offset, block_addr, endness=reg_end, inspect=False)

    else:
      state0.registers.store(self.arch.bp_offset, init_bp, endness=reg_end, inspect=False)
      state0.registers.store(self.arch.sp_offset, init_sp, endness=reg_end, inspect=False)
      state0.registers.store(self.arch.ip_offset, block_addr, endness=reg_end, inspect=False)

    # Finally add the hooking functions
    state0.inspect.add_breakpoint("reg_read",
        BP(when=angr.BP_BEFORE, enabled=True, action=self.register_read_before_hook))

    state0.inspect.add_breakpoint("mem_read",
        BP(when=angr.BP_BEFORE, enabled=True, action=self.memory_read_before_hook))

    state0.inspect.add_breakpoint("reg_write",
        BP(when=angr.BP_BEFORE, enabled=True, action=self.register_write_before_hook))

    state0.inspect.add_breakpoint("mem_write",
        BP(when=angr.BP_BEFORE, enabled=True, action=self.memory_write_before_hook))

    return state0

  # Load from CFG, accounting for possible ARM thumbs
  def get_cfg_node(self, addr : int):
    if isinstance(self.arch, ArchARM):
      node = self.cfg.model.get_node(addr)
      if node is not None:
        return node
      else:
        return self.cfg.model.get_node(addr + 1)
    else:
      return self.cfg.model.get_node(addr)

  def get_cfg_function(self, addr : int):
    if isinstance(self.arch, ArchARM):
      func = self.cfg.functions.get(addr)
      if func is not None:
        return func
      else:
        return self.cfg.functions.get(addr + 1)
    else:
      return self.cfg.functions.get(addr)

  # Get the CFG jumpkind
  def cfg_jumpkind(self, this_addr, next_addr):
    this_cnode = self.get_cfg_node(this_addr)
    next_cnode = self.get_cfg_node(next_addr)
    edge_data = self.cfg.graph.get_edge_data(this_cnode, next_cnode)
    return None if edge_data is None else edge_data.get("jumpkind")

  # Seed the next symbolic state as a function of the jumpkind
  def seed_next_state(self, next_addr, prev_state, jumpkind):
    state0 = prev_state.copy()
    state0.registers.store(self.arch.ip_offset, next_addr, size=self.arch.bytes, inspect=False)

    # Do nothing
    if jumpkind == "Ijk_Boring":
      return state0

    elif jumpkind == "Ijk_FakeRet":
      # Symbolize the return register
      ret_bv = self.register_offset_bitvector(self.arch.ret_offset)
      sym_ret = state0.solver.BVS("ret", self.arch.bits)
      sym_ret = sym_ret.annotate(RegisterWriteAnnotation(state0.addr, ret_bv, sym_ret))
      state0.registers.store(self.arch.ret_offset, sym_ret, size=self.arch.bytes, inspect=False)

      # Also symbolize the argument registers
      for areg_offset in self.arch.argument_registers:
        areg_name = self.arch.translate_register_name(areg_offset)
        areg_info = self.arch.registers.get(areg_name)
        if (isinstance(areg_info, tuple) and len(areg_info) == 2 and
            not (areg_offset == self.arch.bp_offset or areg_offset == self.arch.sp_offset)):
          areg_bv = self.register_offset_bitvector(areg_offset)
          areg_bitsize = areg_info[1] * self.arch.byte_width
          sym_areg = state0.solver.BVS("r[{areg_offset}]", areg_bitsize)
          sym_areg = sym_areg.annotate(RegisterWriteAnnotation(state0.addr, areg_bv, sym_areg))
          state0.registers.store(areg_offset, sym_areg, size=areg_info[1], inspect=False)

      return state0

    else:
      if self.config["verbose"]:
        print(f"seed_next_state: from {prev_state} to {hex(next_addr)} has jumpkind {jumpkind}")
      return None

  # Run symbolic on a single non-branching basic block
  # Execution stops when one of the following happens:
  #   * There is more than one successor state
  #   * The next instruction lies outside the current basic block
  #   * We have run all the instructions in the basic block
  def sim_exec_block(self, state0, block):
    if block is None:
      return None

    rem_instrs = block.instructions
    max_steps = block.instructions
    step_counter = 0

    block_instruction_addrs = set(block.instruction_addrs)
    state = state0

    while state.addr in block_instruction_addrs and step_counter < max_steps:
      this_state = state
      if self.config["verbose"]:
        print(f"about to step: {this_state}")

      # A step size is specified
      if "step_size" in self.config:
        # We have sufficient budget
        if step_counter + self.config["step_size"] + 5 < max_steps:
          step_size = self.config["step_size"]
        else:
          step_size = 1

      # A step size is not specified
      else:
        if step_counter + 5 < max_steps:
          step_size = 2
        else:
          step_size = 1

      step_counter += step_size

      # Do the step
      try:
        this_succs = this_state.step(num_inst=step_size)
      except Exception as exc:
        state = this_state
        if self.config["verbose"]:
          print(f"sim_exec_block: returning with {state} due to exception {exc}")
        break

      # If there are no successors, return
      if len(this_succs.successors) == 0:
        state = this_state
        break

      # If there is more than one successor, stop!
      elif len(this_succs.successors) > 1:
        # Arbitrarily choose to return the first successor
        state = this_succs.successors[0]
        break

      # If there is exactly one successor, can maybe do something
      else:
        next_state = this_succs.successors[0]

        # Check whether a jump happened
        reg_end = self.arch.register_endness
        next_addr = next_state.registers.load(self.arch.ip_offset, self.arch.bytes, endness=reg_end, inspect=False)
        jump_happened = (next_addr.symbolic or
                         next_state.addr not in block_instruction_addrs or
                         next_state.addr < this_state.addr)
        if jump_happened:
          state = next_state
          break

        # Check if the stack is unwinding
        # We should be able to eval both bp's down to actual values
        this_bp = this_state.registers.load(self.arch.bp_offset, self.arch.bytes, endness=reg_end, inspect=False)
        next_bp = next_state.registers.load(self.arch.bp_offset, self.arch.bytes, endness=reg_end, inspect=False)

        this_bp_int = this_state.solver.eval(this_bp)
        next_bp_int = next_state.solver.eval(next_bp)
        if this_bp_int < next_bp_int:
          state = this_state
          break

        # ... Otherwise proceed
        state = next_state

    return state

  # Reset things
  def reset(self):
    raise NotImplementedError

  def init_at_addr(self, func_addr):
    raise NotImplementedError

  # Returns a list of final states
  def sim_exec_function(self, func_addr):
    raise NotImplementedError

# A very basic dominator strategy
class SimpleDominatorStrategy(SimExecStrategy):
  def __init__(self, proj, config=None):
    super().__init__(proj, config=config)

    if config and "max_visit_nodes" in config:
      self.max_visit_nodes = config["max_visit_nodes"]
    else:
      self.max_visit_nodes = 200

    self._visited_counter = 0
    self._function = None
    self._predoms = None

  def reset(self):
    self._bv_to_expr = dict()
    self._visited_counter = 0
    self._function = None
    self._predoms = None

  def init_at_addr(self, func_addr):
    self.reset()
    self._function = self.get_cfg_function(func_addr)
    self._predoms = nx.DiGraph(list(nx.immediate_dominators(self._function.graph, self._function.startpoint).items()))
    return self._function.startpoint

  def is_predom_leaf(self, dnode):
    preds = list(self._predoms.predecessors(dnode))
    return len(preds) == 0 or (preds == [dnode])

  def walk_predom_asts(self, this_dnode, this_state0):
    this_cnode = self.get_cfg_node(this_dnode.addr)
    this_block = this_cnode.block
    this_statef = self.sim_exec_block(this_state0, this_block)

    self._visited_counter += 1

    # Figure out which nodes are strictly dominated
    child_results = []
    if self._visited_counter <= self.max_visit_nodes:
      child_dnodes = list(self._predoms.predecessors(this_dnode))
      for child_dnode in child_dnodes:
        # Note: for some reason the idom calculation leaves a copy of the root node in the
        # dominated set, no idea why. Ayways, we must prune this out to avoid recursion
        if child_dnode.addr is not this_dnode.addr:
          jumpkind = self.cfg_jumpkind(this_dnode.addr, child_dnode.addr)
          child_state0 = self.seed_next_state(child_dnode.addr, this_statef, jumpkind)
          if child_state0 is not None:
            child_ret = self.walk_predom_asts(child_dnode, child_state0)
            child_results.extend(child_ret)

    return [(this_dnode, this_state0, this_statef)] + child_results

  def sim_exec_function(self, func_addr):
    init_dnode = self.init_at_addr(func_addr)
    state0 = self.init_symbolic_state(func_addr)
    results = self.walk_predom_asts(init_dnode, state0)

    tups = []

    for trip in results:
      dnode = trip[0]
      # If this thing has any predecessors, that means it is the leaf in the dom tree, so return
      if self.is_predom_leaf(dnode):
        cnode = self.get_cfg_node(dnode.addr)
        tups.append((cnode, trip[1], trip[2]))

    return SimExecResult(self.proj, self.cfg, self.config, tups)


# A more sophisticated one
class LessSimpleDominatorStrategy(SimExecStrategy):
  def __init__(self, proj, config=None):
    super().__init__(proj, config=config)

    if config and "max_visit_nodes" in config:
      self.max_visit_nodes = config["max_visit_nodes"]
    else:
      self.max_visit_nodes = 100

    if config and "max_state_constraints" in config:
      self.max_state_constraints = config["max_state_constraints"]
    else:
      self.max_state_constraints = 10000

    # Some temporary variables used on a per-function basis
    self._function = None
    self._predoms = None
    self._postdoms = None
    self._addr_to_final_state = dict()
    self._addr_exec_order = []
    self._addrs_used_to_seed = set()

  def reset(self):
    self._bv_to_expr = dict()
    self._function = None
    self._predoms = None
    self._postdoms = None
    self._addr_to_final_state = dict()
    self._addr_exec_order = []
    self._addrs_used_to_seed = set()

  def init_at_addr(self, func_addr):
    self.reset()
    self._function = self.get_cfg_function(func_addr)
    self._function_graph = self._function.graph

    # NOTE: the cfg nodes are different from the pre/post dom nodes!
    self._predoms = nx.DiGraph(list(nx.immediate_dominators(self._function_graph, self._function.startpoint).items()))

    self._function_graph_rev = self._function.graph.reverse(copy=True)
    # One issue with post-domination is that the CFG may have multiple endpoints,
    # as angr doesn't seem to provide a single sink.
    # Consequently, we sample a few endpoints and pick the one with most nodes
    max_endpoint_samples = 3
    sample_postdoms = []
    for (i, endpoint) in enumerate(self._function.endpoints):
      if i + 1 > max_endpoint_samples:
        break

      postdom = nx.DiGraph(list(nx.immediate_dominators(self._function_graph_rev, endpoint).items()))
      sample_postdoms.append(postdom)

    if len(sample_postdoms) == 0:
      if self.config["verbose"]:
        print(f"LessSimpleDominatorStrategy: function at {hex(func_addr)} has no endpoints?")
      self._postdoms = nx.DiGraph()
      self._postdoms.add_node(self._function.startpoint)
      return

    sample_postdoms.sort(key=lambda g : g.order(), reverse=True)
    self._postdoms = sample_postdoms[0]

    # We can also tell the order to execute things in
    dnode_exec_order = sorted(self._predoms.nodes(), key=cmp_to_key(self.exec_order_cmp))
    self._addr_exec_order = list(map(lambda dnode : dnode.addr, dnode_exec_order))

  # The dominator trees point upwards to the root, hence the descendant check
  # Also: every node technically both pre- and post-dominates itself
  def pre_dominates(self, a, b):
    if a == b:
      return True
    else:
      return self._predoms.has_node(b) and (a in nx.descendants(self._predoms, b))

  def post_dominates(self, a, b):
    if a == b:
      return True
    else:
      return self._postdoms.has_node(b) and (a in nx.descendants(self._postdoms, b))

  # Determine if one state should be executed before the other based on dominator comparison
  def exec_order_cmp(self, a, b):
    if a == b: return 0
    elif self.pre_dominates(a, b): return -1
    elif self.pre_dominates(b, a): return 1
    else:
      if self.post_dominates(a, b): return 1
      elif self.post_dominates(b, a): return -1
      else: return 0

  # Figure out the next state given the address
  def init_next_state(self, addr):

    # If have nothing in the state cache and this is the initial address
    if addr == self._function.addr:
      return self.init_symbolic_state(addr)

    # Lookup the predecessor addresses in the function graph
    else:
      this_dnode = self._function.get_node(addr)
      # Check if each predecessor has a cached state
      for pred_dnode in self._function_graph.predecessors(this_dnode):
        pred_statef = self._addr_to_final_state.get(pred_dnode.addr)
        # We found somethingin the cache that doesn't have too many constraints!
        if pred_statef is not None and len(pred_statef.solver.constraints) < self.max_state_constraints:
          jumpkind = self.cfg_jumpkind(pred_dnode.addr, this_dnode.addr)
          next_state0 = self.seed_next_state(addr, pred_statef, jumpkind)
          self._addrs_used_to_seed.add(pred_dnode.addr)
          return next_state0

  def sim_exec_function(self, func_addr):
    self.init_at_addr(func_addr)
    pre_tups = []

    total_addrs = len(self._addr_exec_order)
    visit_nodes_counter = 0

    #return addr executed
    rAddr = []

    for i, addr in enumerate(self._addr_exec_order):
      if visit_nodes_counter > self.max_visit_nodes:
        break

      if self.config["verbose"]:
        now = datetime.datetime.now()
        print(f"[{i+1}/{total_addrs}] {now} \t starting at {hex(addr)} \t counter = {visit_nodes_counter}")

      state0 = self.init_next_state(addr)

      # If we succeeded in initializing the next state from the cache ...
      if state0 is not None:
        visit_nodes_counter += 1
        block = self.proj.factory.block(addr)
        statef = self.sim_exec_block(state0, block)
        if statef is not None:
          self._addr_to_final_state[addr] = statef
          cnode = self.get_cfg_node(addr)
          pre_tups.append((cnode, state0, statef))
          #added
          rAddr.append(addr)
        else:
          if self.config["verbose"]:
            print(f"LessSimple: failed to execute state to termination starting from {hex(addr)}")

      else:
        if self.config["verbose"]:
          print(f"LessSimple: failed to initialize state at {hex(addr)}")

      if self.config["verbose"]:
        print(f"items in expr cache: {len(self._bv_to_expr)}")

    tups = list(filter(lambda t : t[1].addr not in self._addrs_used_to_seed, pre_tups))
    return SimExecResult(self.proj, self.cfg, self.config, tups, rAddr)

