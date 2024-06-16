from typing import *
import copy

from angr import Project
from angr.analyses.cfg.cfg_fast import CFGFast
from pyvex import IRSB
from pyvex.stmt import *
from pyvex.expr import IRExpr, RdTmp
from archinfo.arch import Arch

from ..dwarf import VarLocation
from .tmp_var import TmpVar
from . import const_fold

base_addr = 0x000000

load_options = {
  "main_opts": {
    "base_addr": base_addr
  },
  "auto_load_libs": False
}

def get_vex_irsbs(input_file_name: str) -> List[IRSB]:
  # Initialize the project and CFG
  proj = Project(input_file_name, load_options=load_options)
  cfg = proj.analyses.CFGFast()

  # Gather all the nodes, each of which has a block, which has an IRSB
  all_nodes = cfg.graph.nodes()
  addrs = sorted(map(lambda node: node.addr, all_nodes))

  # Get all the blocks
  blocks = map(proj.factory.block, addrs)
  irsbs = list(map(lambda block : block.vex, blocks))
  return irsbs

def irsb_pc_range(irsb: IRSB) -> Tuple[int, int]:
  imarks = filter(lambda stmt: isinstance(stmt, IMark), irsb.statements)
  pcs = list(map(lambda imark: imark.addr, imarks))
  pcs.sort()
  if len(pcs) > 0:
    return pcs[0], pcs[-1]
  else:
    return 0, 0

def iter_irsbs_in_range(low: int, high: int, irsbs: List[IRSB]) -> Iterator[Tuple[int, IRSB]]:
  for i, irsb in enumerate(irsbs):
    (irsb_low_pc, irsb_high_pc) = irsb_pc_range(irsb)
    if low <= irsb_low_pc and irsb_high_pc <= high:
      yield i, irsb

def accumulate_tmp_var_expr(tmp: int, raw_dict: Dict[int, IRExpr]) -> IRExpr:
  # Adding helper to track depth
  def helper(depth: int, tmp: int, raw_dict: Dict[int, IRExpr]) -> IRExpr:
    if tmp in raw_dict:
      rhs = copy.deepcopy(raw_dict[tmp])
      # rhs = raw_dict[tmp]
      for child_expr in rhs.child_expressions:
        if isinstance(child_expr, RdTmp):
          child_tmp = child_expr.tmp
          if depth > 0:
            child_rhs = helper(depth - 1, child_tmp, raw_dict)
            if child_rhs is None:
              return None
          else:
            return None
          rhs.replace_expression(child_expr, child_rhs)
      return rhs
    else:
      print(f"accum_tmp_rhs: t{tmp} not found")
      return None

  # We limit the depth to be 5
  return helper(5, tmp, raw_dict)

TmpVarExprDict = Dict[int, IRExpr]

def tmp_var_expr_dict(irsb: IRSB) -> TmpVarExprDict:
  raw_dict = { stmt.tmp: stmt.data for stmt in irsb.statements if isinstance(stmt, WrTmp) }
  red_dict = {}
  for tmp in raw_dict.keys():
    accum_expr = accumulate_tmp_var_expr(tmp, raw_dict)
    if accum_expr is None:
      continue
    if const_fold.reduceable(accum_expr):
      red_expr = const_fold.reduce_expr(accum_expr)
    else:
      red_expr = accum_expr
    if red_expr:
      red_dict[tmp] = red_expr
  return red_dict

def get_or_create_expr_dict(i: int, irsb: IRSB, cache: Dict[int, TmpVarExprDict]) -> TmpVarExprDict:
  if not i in cache:
    expr_dict = tmp_var_expr_dict(irsb)
    cache[i] = expr_dict
  return cache[i]

def find_tmp_var_with_var_loc(var_loc: VarLocation, expr_dict: TmpVarExprDict, arch: Arch) -> Iterator[int]:
  for tmp, expr in expr_dict.items():
    reg_offset = const_fold.get_reg_offset(expr, arch)
    if reg_offset:
      (reg, offset) = reg_offset
      if arch.qemu_name == "x86_64" or arch.qemu_name == "x64" or arch.qemu_name == "x86":
        if reg == "rsp" and var_loc.offset == offset - 8:
          yield tmp
        elif reg == "rbp" and var_loc.offset == offset - 16:
          yield tmp
      elif arch.qemu_name == "i386":
        if reg == "esp" and var_loc.offset == offset - 4:
          yield tmp
        elif reg == "ebp" and var_loc.offset == offset - 8:
          yield tmp
      elif arch.qemu_name == "aarch64":
        if reg == "xsp" and var_loc.offset == offset:
          yield tmp
      elif arch.qemu_name == "arm" and arch.bits == 32:
        if reg == "sp" and var_loc.offset == offset:
          yield tmp
      else:
        print(f"Unregonized arch {arch.qemu_name}")

def find_tmp_var_within_range(irsbs: List[IRSB], cache: Dict[int, Dict[int, IRExpr]], low_high_pc: Tuple[int, int], var_loc: VarLocation) -> Iterator[Tuple[IRSB, TmpVar]]:
  low_pc, high_pc = low_high_pc
  for i, irsb in iter_irsbs_in_range(base_addr + low_pc, base_addr + high_pc, irsbs):
    expr_dict = get_or_create_expr_dict(i, irsb, cache)
    for tmp_var in find_tmp_var_with_var_loc(var_loc, expr_dict, irsb.arch):
      yield irsb, tmp_var
