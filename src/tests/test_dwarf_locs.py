import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import angr

from elftools.dwarf.dwarf_expr import *
from elftools.dwarf.locationlists import *
from elftools.dwarf.descriptions import *
from elftools.dwarf.die import *

from analysis.dwarf import *
from analysis.dwarf.location import *

def parser():
  parser = argparse.ArgumentParser(description="Test dwarf lookup")
  parser.add_argument("-i", "--input_file", type=str, required=True)
  parser.add_argument("-a", "--addr", type=str)
  return parser

def subprog_die_to_var_dies(subprog_die):
  for die in subprog_die.iter_children():
    if die.tag == "DW_TAG_variable" or die.tag == "DW_TAG_formal_parameter":
      yield die

def parse_dwarf_expr_op(dw_expr_op):
  if not isinstance(dw_expr_op, DWARFExprOp):
    return None

  op_name = dw_expr_op.op_name
  args = dw_expr_op.args

  if op_name == "DW_OP_stack_value":
    return "stack_value"

  elif op_name.startswith("DW_OP_fbreg") and len(args) == 1:
    return ("fbreg", args[0])

  elif op_name.startswith("DW_OP_regx") and len(args) == 1:
    return ("reg", args[0])

  elif op_name.startswith("DW_OP_reg") and len(args) == 0:
    regnum = int("".join([d for d in op_name if d.isdigit()]))
    return ("reg", regnum)

  elif op_name.startswith("DW_OP_breg") and len(args) == 1:
    regnum = int("".join([d for d in op_name if d.isdigit()]))
    return ("breg", regnum, args[0])

  elif op_name.startswith("DW_OP_addr"):
    return ("addr", args[0])


def die_locations(die, loc_parser, expr_dumper):
  def parse_loc_expr(loc_expr):
    parsed_locs = expr_dumper.expr_parser.parse_expr(loc_expr)
    if len(parsed_locs) == 0:
      return None

    else:
      return parsed_locs

  set_global_machine_arch(die.dwarfinfo.config.machine_arch)
  dwarf_version = die.cu["version"]

  if "DW_AT_location" not in die.attributes:
    return None

  loc_attr = die.attributes["DW_AT_location"]
  loc = loc_parser.parse_from_attribute(loc_attr, dwarf_version)
  if isinstance(loc, LocationExpr):
    yield (0, 0, parse_loc_expr(loc.loc_expr))

  elif isinstance(loc, list):
    for elem in loc:
      if isinstance(elem, LocationEntry):
        parsed = parse_loc_expr(elem.loc_expr)
        if parsed is not None:
          begin_off = elem.begin_offset
          end_off = elem.end_offset
          yield (begin_off, end_off, parsed)


# Run the stuff

args = parser().parse_args()

# Initialize things
dwinfo = get_dwarf_info(args.input_file)
dwctx = dwarf_info_to_context(dwinfo)
(loc_parser, expr_dumper, type_die_dicts) = dwctx

if args.addr:
  tgt_addr = int(args.addr, 16)
else:
  tgt_addr = None


parseds = []

# Iterate through the subprograms
for subprog in dwarf_info_to_subprograms(dwinfo):
  (directory, file_name, func_name, low_high_pc, cu_offset, subprog_die) = subprog
  (low_pc, high_pc) = low_high_pc
  if args.addr is None or tgt_addr == low_pc:
    print(subprog_die)
    for var_die in subprog_die_to_var_dies(subprog_die):
      var_name_attr = var_die.attributes.get("DW_AT_name")
      if var_name_attr is not None:
        var_name = var_name_attr.value
        print(f"{var_name}")
        for (begin_addr, end_addr, dwloc) in die_location_list(var_die, loc_parser, expr_dumper, low_high_pc):
          print(f"\t{hex(begin_addr)}, {hex(end_addr)}, {dwloc}")


def test(parseds):
  for expr in parseds:
    print(f"{parse_dwarf_expr_op(expr)} \t\t {expr}")

