from typing import *

from elftools.dwarf.dwarf_expr import *
from elftools.dwarf.locationlists import *
from elftools.dwarf.descriptions import *
from elftools.dwarf.die import *
from elftools.dwarf.dwarfinfo import *

class DwarfLocation: pass

class CfaLocation(DwarfLocation):
  def __init__(self, arg: int):
    self.arg = arg

  def __str__(self):
    return f"cfa {self.arg}"

  def __repr__(self):
    return f"CfaLocation({self.arg})"

class RegLocation(DwarfLocation):
  def __init__(self, reg_num: int):
    self.reg_num = reg_num
    self.reg_name = describe_reg_name(reg_num)

  def __str__(self):
    return f"reg{self.reg_num} ({self.reg_name})"

  def __repr__(self):
    return f"RegLocation({self.reg_num})"

class BregLocation(DwarfLocation):
  def __init__(self, reg_num: int, arg: int):
    self.reg_num = reg_num
    self.reg_name = describe_reg_name(reg_num)
    self.arg = arg

  def __str__(self):
    return f"breg{self.reg_num} ({self.reg_name}) {self.arg}"

  def __repr__(self):
    return f"BregLocation({self.reg_num}, {self.arg})"

class AddrLocation(DwarfLocation):
  def __init__(self, arg: int):
    self.arg = arg

  def __str__(self):
    return f"addr {self.arg}"

  def __repr__(self):
    return f"AddrLocation({self.arg})"

class StackValueLocation(DwarfLocation):
  def __str__(self):
    return f"stack_value"

  def __repr__(self):
    return f"StackValueLocation()"

LocationTriple = Tuple[int, int, DwarfLocation]
LocationList = List[LocationTriple]

#

def subprog_die_low_high_pc(subprog_die : DIE , dwarf_info: DWARFInfo):

  # extract low and high pc from ranges
  # if range exist, we will not need low/high pc
  if "DW_AT_ranges" in subprog_die.attributes:
    # print("DW_AT_ranges")
    range_lists = dwarf_info.range_lists()
    ranges_offset = subprog_die.attributes["DW_AT_ranges"].value
    ranges = range_lists.get_range_list_at_offset(ranges_offset)

    range_tuples = [(r.begin_offset, r.end_offset) for r in ranges]
    low_pc = min(range_tuples)[0]
    high_pc = max(range_tuples)[1]
    # print(low_pc, high_pc)
    return (low_pc, high_pc)
  
  if "DW_AT_low_pc" not in subprog_die.attributes:
    return None

  # Extract low and high pc
  low_pc = subprog_die.attributes["DW_AT_low_pc"].value
  highpc_attr = subprog_die.attributes["DW_AT_high_pc"]
  highpc_attr_class = describe_form_class(highpc_attr.form)
  if highpc_attr_class == "address":
    high_pc = highpc_attr.value
  elif highpc_attr_class == "constant":
    high_pc = low_pc + highpc_attr.value
  else:
    raise Exception(f"Invalid 'DW_AT_high_pc' class {highpc_attr_class}")
  return (low_pc, high_pc)


def parse_dwarf_expr_op(dwarf_expr_op : DWARFExprOp, machine_arch):
  op_name = dwarf_expr_op.op_name
  args = dwarf_expr_op.args

  if op_name == "DW_OP_stack_value":
    return StackValueLocation()

  elif op_name.startswith("DW_OP_fbreg") and len(args) == 1:
    return CfaLocation(args[0])

  # Technically breg6 means "the value held in reg6",
  # but in this case we know what it corresponds to for x64 machine
  elif op_name == "DW_OP_breg6" and machine_arch == "x64" and len(args) == 1:
    return CfaLocation(args[0])

  # Similarly for breg5 on x86
  elif op_name == "DW_OP_breg5" and machine_arch == "x86" and len(args) == 1:
    return CfaLocation(args[0])

  # Generic breg handling
  elif op_name.startswith("DW_OP_breg") and len(args) == 1:
    reg_num = int("".join([d for d in op_name if d.isdigit()]))
    return BregLocation(reg_num, args[0])

  elif op_name.startswith("DW_OP_regx") and len(args) == 1:
    return RegLocation(args[0])

  elif op_name.startswith("DW_OP_reg") and len(args) == 0:
    reg_num = int("".join([d for d in op_name if d.isdigit()]))
    return RegLocation(reg_num)

  elif op_name.startswith("DW_OP_addr") and len(args) == 1:
    return AddrLocation(args[0])


def die_location_list(die : DIE, loc_parser, expr_dumper : ExprDumper, subprog_low_high_pc):

  def parse_loc_expr(loc_expr):
    try:
      dw_expr_ops = expr_dumper.expr_parser.parse_expr(loc_expr)
    except KeyError:
      dw_expr_ops = None
      
    if dw_expr_ops is not None and len(dw_expr_ops) > 0:
      return dw_expr_ops

  machine_arch = die.dwarfinfo.config.machine_arch
  set_global_machine_arch(machine_arch)

  if "DW_AT_location" not in die.attributes:
    return []

  loc_attr = die.attributes["DW_AT_location"]
  loc = loc_parser.parse_from_attribute(loc_attr, die.cu["version"])
  (low_pc, high_pc) = subprog_low_high_pc

  # A single expression suffices to describe the variable;
  # the min/max offset is therefore the low / high PC of the subprogram
  loc_list = []

  # We look for DWARFExprOp lists of only 1 element;
  # This guarantees we can avoid those DW_OP_stack_value ones which don't exist
  if isinstance(loc, LocationExpr):
    dw_expr_ops = parse_loc_expr(loc.loc_expr)
    if dw_expr_ops is not None and len(dw_expr_ops) == 1:
      dwloc = parse_dwarf_expr_op(dw_expr_ops[0], machine_arch)
      if (isinstance(dwloc, CfaLocation) or isinstance(dwloc, BregLocation) or
          isinstance(dwloc, RegLocation) or isinstance(dwloc, AddrLocation)):
        loc_list.append((low_pc, high_pc, dwloc))

  elif isinstance(loc, list):
    for elem in loc:
      if isinstance(elem, LocationEntry):
        dw_expr_ops = parse_loc_expr(elem.loc_expr)
        if dw_expr_ops is not None and len(dw_expr_ops) == 1:
          dwloc = parse_dwarf_expr_op(dw_expr_ops[0], machine_arch)
          if (isinstance(dwloc, CfaLocation) or isinstance(dwloc, BregLocation) or
              isinstance(dwloc, RegLocation) or isinstance(dwloc, AddrLocation)):
            maybe_low_pc = low_pc if elem.begin_offset < low_pc else 0
            begin_addr = maybe_low_pc + elem.begin_offset
            end_addr = maybe_low_pc + elem.end_offset
            loc_list.append((begin_addr, end_addr, dwloc))

  return loc_list


def struct_die_location_list(child_die: DIE, die: DIE, loc_parser, expr_dumper: ExprDumper, subprog_low_high_pc):
  def parse_loc_expr(loc_expr):
    try:
      dw_expr_ops = expr_dumper.expr_parser.parse_expr(loc_expr)
    except KeyError:
      dw_expr_ops = None

    if dw_expr_ops is not None and len(dw_expr_ops) > 0:
      return dw_expr_ops

  machine_arch = die.dwarfinfo.config.machine_arch
  set_global_machine_arch(machine_arch)

  if "DW_AT_location" not in die.attributes or "DW_AT_data_member_location" not in child_die.attributes:
    return []

  loc_attr = die.attributes["DW_AT_location"]
  loc = loc_parser.parse_from_attribute(loc_attr, die.cu["version"])
  loc_offset = child_die.attributes["DW_AT_data_member_location"].value
  
  if isinstance(loc, LocationExpr):
    dw_expr_ops = parse_loc_expr(loc.loc_expr)
    # print(dw_expr_ops)
    ## filter out dwarf expression with no args
    if len(dw_expr_ops[0].args) == 0:
      return []
    dw_expr_ops[0].args[0] += loc_offset
  (low_pc, high_pc) = subprog_low_high_pc

  # A single expression suffices to describe the variable;
  # the min/max offset is therefore the low / high PC of the subprogram
  loc_list = []

  # We look for DWARFExprOp lists of only 1 element;
  # This guarantees we can avoid those DW_OP_stack_value ones which don't exist
  if isinstance(loc, LocationExpr):
    # dw_expr_ops = parse_loc_expr(loc.loc_expr)
    if dw_expr_ops is not None and len(dw_expr_ops) == 1:
      dwloc = parse_dwarf_expr_op(dw_expr_ops[0], machine_arch)
      if (isinstance(dwloc, CfaLocation) or isinstance(dwloc, BregLocation) or
              isinstance(dwloc, RegLocation) or isinstance(dwloc, AddrLocation)):
        loc_list.append((low_pc, high_pc, dwloc))

  elif isinstance(loc, list):
    for elem in loc:
      if isinstance(elem, LocationEntry):
        dw_expr_ops = parse_loc_expr(elem.loc_expr)
        if dw_expr_ops is not None and len(dw_expr_ops) == 1:
          dwloc = parse_dwarf_expr_op(dw_expr_ops[0], machine_arch)
          if (isinstance(dwloc, CfaLocation) or isinstance(dwloc, BregLocation) or
                  isinstance(dwloc, RegLocation) or isinstance(dwloc, AddrLocation)):
            maybe_low_pc = low_pc if elem.begin_offset < low_pc else 0
            begin_addr = maybe_low_pc + elem.begin_offset
            end_addr = maybe_low_pc + elem.end_offset
            loc_list.append((begin_addr, end_addr, dwloc))

  return loc_list


############################

# The stuff below is outdated and should probably be deleted

# Old! Making it a sub-class so any type checking won't fail for legacy code
class VarLocation(DwarfLocation):
  def __init__(self, offset: int):
    self.offset = offset

  def __eq__(self, other: Any):
    return isinstance(other, VarLocation) and other.offset == self.offset

  def __repr__(self):
    return str(self)

  def __str__(self):
    return f"fbreg {hex(self.offset)}"

  def __hash__(self):
    return hash(str(self))

def stack_var_die_to_cfa_location(die: DIE, loc_parser: LocationParser, expr_dumper: ExprDumper) -> Optional[VarLocation]:
  # DWARF-specific parsing
  def parse_expr_loc(loc_expr) -> int:
    parsed_locs = expr_dumper.expr_parser.parse_expr(loc_expr)
    if len(parsed_locs) == 0:
      return None

    loc0 = parsed_locs[0]
    op_name = loc0.op_name
    args = loc0.args

    machine_arch = die.dwarfinfo.config.machine_arch

    if op_name == "DW_OP_fbreg" and len(args) == 1:
      return VarLocation(args[0])

    # x86's ebp
    elif machine_arch == "x86":
      if op_name == "DW_OP_breg5" and len(args) == 1:
        return VarLocation(args[0] + 8)
      else:
        return None

    # x64's rbp
    elif machine_arch == "x64":
      if op_name == "DW_OP_breg6" and len(args) == 1:
        return VarLocation(args[0] + 16)
      else:
        return None

    else:
      return None

  # Set the location attribute so that the lookup works correctly
  set_global_machine_arch(die.dwarfinfo.config.machine_arch)
  dwarf_version = die.cu["version"]

  # This thing better have a DW_AT_location
  if "DW_AT_location" not in die.attributes:
    return None

  loc_attr = die.attributes["DW_AT_location"]
  loc = loc_parser.parse_from_attribute(loc_attr, dwarf_version)
  if isinstance(loc, LocationExpr):
    return parse_expr_loc(loc.loc_expr)

  elif isinstance(loc, list):
    for elem in loc:
      if isinstance(elem, LocationEntry):
        parsed = parse_expr_loc(elem.loc_expr)
        if parsed is not None:
          return parsed
    return None
  else:
    return None

