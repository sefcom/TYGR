from typing import Tuple, Optional, Iterator, Dict, Any

from elftools.elf.elffile import *
from elftools.dwarf.die import *
from elftools.dwarf.dwarfinfo import *
from elftools.dwarf.locationlists import LocationParser
from elftools.dwarf.descriptions import ExprDumper, describe_form_class

from .location import *
from .utils import *
from .types import type_die_to_dwarf_type, dwarf_type_to_type
from ..types import Type
from .. import types as our_types

DwarfVariable = Tuple[
  Optional[str], # Directory
  Optional[str], # File name
  Optional[str], # Function name
  Tuple[int, int], # Low, High PC of the function
  Optional[str], # Variable name
  LocationList,
  Type, # Type
]

DwarfSubprogram = Tuple[
  Optional[str], # Directory
  Optional[str], # File name
  Optional[str], # Function name
  Tuple[int, int], # Low, High PC of the function
  int, # CU Offset
  DIE, # Subprogram DIE
]

DwarfVariable2 = Tuple[
  Optional[str], # Variable name
  LocationList, # Dwarf Location
  Type, # Type
]

DwarfContext = Tuple[
  LocationParser,
  ExprDumper,
  Dict[int, Dict[int, DIE]],
]

def get_type_die_dict(die: DIE, cu_offset: int) -> Dict[int, DIE]:
  return { (type_die.offset - cu_offset): type_die for type_die in iter_type_die(die) }

def subprog_die_is_base_offset(subprog_die: DIE, loc_parser: LocationParser, expr_dumper: ExprDumper) -> bool:
  if "DW_AT_frame_base" not in subprog_die.attributes:
    return None

  frame_base_attr = subprog_die.attributes["DW_AT_frame_base"]
  loc = loc_parser.parse_from_attribute(frame_base_attr, subprog_die.cu["version"])
  parsed_locs = expr_dumper.expr_parser.parse_expr(loc.loc_expr)

  if len(parsed_locs) == 0:
    return None

  loc0 = parsed_locs[0]
  return loc0.op_name in {"DW_OP_call_frame_cfa", "DW_OP_reg6"}  # TODO: DW_OP_reg6 is rbp on AMD64 DWARF

def subprog_die_to_stack_var_dies(subprog_die: DIE, loc_parser: LocationParser, expr_dumper: ExprDumper) -> Iterator[DIE]:
  subprog_is_base_offset = subprog_die_is_base_offset(subprog_die, loc_parser, expr_dumper)
  if subprog_is_base_offset:
    for die in iter_die(subprog_die):
      if die.tag == "DW_TAG_variable" or die.tag == "DW_TAG_formal_parameter":
        yield die

def stack_var_die_to_type(stack_var_die: DIE, type_die_dict: Dict[int, DIE]) -> Optional[Type]:
  type_die_id = get_die_attribute(stack_var_die, "DW_AT_type")
  if type_die_id:
    dwarf_type = type_die_to_dwarf_type(type_die_dict[type_die_id], type_die_dict)
    ## exception handle for bitvictor size float 96
    try:
      return dwarf_type_to_type(dwarf_type)
    except:
      return None
  else:
    return None

def subprog_die_to_vars(subprog_die: DIE, loc_parser: LocationParser, expr_dumper: ExprDumper, type_die_dict: Dict[int, DIE], dwarf_info: DWARFInfo) -> Iterator[Tuple[Optional[str], LocationList, Type]]:
  low_high_pc = subprog_die_low_high_pc(subprog_die, dwarf_info)
 
  for stack_var_die in subprog_die_to_stack_var_dies(subprog_die, loc_parser, expr_dumper):
    
    ## parse name & type first
    var_name = get_die_name(stack_var_die)
    var_type = stack_var_die_to_type(stack_var_die, type_die_dict)
    
    ## hanlde struct inner members
    if var_type == "struct":
      type_die_id = get_die_attribute(stack_var_die, "DW_AT_type")
      if type_die_id not in type_die_dict.keys():
        continue
      type_die = type_die_dict[type_die_id]
      
      for child_die in type_die.iter_children():
        if child_die.tag == "DW_TAG_member":     
          member_locs = struct_die_location_list(child_die, stack_var_die, loc_parser, expr_dumper, low_high_pc)
          inner_type = stack_var_die_to_type(child_die, type_die_dict)
          if inner_type is None:
            continue
          
          member_type = our_types.StructMember(inner_type)
          # member_type = inner_type
          
          ## filter condition
          if member_type is None or (isinstance(member_locs, list) and len(member_locs) == 0):
            continue
          if isinstance(var_name, bytes):
            var_name = var_name.decode("utf-8")        
        
          ## function parameter capture  
          func_param = False
          if stack_var_die.tag == "DW_TAG_formal_parameter":
              func_param = True
          yield (var_name, member_locs, member_type, func_param)
          
      ## skip struct type
      continue
    
    ## handle other types
    try:
      var_locs = die_location_list(stack_var_die, loc_parser, expr_dumper, low_high_pc)
    except:
      print("var_loc exception")
      continue
    
    ## function parameter capture
    func_param = False
    if stack_var_die.tag == "DW_TAG_formal_parameter":
      func_param = True

    ## filter condition
    if var_type is None or (isinstance(var_locs, list) and len(var_locs) == 0):
      continue
    if isinstance(var_name, bytes):
      var_name = var_name.decode("utf-8")
    # print(var_name, var_locs, var_type)
    yield (var_name, var_locs, var_type, func_param)

def cu_die_to_vars(cu_die: DIE, loc_parser: LocationParser, expr_dumper: ExprDumper, type_die_dict: Dict[int, DIE], dwarf_info: DWARFInfo) -> Iterator[DwarfVariable]:
  directory = cu_die.attributes["DW_AT_comp_dir"].value
  file_name = get_die_name(cu_die)
  for subprog_die in iter_subprogram_die(cu_die):
    func_name = get_die_name(subprog_die)
    low_high_pc = subprog_die_low_high_pc(subprog_die, dwarf_info)
    if low_high_pc is None:
      continue
    for var_name, var_locs, var_type, func_param in subprog_die_to_vars(subprog_die, loc_parser, expr_dumper, type_die_dict, dwarf_info):
      if isinstance(directory, bytes):
        directory = directory.decode("utf-8")
      if isinstance(file_name, bytes):
        file_name = file_name.decode("utf-8")
      if isinstance(func_name, bytes):
        func_name = func_name.decode("utf-8")
      yield (directory, file_name, func_name, low_high_pc, var_name, var_locs, var_type, func_param)

def dwarf_info_to_vars(dwarf_info: DWARFInfo) -> Iterator[DwarfVariable]:
  loc_parser = LocationParser(dwarf_info.location_lists())
  expr_dumper = ExprDumper(dwarf_info.structs)
  for cu in dwarf_info.iter_CUs():
    cu_die = cu.get_top_DIE()
    type_die_dict = get_type_die_dict(cu_die, cu.cu_offset)
    for var_info in cu_die_to_vars(cu_die, loc_parser, expr_dumper, type_die_dict, dwarf_info):
      yield var_info

def cu_die_to_subprograms(cu_offset: int, cu_die: DIE, dwarf_info: DWARFInfo) -> Iterator[DwarfSubprogram]:
  directory = cu_die.attributes["DW_AT_comp_dir"].value
  file_name = get_die_name(cu_die)
  for subprog_die in iter_subprogram_die(cu_die):
    func_name = get_die_name(subprog_die)
    low_high_pc = subprog_die_low_high_pc(subprog_die, dwarf_info)
    if low_high_pc is None:
      continue
    if isinstance(directory, bytes):
      directory = directory.decode("utf-8")
    if isinstance(file_name, bytes):
      file_name = file_name.decode("utf-8")
    if isinstance(func_name, bytes):
      func_name = func_name.decode("utf-8")
    yield (directory, file_name, func_name, low_high_pc, cu_offset, subprog_die)

def dwarf_info_to_subprograms(dwarf_info: DWARFInfo) -> Iterator[DwarfSubprogram]:
  for cu in dwarf_info.iter_CUs():
    cu_die = cu.get_top_DIE()
    for subprog_info in cu_die_to_subprograms(cu.cu_offset, cu_die, dwarf_info):
      yield subprog_info

def dwarf_info_to_context(dwarf_info: DWARFInfo) -> DwarfContext:
  loc_parser = LocationParser(dwarf_info.location_lists())
  expr_dumper = ExprDumper(dwarf_info.structs)
  type_dicts = { cu.cu_offset: get_type_die_dict(cu.get_top_DIE(), cu.cu_offset) for cu in dwarf_info.iter_CUs() }
  return (loc_parser, expr_dumper, type_dicts)

def dwarf_subprogram_to_vars(dwarf_subprog: DwarfSubprogram, dwarf_ctx: DwarfContext, dwarf_info: DWARFInfo) -> Iterator[DwarfVariable2]:
  (loc_parser, expr_dumper, type_die_dicts) = dwarf_ctx
  (_, _, _, _, cu_offset, subprog_die) = dwarf_subprog
  type_die_dict = type_die_dicts[cu_offset]
  for (var_name, var_locs, var_type, func_param) in subprog_die_to_vars(subprog_die, loc_parser, expr_dumper, type_die_dict, dwarf_info):
    yield (var_name, var_locs, var_type, func_param)

