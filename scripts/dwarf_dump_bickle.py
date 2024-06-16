'''
  python3 /path/to/dwarf_dump_bickle.py \
    -i /path/to/binary \
    -o /path/to/binary.bkl
'''

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import argparse

from typing import *

from elftools.elf.elffile import *
from elftools.dwarf.compileunit import *
from elftools.dwarf.dwarfinfo import *
from elftools.dwarf.die import *

from elftools.dwarf.descriptions import *
from elftools.dwarf.locationlists import *

import copy
import pickle

from analysis.dwarf import *
from analysis.dwarf.utils import *
from analysis.dwarf.location import *
from analysis.dwarf.lookup import *

# String constants
DW_TAG_variable = "DW_TAG_variable"
DW_TAG_formal_parameter = "DW_TAG_formal_parameter"
DW_AT_location = "DW_AT_location"

FORM_CLASS_address = "address"
FORM_CLASS_constant = "constant"

DW_TAG_array_type = "DW_TAG_array_type"
DW_TAG_base_type = "DW_TAG_base_type"
DW_TAG_class_type = "DW_TAG_class_type"
DW_TAG_const_type = "DW_TAG_const_type"
DW_TAG_enumeration_type = "DW_TAG_enumeration_type"
DW_TAG_pointer_type = "DW_TAG_pointer_type"
DW_TAG_structure_type = "DW_TAG_structure_type"
DW_TAG_union_type = "DW_TAG_union_type"
DW_TAG_volatile_type = "DW_TAG_volatile_type"
DW_TAG_typedef = "DW_TAG_typedef"

DW_AT_name = "DW_AT_name"
DW_AT_type = "DW_AT_type"
DW_AT_encoding = "DW_AT_encoding"
DW_AT_byte_size = "DW_AT_byte_size"
DW_AT_bit_size = "DW_AT_bit_size"

# Base Type encodings, Section 7.8 Figure 25 of DwarfV4
base_type_tbl = {
  0x01 : "address",
  0x02 : "boolean",
  0x03 : "complex_float",
  0x04 : "float",
  0x05 : "signed",
  0x06 : "signed_char",
  0x07 : "unsigned",
  0x08 : "unsigned_char",
  0x09 : "imaginary_float",
  0x0a : "packed_decimal",
  0x0b : "numeric_string",
  0x0c : "edited",
  0x0d : "signed_fixed",
  0x0e : "unsigned_fixed",
  0x0f : "decimal_float",
  0x10 : "UTF",
  0x80 : "lo_user",
  0xff : "hi_user"
}

DieDict = Dict[int, DIE]

def iter_children_with_tag(root_die : DIE, tag : str):
  for die in root_die.iter_children():
    if die.tag == tag:
      yield die

def get_die_type_die(die: DIE, die_dict: DieDict) -> DIE:
  if DW_AT_type in die.attributes:
    key = die.attributes[DW_AT_type].value
    return die_dict[key]
  else:
    return None

def get_bitsize(die: DIE) -> int:
  bitsize = get_die_attribute(die, DW_AT_bit_size)
  bytesize = get_die_attribute(die, DW_AT_byte_size)
  if bitsize is not None:
    return bitsize
  elif bytesize is not None:
    return bytesize * 8
  else:
    return bitsize

def get_btype(type_die : DIE, die_dict : DieDict):
  if type_die is None: return None

  tag = type_die.tag

  # Check that this is indeed a type DIE
  if not (tag.endswith("type") or tag.endswith("typedef")): return None

  # Pre-calculate some things; None if non-existent
  name = get_die_attribute(type_die,DW_AT_name)
  if name is not None: name = name.decode()

  # Recursion step
  inner_offset = get_die_attribute(type_die, DW_AT_type)
  inner_btype = None
  if inner_offset is not None:
    inner_type_die = die_dict.get(inner_offset)
    inner_btype = get_btype(inner_type_die, die_dict)

  if tag == DW_TAG_base_type:
    encoding = get_die_attribute(type_die, DW_AT_encoding)
    bitsize = get_bitsize(type_die)
    return (("base", base_type_tbl[encoding]), ("bitsize", bitsize))

  elif tag == DW_TAG_const_type:
    return (("const", inner_btype),)

  elif tag == DW_TAG_pointer_type:
    return (("pointer", inner_btype),)

  elif tag == DW_TAG_volatile_type:
    return (("volatile", inner_btype),)

  elif tag == DW_TAG_array_type:
    return (("array", inner_btype),)

  elif tag == DW_TAG_union_type:
    return (("union", name),)

  elif tag == DW_TAG_enumeration_type:
    return (("enum", name),)

  elif tag == DW_TAG_structure_type:
    return (("struct", name),)

  elif tag == DW_TAG_class_type:
    return (("class", name),)

  elif tag == DW_TAG_typedef:
    return (("typedef", inner_btype),)

  # Failed to find anything that we can represent
  else:
    return None

# Insert sometihng to a set-valued dictionary
def add_to_set_valued(d, k, v):
  vals = d.get(k)
  new_vals = (vals if vals else set()) | {v}
  d[k] = new_vals

def generate_btype_dict(dwarf_info : DWARFInfo):
  loc_parser = LocationParser(dwarf_info.location_lists())
  expr_dumper = ExprDumper(dwarf_info.structs)

  var_dict = {}

  for cu in dwarf_info.iter_CUs():
    # Maps low_pc -> btype_dict

    cu_die = cu.get_top_DIE()
    die_dict = get_type_die_dict(cu_die, cu.cu_offset)

    # for subprog_die in iter_subprogram_dies(cu_die):
    for subprog_die in iter_subprogram_die(cu_die):
      # Maps stack_offset / parameter -> btype

      subprog_low_high_pc = subprog_die_low_high_pc(subprog_die)

      if subprog_low_high_pc is None:
        continue

      (subprog_low_pc, subprog_high_pc) = subprog_low_high_pc

      if subprog_low_pc is None: continue

      btype_dict = dict()
      if subprog_low_pc in var_dict:
        btype_dict = var_dict[subprog_low_pc]

      if subprog_low_pc is not None:
        for var_die in iter_children_with_tag(subprog_die, DW_TAG_variable):
          type_die = get_die_type_die(var_die, die_dict)
          btype = get_btype(type_die, die_dict)

          loc_list = die_location_list(var_die, loc_parser, expr_dumper, subprog_low_high_pc)

          for (low_pc, high_pc, dwloc) in loc_list:
            # Does it span the entire subprogram?
            if low_pc == subprog_low_pc and high_pc == subprog_high_pc:
              high_pc = None

            if isinstance(dwloc, CfaLocation):
              add_to_set_valued(btype_dict, ("cfa", dwloc.arg), (low_pc, high_pc, btype))

            elif isinstance(dwloc, RegLocation):
              add_to_set_valued(btype_dict, ("reg", dwloc.reg_num), (low_pc, high_pc, btype))

            elif isinstance(dwloc, AddrLocation):
              add_to_set_valued(btype_dict, ("addr", dwloc.arg), (low_pc, high_pc, btype))

          # print(f"loc list is {loc_list}")

        ''' # Don't dump params since IDA can't easily relate them to an offset
        # According to DWARF 4, params appear in order of declaration!
        param_dies = list(iter_children_with_tag(subprog_die, DW_TAG_formal_parameter))
        for i, param_die in enumerate(param_dies):
          print("encountered param!")
          type_die = get_die_type_die(param_die, die_dict)
          btype = get_btype(type_die, die_dict)
          param_off = "param%s" % i
          btype_dict[param_off] = btype
        '''

      var_dict[subprog_low_pc] = copy.deepcopy(btype_dict)

  return var_dict


def parser():
  parser = argparse.ArgumentParser(description="Extract ground truths from DWARF")
  parser.add_argument("-i", "--input_file", type=str, required=True)
  parser.add_argument("-o", "--output_file", type=str, required=True)
  return parser

args = parser().parse_args()

dwinfo = get_dwarf_info(args.input_file)
var_dict = generate_btype_dict(dwinfo)
with open(args.output_file, "wb") as f:
  pickle.dump(var_dict, f)
  f.close()

