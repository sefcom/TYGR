from typing import *

from elftools.dwarf.die import *

from .utils import *
from .. import types as our_types

# Base Type encodings, Section 7.8 Figure 25 of DwarfV4
base_type_tbl = {
  0x01: "address",
  0x02: "boolean",
  0x03: "complex_float",
  0x04: "float",
  0x05: "signed",
  0x06: "signed_char",
  0x07: "unsigned",
  0x08: "unsigned_char",
  0x09: "imaginary_float",
  0x0a: "packed_decimal",
  0x0b: "numeric_string",
  0x0c: "edited",
  0x0d: "signed_fixed",
  0x0e: "unsigned_fixed",
  0x0f: "decimal_float",
  0x10: "UTF",
  0x80: "lo_user",
  0xff: "hi_user"
}

class DwarfType(object):
  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

  def __repr__(self):
    return str(self)

class BaseType(DwarfType):
  def __init__(self, encoding: str, bitsize: int):
    DwarfType.__init__(self)
    self.encoding = encoding
    self.bitsize = bitsize

  def encoding_str(self):
    if self.encoding in base_type_tbl:
      return base_type_tbl[self.encoding]
    else:
      return self.encoding

  def __str__(self):
    return f"BaseType({self.encoding_str()}, {self.bitsize})"

# Unspecified type -- kinda like void
class UnspecifiedType(DwarfType):
  def __init__(self, name: str):
    self.name = name

# Type tag modifiers
class ConstType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class PackedType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class PointerType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class ReferenceType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class RestrictType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class RValueReferenceType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class SharedType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

class VolatileType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

# Array
class ArrayType(DwarfType):
  def __init__(self, _type: DwarfType):
    self._type = _type

# Struct, union, and class
class StructType(DwarfType):
  def __init__(self, name: str, members: List[Tuple[str, int]]):
    self.name = name
    self.members = members

class UnionType(DwarfType):
  def __init__(self, name: str, members: List[Tuple[str, int]]):
    self.name = name
    self.members = members

class ClassType(DwarfType):
  def __init__(self, name: str, members: List[Tuple[str, int]]):
    self.name = name
    self.members = members

class InterfaceType(DwarfType):
  def __init__(self, name: str, members: List[Tuple[str, int]]):
    self.name = name
    self.members = members

# Enumeration
class EnumType(DwarfType):
  def __init__(self, name: str):
    self.name = name

# Subroutine
class SubroutineType(DwarfType):
  def __init__(self, name: str, _type: DwarfType):
    self.name = name
    self._type = _type

# String type; does not appear in C, but does so in Fortran
class StringType(DwarfType):
  def __init__(self, name: str):
    self.name = name

# Set type; appears in PASCAL
class SetType(DwarfType):
  def __init__(self, name: str):
    self.name = name

# Subrange type
class SubrangeType(DwarfType):
  def __init__(self, name: str):
    self.name = name

# Pointer to member type
class PointerToMemberType(DwarfType):
  def __init__(self, name: int, _type: DwarfType):
    self.name = name
    self._type = _type

# File types, present in Pascal
class FileType(DwarfType):
  def __init__(self, name: int, _type: DwarfType):
    self.name = name
    self._type = _type

# Include typedefs
class TypedefType(DwarfType):
  def __init__(self, name: int, _type: DwarfType):
    self.name = name
    self._type = _type

def get_bitsize(die: DIE) -> int:
  bitsize = get_die_attribute(die, "DW_AT_bit_size")
  bytesize = get_die_attribute(die, "DW_AT_byte_size")
  if bitsize is not None:
    return bitsize
  elif bytesize is not None:
    return bytesize * 8
  else:
    return bitsize

def get_child_members(comp_type_die: DIE) -> List[Tuple[str, int]]:
  members = []
  for child_die in comp_type_die.iter_children():
    if child_die.tag == "DW_TAG_member":
      name = get_die_attribute(child_die, "DW_AT_name")
      type_offset = get_die_attribute(child_die, "DW_AT_type")
      members.append((name, type_offset))
  return members

def type_die_to_dwarf_type(type_die: DIE, die_dict: Dict[int, DIE]) -> Optional[Type]:
  if type_die == None:
    return None

  tag = type_die.tag

  # Check that this is a type DIE
  if not (tag.endswith("type") or tag.endswith("typedef")):
    return None

  # Pre-calculate some things; None if non-existent
  bitsize = get_bitsize(type_die)
  encoding = get_die_attribute(type_die, "DW_AT_encoding")
  name = get_die_attribute(type_die, "DW_AT_name")

  # Recursion step
  inner_offset = get_die_attribute(type_die, "DW_AT_type")
  inner_type = None
  if inner_offset is not None:
    inner_type_die = die_dict.get(inner_offset)
    inner_type = type_die_to_dwarf_type(inner_type_die, die_dict)

  # Base type
  if tag == "DW_TAG_base_type": return BaseType(encoding, bitsize)

  # Unspecified type
  elif tag == "DW_TAG_unspecified_type": return UnspecifiedType(name)

  # Type modifiers
  elif tag == "DW_TAG_const_type": return ConstType(inner_type)
  elif tag == "DW_TAG_packed_type": return PackedType(inner_type)
  elif tag == "DW_TAG_pointer_type": return PointerType(inner_type)
  elif tag == "DW_TAG_reference_type": return ReferenceType(inner_type)
  elif tag == "DW_TAG_restrict_type": return RestrictType(inner_type)
  elif tag == "DW_TAG_rval_reference_type": return RValueReferenceType(inner_type)
  elif tag == "DW_TAG_shared_type": return SharedType(inner_type)
  elif tag == "DW_TAG_volatile_type": return VolatileType(inner_type)

  # Array
  elif tag == "DW_TAG_array_type": return ArrayType(inner_type)

  # Struct, union, class, interface, enum
  elif tag == "DW_TAG_structure_type":
    return StructType(name, get_child_members(type_die))
  elif tag == "DW_TAG_union_type":
    return UnionType(name, get_child_members(type_die))
  elif tag == "DW_TAG_class_type":
    return ClassType(name, get_child_members(type_die))
  elif tag == "DW_TAG_interface_type":
    return InterfaceType(name, get_child_members(type_die))
  elif tag == "DW_TAG_enumeration_type": return EnumType(name)

  # Subroutine
  elif tag == "DW_TAG_subroutine_type": return SubroutineType(name, inner_type)

  # String, set, subrange
  elif tag == "DW_TAG_string_type": return StringType(name)
  elif tag == "DW_TAG_set_type": return SetType(name)
  elif tag == "DW_TAG_subrange_type": return SubrangeType(name)

  # Pointer-to-member
  elif tag == "DW_TAG_ptr_to_member_type": return PointerToMemberType(name, inner_type)

  # File
  elif tag == "DW_TAG_file_type": return FileType(name, inner_type)

  # Typedef
  elif tag == "DW_TAG_typedef" : return TypedefType(name, inner_type)

  # Return None otherwise
  else: return None

def dwarf_type_to_type(dwarf_type: DwarfType) -> our_types.Type:
  if isinstance(dwarf_type, BaseType):
    enc = dwarf_type.encoding_str()
    if enc == "boolean":
      return our_types.BooleanType()
    elif enc == "float":
      if dwarf_type.bitsize == 32:
        return our_types.F32Type()
      elif dwarf_type.bitsize == 64:
        return our_types.F64Type()
      elif dwarf_type.bitsize == 128:
        return our_types.F128Type()
      else:
        raise Exception(f"Unknown float type bitsize {dwarf_type.bitsize}")
    elif enc == "signed":
      if dwarf_type.bitsize == 1:
        return our_types.I1Type()
      elif dwarf_type.bitsize == 8:
        return our_types.I8Type()
      elif dwarf_type.bitsize == 16:
        return our_types.I16Type()
      elif dwarf_type.bitsize == 32:
        return our_types.I32Type()
      elif dwarf_type.bitsize == 64:
        return our_types.I64Type()
      elif dwarf_type.bitsize == 128:
        return our_types.I128Type()
      else:
        raise Exception(f"Unknown signed type bitsize {dwarf_type.bitsize}")
    elif enc == "signed_char":
      return our_types.CharType()
    elif enc == "unsigned":
      if dwarf_type.bitsize == 1:
        return our_types.U1Type()
      elif dwarf_type.bitsize == 8:
        return our_types.U8Type()
      elif dwarf_type.bitsize == 16:
        return our_types.U16Type()
      elif dwarf_type.bitsize == 32:
        return our_types.U32Type()
      elif dwarf_type.bitsize == 64:
        return our_types.U64Type()
      elif dwarf_type.bitsize == 128:
        return our_types.U128Type()
      else:
        raise Exception(f"Unknown unsigned type bitsize {dwarf_type.bitsize}")
    elif enc == "unsigned_char":
      return our_types.CharType()
    else:
      raise Exception(f"Unknown base type encoding {enc}")
  elif isinstance(dwarf_type, ConstType):
    return dwarf_type_to_type(dwarf_type._type)
  elif isinstance(dwarf_type, PackedType):
    return dwarf_type_to_type(dwarf_type._type)
  elif isinstance(dwarf_type, PointerType):
    if dwarf_type._type == None:
      return our_types.PointerType(our_types.VoidType())
    else:
      return our_types.PointerType(dwarf_type_to_type(dwarf_type._type))
  elif isinstance(dwarf_type, ReferenceType):
    return our_types.PointerType(dwarf_type_to_type(dwarf_type._type))
  elif isinstance(dwarf_type, RestrictType):
    return dwarf_type_to_type(dwarf_type._type)
  elif isinstance(dwarf_type, RValueReferenceType):
    return our_types.PointerType(dwarf_type_to_type(dwarf_type._type))
  elif isinstance(dwarf_type, ArrayType):
    return our_types.ArrayType(dwarf_type_to_type(dwarf_type._type))
  elif isinstance(dwarf_type, StructType):
    return our_types.StructType()
  elif isinstance(dwarf_type, UnionType):
    return our_types.UnionType()
  elif isinstance(dwarf_type, ClassType):
    return our_types.StructType()
  elif isinstance(dwarf_type, EnumType):
    return our_types.EnumType()
  elif isinstance(dwarf_type, TypedefType):
    return dwarf_type_to_type(dwarf_type._type)
