# Dictionary representation of bityr's types, so that we can compare against other tools

from typing import *
from .types import *

def type_to_btype(ty : Type) -> dict:

  # We can't exactly handle void types as-is
  if isinstance(ty, VoidType):
    return None

  elif isinstance(ty, StructType):
    return (("struct", "STRUCT"),)

  elif isinstance(ty, ArrayType):
    if isinstance(ty.elem, VoidType):
      return (("array", None),)

    inner_bty = type_to_btype(ty.elem)
    if inner_bty is not None:
      return (("array", inner_btype),)

  elif isinstance(ty, UnionType):
    return (("union", "UNION"),)

  elif isinstance(ty, EnumType):
    return (("enum", "ENUM"),)

  elif isinstance(ty, BooleanType):
    return (("base", "boolean"), ("bitsize", 8))

  elif isinstance(ty, I1Type):
    return (("base", "signed"), ("bitsize", 1))

  elif isinstance(ty, I8Type):
    return (("base", "signed"), ("bitsize", 8))

  elif isinstance(ty, I16Type):
    return (("base", "signed"), ("bitsize", 16))

  elif isinstance(ty, I32Type):
    return (("base", "signed"), ("bitsize", 32))

  elif isinstance(ty, I64Type):
    return (("base", "signed"), ("bitsize", 64))

  elif isinstance(ty, I128Type):
    return (("base", "signed"), ("bitsize", 128))

  elif isinstance(ty, U1Type):
    return (("base", "unsigned"), ("bitsize", 1))

  elif isinstance(ty, U8Type):
    return (("base", "unsigned"), ("bitsize", 8))

  elif isinstance(ty, U16Type):
    return (("base", "unsigned"), ("bitsize", 16))

  elif isinstance(ty, U32Type):
    return (("base", "unsigned"), ("bitsize", 32))

  elif isinstance(ty, U64Type):
    return (("base", "unsigned"), ("bitsize", 64))

  elif isinstance(ty, U128Type):
    return (("base", "unsigned"), ("bitsize", 128))

  elif isinstance(ty, CharType):
    return (("base", "signed_char"), ("bitsize", 8))

  elif isinstance(ty, F32Type):
    return (("base", "float"), ("bitsize", 32))

  elif isinstance(ty, F64Type):
    return (("base", "float"), ("bitsize", 64))

  elif isinstance(ty, F128Type):
    return (("base", "float"), ("bitsize", 128))

  elif isinstance(ty, PointerType):
    if isinstance(ty.elem, VoidType):
      return (("pointer", None),)

    inner_bty = type_to_btype(ty.elem)
    if inner_bty is not None:
      return (("pointer", inner_bty),)

  else:
    print(f"type_to_type: failed to convert {ty}")
  
  return None


