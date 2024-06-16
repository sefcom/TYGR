from typing import *

import torch
from torch import Tensor

from .types import *

class TypeSet:
  def __init__(self):
    self.types = []

  def num_types(self) -> int:
    return len(self.types)

  def index_of_type(self, ty: Type) -> int:
    return self.types.index(ty)

  def type_to_tensor(self, ty: Type) -> Tensor:
    ty_id = self.index_of_type(ty)
    return torch.Tensor([1 if ty_id == i else 0 for i in range(self.num_types())])

  def tensor_to_index(self, ts: Tensor) -> int:
    return ts.argmax()

  def tensor_to_type(self, ts: Tensor) -> Type:
    return self.index_to_type(ts.argmax())

  def index_to_type(self, index: int) -> Type:
    return self.types[index]

  def tensor_to_topk_types(self, ts: Tensor, k: int) -> List[Type]:
    topk_indices = torch.topk(ts, k).indices
    return [self.index_to_type(i) for i in topk_indices]

class StandardTypeSet(TypeSet):
  def __init__(self):
    super().__init__()

    self.ty = Type()
    self.void_ptr = PointerType(VoidType())
    self.void_ptr_ptr = PointerType(PointerType(VoidType()))
    default_types = [
      self.ty,
      self.void_ptr,
      self.void_ptr_ptr,
    ]
    self.void_array = ArrayType(VoidType())
    base_types = [
      self.void_array,
      StructType(),
      UnionType(),
      EnumType(),
      BooleanType(),
      CharType(),
      I1Type(),
      I8Type(),
      I16Type(),
      I32Type(),
      I64Type(),
      I128Type(),
      U1Type(),
      U8Type(),
      U16Type(),
      U32Type(),
      U64Type(),
      U128Type(),
      F32Type(),
      F64Type(),
      F128Type(),
    ]
    single_ptr_types = [PointerType(ty) for ty in base_types]
    all_types = default_types + base_types + single_ptr_types 

    self.types.extend(all_types)

  def index_of_type(self, ty: Type) -> int:
    if ty in self.types:
      return self.types.index(ty)
    elif isinstance(ty, ArrayType):
      return self.types.index(self.void_array)
    elif isinstance(ty, PointerType) and isinstance(ty.elem, PointerType):
      return self.types.index(self.void_ptr_ptr)
    elif isinstance(ty, PointerType):
      return self.types.index(self.void_ptr)
    else:
      return self.types.index(self.ty)

class ReducedStdTypeSet(TypeSet):
  def __init__(self):
    super().__init__()

    self.ty = Type()
    self.void_ptr = PointerType(VoidType())
    self.void_ptr_ptr = PointerType(PointerType(VoidType()))
    self.void_array = ArrayType(VoidType())
    self.void_struct_array = StructMember(ArrayType(VoidType()))

    default_types = [
      self.ty,
      self.void_ptr,
      self.void_ptr_ptr,
      self.void_array,
      self.void_struct_array,
    ]

    base_types = [
      StructType(),
      UnionType(),
      EnumType(),
      BooleanType(),
      CharType(),
      I1Type(),
      I8Type(),
      I16Type(),
      I32Type(),
      I64Type(),
      I128Type(),
      U1Type(),
      U8Type(),
      U16Type(),
      U32Type(),
      U64Type(),
      U128Type(),
      F32Type(),
      F64Type(),
      F128Type(),
    ]
    single_ptr_types = [PointerType(ty) for ty in [
      StructType(),
      UnionType(), 
      EnumType(), 
      CharType(),
      I16Type(), 
      I32Type(), 
      I64Type(), 
      I128Type(),
      U16Type(), 
      U32Type(), 
      U64Type(), 
      U128Type(),
      F32Type(), 
      F64Type(), 
      F128Type(),
    ]]

    struct_member = [StructMember(ty) for ty in (base_types + single_ptr_types)]

    
    all_types = default_types + base_types + single_ptr_types + struct_member

    self.types.extend(all_types)

  def index_of_type(self, ty: Type) -> int:
    if ty in self.types:
      return self.types.index(ty)
    elif isinstance(ty, ArrayType):
      return self.types.index(self.void_array)
    elif isinstance(ty, PointerType) and isinstance(ty.elem, PointerType):
      return self.types.index(self.void_ptr_ptr)
    elif isinstance(ty, PointerType):
      return self.types.index(self.void_ptr)
    elif isinstance(ty,StructMember) and isinstance(ty.elem, ArrayType):
      return self.types.index(self.void_struct_array)
    else:
      return self.types.index(self.ty)
    
class DebinTypeSet(TypeSet):
  def __init__(self):
    super().__init__()

    self.ty = Type()
    self.ptr = PointerType(VoidType())
    self.arr = ArrayType(VoidType())
    default_types = [
      self.ty,
      self.ptr,
      self.arr,
    ]
    base_types = [
      StructType(),
      UnionType(),
      EnumType(),
      BooleanType(),
      CharType(),
      I1Type(),
      I8Type(),
      I16Type(),
      I32Type(),
      I64Type(),
      I128Type(),
      U1Type(),
      U8Type(),
      U16Type(),
      U32Type(),
      U64Type(),
      U128Type(),
    ]
    all_types = default_types + base_types

    self.types.extend(all_types)

  def index_of_type(self, ty: Type) -> int:
    if ty in self.types:
      return self.types.index(ty)
    elif isinstance(ty, ArrayType):
      return self.types.index(self.arr)
    elif isinstance(ty, PointerType):
      return self.types.index(self.ptr)
    else:
      return self.types.index(self.ty)

DEFAULT_TYPE_SET_CLASS = "rstd"

type_set_classes = {
  "std": StandardTypeSet,
  "rstd": ReducedStdTypeSet,
  "debin": DebinTypeSet,
}

def get_type_set(name: str) -> TypeSet:
  return type_set_classes[name]()
