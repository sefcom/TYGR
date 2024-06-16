from typing import Optional

class Type:
  def __init__(self):
    pass

  def __str__(self):
    return "type"

  def __repr__(self):
    return str(self)

  def size(self) -> Optional[int]:
    raise Exception("Not implemented")

  def __hash__(self):
    return hash(str(self))

  def __eq__(self, other):
    return str(self) == str(other)

class VoidType:
  def __str__(self):
    return "void"

  def size(self) -> Optional[int]:
    return None

class StructType(Type):
  def __init__(self):
    super().__init__()

  def __str__(self):
    return "struct"

  def size(self) -> Optional[int]:
    return None

class ArrayType(Type):
  def __init__(self, elem: Type):
    super().__init__()
    self.elem = elem

  def __str__(self):
    return f"array<{str(self.elem)}>"

  def elem_type(self) -> Type:
    return self.elem

  def size(self) -> Optional[int]:
    return None

class UnionType(Type):
  def __init__(self):
    super().__init__()

  def __str__(self):
    return "union"

  def size(self) -> Optional[int]:
    return None

class EnumType(Type):
  def __init__(self):
    super().__init__()

  def __str__(self):
    return "enum"

  def size(self) -> Optional[int]:
    return None

class BooleanType(Type):
  def size(self) -> Optional[int]:
    return 8

  def __str__(self):
    return "bool"

class I1Type(Type):
  def size(self) -> Optional[int]:
    return 1

  def __str__(self):
    return "i1"

class I8Type(Type):
  def size(self) -> Optional[int]:
    return 8

  def __str__(self):
    return "i8"

class I16Type(Type):
  def size(self) -> Optional[int]:
    return 16

  def __str__(self):
    return "i16"

class I32Type(Type):
  def size(self) -> Optional[int]:
    return 32

  def __str__(self):
    return "i32"

class I64Type(Type):
  def size(self) -> Optional[int]:
    return 64

  def __str__(self):
    return "i64"

class I128Type(Type):
  def size(self) -> Optional[int]:
    return 128

  def __str__(self):
    return "i128"

class U1Type(Type):
  def size(self) -> Optional[int]:
    return 1

  def __str__(self):
    return "u1"

class U8Type(Type):
  def size(self) -> Optional[int]:
    return 8

  def __str__(self):
    return "u8"

class U16Type(Type):
  def size(self) -> Optional[int]:
    return 16

  def __str__(self):
    return "u16"

class U32Type(Type):
  def size(self) -> Optional[int]:
    return 32

  def __str__(self):
    return "u32"

class U64Type(Type):
  def size(self) -> Optional[int]:
    return 64

  def __str__(self):
    return "u64"

class U128Type(Type):
  def size(self) -> Optional[int]:
    return 128

  def __str__(self):
    return "u128"

class CharType(Type):
  def size(self) -> Optional[int]:
    return 8

  def __str__(self):
    return "char"

class F32Type(Type):
  def size(self) -> Optional[int]:
    return 32

  def __str__(self):
    return "f32"

class F64Type(Type):
  def size(self) -> Optional[int]:
    return 64

  def __str__(self):
    return "f64"

class F128Type(Type):
  def size(self) -> Optional[int]:
    return 128

  def __str__(self):
    return "f128"

class PointerType(Type):
  def __init__(self, elem: Type):
    super().__init__()
    if elem is None:
      self.elem = VoidType()
    else:
      self.elem = elem

  def __str__(self):
    return f"{str(self.elem)}*"

  def elem_type(self) -> Type:
    return self.elem

  def size(self) -> Optional[int]:
    return None

class StructMember(Type):
  def __init__(self, elem: Type):
    super().__init__()
    self.elem = elem

  def __str__(self):
    return f"struct_{str(self.elem)}"

  def elem_type(self) -> Type:
    return self.elem

  def size(self) -> Optional[int]:
    return None