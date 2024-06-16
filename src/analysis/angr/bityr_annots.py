import angr
from angr import *

import claripy as cp
import claripy.ast as ast

# A custom base class with some properties already declared
class BityrAnnotation(cp.Annotation):
  def __init__(self, addr, loc, data):
    assert isinstance(addr, int)
    assert isinstance(loc, ast.BV)
    assert isinstance(data, ast.BV)
    # assert isinstance(baseAddr, ast.BV)
    # assert isinstance(index, ast.BV)

    self.addr = addr
    self.loc = loc
    self.data = data
    # self.baseAddr = baseAddr
    # self.index = index

  @property
  def eliminatable(self):
    return True

  @property
  def relocatable(self):
    return False

  # This is supposed to be not called because relocatable is False
  def relocate(self, src, dst):
    raise NotImplementedError


# Register reads
class RegisterReadAnnotation(BityrAnnotation):
  def __init__(self, addr, reg_offset, data):
    super().__init__(addr, reg_offset, data)

  def __repr__(self):
    return f"RegisterReadAnnotation({hex(self.addr)}, {self.loc}, {self.data})"


# Memory reads
class MemoryReadAnnotation(BityrAnnotation):
  def __init__(self, addr, mem_addr, data):
    super().__init__(addr, mem_addr, data)

  def __repr__(self):
    return f"MemoryReadAnnotation({hex(self.addr)}, {self.loc}, {self.data})"


# Register writes
class RegisterWriteAnnotation(BityrAnnotation):
  def __init__(self, addr, reg_offset, data):
    super().__init__(addr, reg_offset, data)

  def __repr__(self):
    return f"RegisterWriteAnnotation({hex(self.addr)}, {self.loc}, {self.data})"


# Memory writes
class MemoryWriteAnnotation(BityrAnnotation):
  # add baseAddr & index
  def __init__(self, addr, mem_addr, data, baseAddr = None, index = None):
    self.baseAddr = baseAddr
    self.index = index
    super().__init__(addr, mem_addr, data)

  def __repr__(self):
    return f"MemoryWriteAnnotation({hex(self.addr)}, {self.loc}, {self.data})"


# Helper functions

def is_register_annotation(annot):
  return isinstance(annot, RegisterReadAnnotation) or isinstance(annot, RegisterWriteAnnotation)

def is_memory_annotation(annot):
  return isinstance(annot, MemoryReadAnnotation) or isinstance(annot, MemoryWriteAnnotation)


