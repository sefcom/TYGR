# Constant folding for the pyvex IR
from typing import *

from pyvex import *
from pyvex.expr import *
from pyvex.const import *
from pyvex.stmt import *
from archinfo.arch import Arch

def str_to_2s_comp(hex_str : str, bitsize : int):
  hex_str = hex_str.lower()
  x = int(hex_str, 16)
  if len(hex_str) > 2 and (hex_str[:3] == "0xf" or hex_str[:1] == "f"):
    return x - (1 << bitsize)
  else:
    return x

def num_to_hex_str(num : int, bitsize) -> str:
  if bitsize == 8: return ("%02x" % num)
  elif bitsize == 16: return ("%04x" % num)
  elif bitsize == 32: return ("%08x" % num)
  elif bitsize == 64: return ("%016x" % num)
  else: return None

def pyvex_const_to_int(con : int, bitsize : int) -> int:
  return str_to_2s_comp(num_to_hex_str(con, bitsize), bitsize)

def is_iop_add(op : str) -> bool:
  return op.startswith("Iop_Add")

def is_iop_sub(op : str) -> bool:
  return op.startswith("Iop_Sub")

def get_iop_bitsize(op : str) -> int:
  if op.endswith("16"): return 16
  elif op.endswith("32"): return 32
  elif op.endswith("64"): return 64
  else: return None

def create_iop_add(bitsize : int) -> str:
  return "Iop_Add" + str(bitsize)

def create_iop_sub(bitsize : int) -> str:
  return "Iop_Sub" + str(bitsize)

def get_irconst(bitsize : int) -> IRConst:
  if bitsize == 1: return U1
  elif bitsize == 8: return U8
  elif bitsize == 16: return U16
  elif bitsize == 32: return U32
  elif bitsize == 64: return U64
  else: return None

def get_reg_offset(expr: IRExpr, arch: Arch) -> Tuple[str, int]:
  if isinstance(expr, Binop):
    get_expr = expr.args[0]
    const_arg = expr.args[1]

    if isinstance(get_expr, Get) and isinstance(const_arg, Const):
      reg_name = arch.translate_register_name(get_expr.offset)
      if reg_name:
        bitsize = const_arg.con.size
        const_val = pyvex_const_to_int(const_arg.con.value, bitsize)
        if is_iop_sub(expr.op):
          return (reg_name, const_val * -1)
        if is_iop_add(expr.op):
          return (reg_name, const_val)

  if isinstance(expr, Get):
    reg_name = arch.translate_register_name(expr.offset)
    return (reg_name, 0)

# Determine if an expr is a value.
# This is the case if it is one of the following forms:
#   ...
def is_value(expr: IRExpr) -> bool:
  if expr == None:
    return True
  if isinstance(expr, Const):
    return True
  elif isinstance(expr, Get):
    return True

  # Things of form Add(rsp, 0x####)
  # Things of form Add(0x####, rsp) first have args flipped
  elif isinstance(expr, Binop):
    if is_iop_add(expr.op) or is_iop_sub(expr.op):
      args = expr.args
      return (isinstance(args[0], Get)) and is_value(args[1])
    else:
      False
  else:
    return False

# Reduce an IRExpr to a more compact "normal" form
def reduce_expr(expr: IRExpr, show_warning=False) -> IRExpr:
  if is_value(expr):
    return expr
  elif isinstance(expr, Binop):
    op = expr.op
    arg0 = expr.args[0]
    arg1 = expr.args[1]
    bitsize = get_iop_bitsize(op)

    # Binop(const1, const2) -> Const(op(const1, const2))
    if isinstance(arg0, Const) and isinstance(arg1, Const):
      if is_iop_add(op):
        return Const(get_irconst(bitsize)(arg0.con.value + arg1.con.value))
      elif is_iop_sub(op):
        return Const(get_irconst(bitsize)(arg0.con.value - arg1.con.value))
      else:
        # print("binop %s not supported: {op}")
        return expr

    # Bino(val0, expr) -> Binop(val0, val1)
    elif is_value(arg0) and not is_value(arg1):
      rhs1 = reduce_expr(arg1)
      return reduce_expr(Binop(op, [arg0, rhs1]))

    # Add(expr, Get) -> Add(Get, expr)
    elif is_iop_add(op) and isinstance(arg1, Get) and not isinstance(arg0, Get):
      return reduce_expr(Binop(op, [arg1, arg0]))

    # Cases where there is a nested operation in the first place
    elif isinstance(arg0, Binop):
      inner_op = arg0.op
      inner_arg0 = arg0.args[0]
      inner_arg1 = arg0.args[1]

      # Add(Add(expr0, expr1), expr2) -> Add(expr0, Add(expr1, expr2))
      if is_iop_add(op) and is_iop_add(inner_op):
        new_args = [inner_arg0, Binop(create_iop_add(bitsize), [inner_arg1, arg1])]
        return reduce_expr(Binop(create_iop_add(bitsize), new_args))

      # Add(Sub(expr0, expr1), expr2) -> Add(expr0, Sub(expr2, expr1)
      elif is_iop_add(op) and is_iop_sub(inner_op):
        new_args = [inner_arg0, Binop(create_iop_sub(bitsize), [arg1, inner_arg1])]
        return reduce_expr(Binop(create_iop_add(bitsize), new_args))

      # Sub(Add(expr0, expr1), expr2) -> Add(expr0, Sub(expr1, expr2))
      elif is_iop_sub(op) and is_iop_add(inner_op):
        new_args = [inner_arg0, Binop(create_iop_sub(bitsize), [inner_arg1, arg1])]
        return reduce_expr(Binop(create_iop_sub(bitsize), new_args))

      # Sub(Sub(expr0, expr1), expr2) -> Sub(expr0, Add(expr1, expr2))
      elif is_iop_sub(op) and is_iop_sub(inner_op):
        new_args = [inner_arg0, Binop(create_iop_add(bitsize), [inner_arg1, arg1])]
        return reduce_expr(Binop(create_iop_sub(bitsize), new_args))

      else:
        if show_warning:
          print("[Warning] unsupported nested combination %s" % expr.__str__())
        return expr

  else:
    if show_warning:
      print("[Warning] expr form not supported: %s" % expr.__str__())
    return expr

# If have only one GET, and the rest are either all consts or binary ops, then we ok
def reduceable(expr : IRExpr) -> bool:
  if expr is None:
    return False

  child_exprs = expr.child_expressions
  total_len = len(child_exprs)
  get_count = 0
  const_count = 0
  binop_count = 0
  for child_expr in child_exprs:
    if isinstance(child_expr, Get): get_count += 1
    if isinstance(child_expr, Const): binop_count += 1
    if isinstance(child_expr, Binop):
      if is_iop_add(child_expr.op) or is_iop_sub(child_expr.op):
        binop_count += 1

  return total_len == (get_count + const_count + binop_count) and binop_count > 0
