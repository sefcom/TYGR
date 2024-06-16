# Usage: /path/to/ida64 -c -A -S"/path/to/ida_dump_btypes.py /path/to/dump.pkl" /path/to/binary

from ida_funcs import *
from ida_nalt import *
from ida_hexrays import *
from idc import *
from ida_kernwin import *

import copy
import pickle

# In accordance with DWARF4 naming conventions
def get_base_name(tif):
  if tif.is_int(): return "signed"
  elif tif.is_uint(): return "unsigned"
  elif tif.is_char(): return "signed_char"
  elif tif.is_uchar(): return "unsigned_char"
  elif tif.is_bool(): return "boolean"
  elif tif.is_float(): return "float"
  elif tif.is_double(): return "double"
  else: return None

# Convers an IDA tinfo_t into a limited nested representation
def tinfo_to_btype(tif):

  # Create an initial nested dictionary which corresponds to the type
  name = tif.get_type_name()

  btype = None

  # The following conditions should be mutually exclusive
  base_name = get_base_name(tif)
  if base_name is not None:
    bitsize = tif.get_size() * 8
    btype = (("base", base_name), ("bitsize", bitsize))

  # If this is a pointer, do a recursion
  elif tif.is_ptr():
    pointed_tif = tif.get_pointed_object()

    # Voids can only be pointed to; they are not standalone
    if pointed_tif.is_void():
      # This adapts to the DWARF convention
      btype = (("pointer", None),)

    else:
      pointed_btype = tinfo_to_btype(pointed_tif)
      if pointed_btype is None: return None
      btype = (("pointer", pointed_btype),)

  # Similarly if this is an array, do a recursion
  elif tif.is_array():
    elem_tif = tif.get_array_element()
    elem_btype = tinfo_to_btype(elem_tif)
    if elem_btype is None: return None

    # Fun fcat: in IDA the array size accounts for the length,
    # but in DWARF it's only the elem type.
    # This was an issue in earlier iterations of this script
    btype = (("array", elem_btype),)

  elif tif.is_union():
    btype = (("union", name),)

  elif tif.is_enum():
    btype = (("enum", name),)

  elif tif.is_struct():
    btype = (("struct", name),)

  # We can still potentially do things in a bit
  elif tif.is_volatile() or tif.is_const():
    pass

  # We can't compact things, so pass
  else:
    btype = None

  # Now annotate things with const and volatile flags
  if tif.is_volatile() and btype is not None:
    btype = (("volatile", copy.deepcopy(btype)),)

  if tif.is_const() and btype is not None:
    btype = (("const", copy.deepcopy(btype)),)

  return btype


# Get the stack variables associated with a particular ea
def get_ea_stack_vars(ea):
  vars = []
  try:
    decomp = decompile(ea)
    if decomp is None: return []

    for var in decomp.get_lvars():
      if var.is_stk_var(): vars.append(var)
  except Exception as e:
    msg(f"decompilation issue at {get_input_file_path()}\n")
    msg(f"exception:\n{e}\n")

  return vars

# Wait for IDA's automatic analysis to finish and initialize hexrays
idaapi.auto_wait()
init_status = init_hexrays_plugin()
msg("\nstart!\n\n")

# Maps ea -> { offset -> tdict }
stack_var_dict = {}

for func_ea in Functions():
  func_name = get_func_name(func_ea)
  frame_size = get_frame_size(func_ea)
  msg(f"{hex(func_ea)} @ {func_name} | size={frame_size}\n")

  # Map offset -> tdict
  vars = get_ea_stack_vars(func_ea)
  btype_dict = dict()
  for var in vars:
    cfa_offset = var.get_stkoff() - frame_size
    btype = tinfo_to_btype(var.tif)
    key = ("cfa", cfa_offset)
    val = {(func_ea, None, btype)}
    btype_dict[key] = val

    msg(f"var={var.name}, off={cfa_offset}, tinfo={var.tif}\n")
    msg(f"btype={btype}\n")

  stack_var_dict[func_ea] = copy.deepcopy(btype_dict)
  msg("\n")

msg("done!\n\n")

# Dump, save results, and close IDA
bickle_filepath = ARGV[1]

with open(bickle_filepath, "wb") as f:
  pickle.dump(stack_var_dict, f)
  f.close()

log_filepath = ARGV[2] if len(ARGV) > 2 else "/dev/null"
msg_save(log_filepath)

exit(0)

