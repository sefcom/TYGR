
# Ghidra/RuntimeScripts/Linux/support/analyzeHeadless ~/foo/test/ghidra-test Project1 -import ~/foo/test/cfiles/file-ext.x64-O0 -postscript ~/foo/bar.py -deleteProject

from ghidra.app.decompiler import DecompileOptions, DecompInterface
from ghidra.program.model.data import BooleanDataType
from ghidra.program.model.data import UnsignedIntegerDataType
from ghidra.program.model.data import IntegerDataType
from ghidra.program.model.data import UnsignedLongDataType
from ghidra.program.model.data import LongDataType
from ghidra.program.model.data import CharDataType
from ghidra.program.model.data import UnsignedShortDataType
from ghidra.program.model.data import ShortDataType
from ghidra.program.model.data import UnsignedCharDataType
from ghidra.program.model.data import SignedCharDataType
from ghidra.program.model.data import FloatDataType
from ghidra.program.model.data import DoubleDataType
from ghidra.program.model.data import PointerDataType
from ghidra.program.model.data import StructureDataType
from ghidra.program.model.data import ArrayDataType
from ghidra.program.model.data import EnumDataType
from ghidra.program.model.data import VoidDataType
from ghidra.program.model.data import TypedefDataType

import pickle

def setup_ifc():
  options = DecompileOptions()
  ifc = DecompInterface()
  ifc.openProgram(currentProgram)
  return ifc

def datatype_to_btype(datatype):
  if datatype is None:
    return None
  elif isinstance(datatype, BooleanDataType):
    return { "base" : "boolean", "bitsize" : 8 }
  elif (isinstance(datatype, UnsignedIntegerDataType) or
        isinstance(datatype, UnsignedLongDataType) or
        isinstance(datatype, UnsignedShortDataType)):
    return { "base" : "unsigned", "bitsize" : datatype.getLength() * 8 }
  elif (isinstance(datatype, IntegerDataType) or
        isinstance(datatype, LongDataType) or
        isinstance(datatype, ShortDataType)):
    return { "base" : "signed", "bitsize" : datatype.getLength() * 8 }
    return { "base" : "signed", "bitsize" : datatype.getLength() * 8 }
    return { "base" : "signed", "bitsize" : datatype.getLength() * 8 }
  elif isinstance(datatype, UnsignedCharDataType):
    return { "base" : "unsigned_char", "bitsize" : 8 }
  elif isinstance(datatype, CharDataType) or isinstance(datatype, SignedCharDataType):
    return { "base" : "signed_char", "bitsize" : 8 }
  elif isinstance(datatype, FloatDataType) or isinstance(datatype, DoubleDataType):
    return { "base" : "float", "bitsize" : datatype.getLength() * 8 }
  elif isinstance(datatype, PointerDataType):
    inner_datatype = datatype.getDataType()
    inner_bty = datatype_to_btype(inner_datatype)
    if isinstance(inner_datatype, VoidDataType):
      return { "pointer" : None }
    elif isinstance(inner_bty, dict):
      return { "pointer" : inner_bty }
    else:
      # print("skipping due to inner type {}".format(datatype.getDataType()))
      return None
  elif isinstance(datatype, StructureDataType):
    return { "struct" : "STRUCT" }
  elif isinstance(datatype, ArrayDataType):
    inner_bty = datatype_to_btype(datatype.getDataType())
    if isinstance(inner_bty, dict):
      return { "array" : inner_bty }
    else:
      return None
  elif isinstance(datatype, EnumDataType):
    return { "enum" : "ENUM" }
  elif isinstance(datatype, TypedefDataType):
    inner_ty = datatype.getDataType()
    return datatype_to_btype(inner_ty)
  else:
    # print("skipping type {}, wihch has type {}".format(datatype, type(datatype)))
    return None


def symbol_info_str(symbol):
  txt = "name: {}, type: {}, pc: {}, size: {}, param: {}, store: {}".format(
    symbol.name, symbol.dataType, symbol.getPCAddress(), symbol.size, symbol.parameter, symbol.storage)
  return txt


def get_function_info(ifc, func):
  decomp = ifc.decompileFunction(func, 60, None)
  high_func = decomp.getHighFunction()
  lsm = high_func.getLocalSymbolMap()
  symbols = list(lsm.getSymbols())
  loc_to_type = dict()
  for i, symbol in enumerate(symbols):
    store = symbol.getStorage()
    ty = symbol.getDataType()
    ty_str = ty.toString().encode("ascii", "ignore")
    bty = datatype_to_btype(ty)

    if bty is None:
      print("store: {} \t skipping type {} \t of btype {}".format(store, ty, bty))
      print("")
      continue

    # This is a stack variable with no
    if symbol.isParameter():
      if store.isStackStorage():
        loc = ("param", "stack", store.getStackOffset())
        loc_to_type[loc] = bty
      elif store.isRegisterStorage():
        reg_name = store.getRegister().getName().encode("ascii", "ignore")
        loc = ("param", "reg", reg_name)
        loc_to_type[loc] = bty
      else:
        print("type is {} | btype is {}".format(ty, bty))
        print("unable to handle parameter:\n\t{}".format(symbol_info_str(symbol)))
    else:
      if store.isStackStorage():
        loc = ("var", "stack", store.getStackOffset())
        loc_to_type[loc] = bty
      elif store.isRegisterStorage():
        reg_name = store.getRegister().getName().encode("ascii", "ignore")
        pc = symbol.getPCAddress().getOffset() - currentProgram.getImageBase().getOffset()
        loc = ("var", "reg", reg_name, pc)
        loc_to_type[loc] = bty
      else:
        print("type is {} | btype is {}".format(ty, bty))
        print("unable to handle non-parameter:\n\t{}".format(symbol_info_str(symbol)))
    
    return loc_to_type



args = getScriptArgs()

state = getState()
project = state.getProject()
program = state.getCurrentProgram()
ifc = setup_ifc()
func_manager = currentProgram.getFunctionManager()
funcs = list(func_manager.getFunctions(True))
addr_to_tydict = dict()

for func in funcs:
  func_addr = func.getEntryPoint().getOffset() - func.program.getImageBase().getOffset()
  print("proecssing addr {}".format(hex(func_addr)))
  loc_to_ty = get_function_info(ifc, func)
  addr_to_tydict[func_addr] = loc_to_ty
  print("\n-----\n\n\n")


if len(args) == 0:
  print("Pass in dump path")
  exit(1)

dump_path = args[0]

with open(dump_path, "wb") as f:
  pickle.dump(addr_to_tydict, f)
  f.close()


print("test A {}".format(program.getDomainFile()))
print("test B {}".format(program.getExecutablePath()))
print("test C {}".format(program.getName()))

print("dumping args:, which has type {}".format(type(args)))

for arg in args:
  print(arg)

print("accessting index")
print("args[0]: {}".format(args[0]))

