from typing import *

from elftools.elf.elffile import ELFFile
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.die import *

def get_dwarf_info(input_file_name: str) -> DWARFInfo:
  with open(input_file_name, "rb") as file:
    elf = ELFFile(file)
    if not elf.has_dwarf_info():
      raise Exception(f"No dwarf info in file {input_file_name}")
    return elf.get_dwarf_info()

def iter_die(die: DIE) -> Iterator[DIE]:
  yield die
  for child in die.iter_children():
    for child_die in iter_die(child):
      yield child_die

def iter_subprogram_die(die: DIE) -> Iterator[DIE]:
  for child in iter_die(die):
    if child.tag == "DW_TAG_subprogram":
      yield child

def iter_type_die(die: DIE) -> Iterator[DIE]:
  for child in iter_die(die):
    if "type" in child.tag:
      yield child

def get_die_name(die: DIE) -> Optional[str]:
  if "DW_AT_name" in die.attributes:
    return die.attributes["DW_AT_name"].value
  else:
    return None

def get_die_attribute(die: DIE, attr: str) -> Optional[Any]:
  if attr in die.attributes:
    return die.attributes[attr].value
  else:
    return None

