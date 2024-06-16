'''
  python3 /path/to/bityr/scripts/dwarf_dumpdir_bickles.py \
    -s /path/to/dump/script.py \
    -idir /path/to/input/dir/ \
    -odir /path/to/output/dir/
'''

import argparse
import os
import sys

from elftools.elf.elffile import ELFFile

def parser():
  parser = argparse.ArgumentParser(description="Dump bickle for a directory")
  parser.add_argument("-s", "--dump-script", type=str, required=True)
  parser.add_argument("-idir", "--input_dir", type=str, required=True)
  parser.add_argument("-odir", "--output_dir", type=str, required=True)
  return parser

def is_elf(path):
  if os.path.isfile(path):
    with open(path, "rb") as f:
      try:
        elffile = ELFFile(f)
        return True
      except:
        return False
  return False

def dump_bickles(args):
  filenames = os.listdir(args.input_dir)
  elfnames = list(filter(lambda f : is_elf(os.path.join(args.input_dir, f)), filenames))
  count = len(elfnames)
  for i, name in enumerate(elfnames):
    in_path = os.path.join(args.input_dir, name)
    out_path = os.path.join(args.output_dir, name + ".bkl")
    cmd = f"python3 {args.dump_script} -i {in_path} -o {out_path}"
    print(f"[{i+1}/{count}]: {cmd}")
    err = os.system(cmd)
    if os.path.exists(out_path):
      pass
    else:
      print("Failed!")

args = parser().parse_args()
dump_bickles(args)

