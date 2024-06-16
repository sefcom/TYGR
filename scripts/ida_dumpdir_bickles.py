'''
    python3 /path/to/bityr/scripts/ida_dumpdir_bickles.py \
        -idir /path/to/bindir/ \
        -odir /path/to/outdir/
'''

import argparse
import sys
import os

from elftools.elf.elffile import ELFFile

def parser():
  parser = argparse.ArgumentParser(description="Dump all the contents of directory using IDA")
  parser.add_argument("--flexlm_dir", type=str, default="/home/taro/foo/flexlm")
  parser.add_argument("--flexlm", type=str, default="/home/taro/foo/flexlm/run.sh")
  parser.add_argument("--ida", type=str, default="/home/taro/foo/idapro-7.6/ida64")
  parser.add_argument("--idapy_script", type=str, default="/home/taro/foo/bityr/scripts/ida_dump_bickle.py")
  parser.add_argument("-idir", "--input_dir", type=str, required=True)
  parser.add_argument("-odir", "--output_dir", type=str, required=True)
  return parser

# Setup the IDA license server
def setup_flexlm(args):
  cwd = os.getcwd()
  os.chdir(args.flexlm_dir)
  cmd = f"bash {args.flexlm}"
  print(cmd)
  err = os.system(cmd)
  os.chdir(cwd)
  if not err:
    print("Successfully started flexlm")
    return True
  else:
    print("Failed to start flexlm")
    return False

# We only care about ELF files
def is_elf(path):
  if os.path.isfile(path):
    with open(path, "rb") as f:
      try:
        elffile = ELFFile(f)
        return True
      except:
        return False
  return False

# Run generation process
def generate_ida_bickles(args):
  filenames = os.listdir(args.input_dir)
  elfnames = list(filter(lambda f : is_elf(os.path.join(args.input_dir, f)), filenames))
  count = len(elfnames)
  for i, name in enumerate(elfnames):
    in_path = os.path.join(args.input_dir, name)
    out_path = os.path.join(args.output_dir, name + ".bkl")
    cmd = f'{args.ida} -c -A -S"{args.idapy_script} {out_path}" {in_path}'
    print(f"[{i+1}/{count}]: {cmd}")
    err = os.system(f"timeout 2m {cmd}")
    if os.path.exists(out_path):
      pass
    else:
      print("Failed!")


# Run the stuff
args = parser().parse_args()
flex_ok = setup_flexlm(args)
if not flex_ok:
  exit(1)

print("")
generate_ida_bickles(args)

