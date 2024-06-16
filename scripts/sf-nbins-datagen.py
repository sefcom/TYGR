import argparse
import sys
import os
import json
import pickle

from elftools.elf.elffile import *

command = "bityr"

initial_ign_fns = [
  "xpalloc",
  "z85_encode",
  "try_tempname_len",
]

def parser():
  parser = argparse.ArgumentParser(description='Data Generation')
  setup_parser(parser)
  return parser

def setup_parser(parser):
  parser.add_argument("directory", type=str)
  parser.add_argument("output_directory", type=str)
  parser.add_argument("-o", "--overwrite", action="store_true")
  parser.add_argument("--merge-file", type=str, default="merged.pkl")
  parser.add_argument("--split-dir", type=str, default="splits")
  parser.add_argument("--train", type=str, default="train.pkl")
  parser.add_argument("--validation", type=str, default="validation.pkl")
  parser.add_argument("--test", type=str, default="test.pkl")
  parser.add_argument("--method", type=str, default="glow")
  parser.add_argument("--exit-when-failed", action="store_true")
  parser.add_argument("--ignore-functions-file", type=str, default=".sf_nbins_ign_fns.json")
  parser.add_argument("-v", "--verbose", action="store_true")

def collect_files(directory):
  for root, _, files in os.walk(directory):
    for file in files:
      yield (root, file)

def main(args):
  processed = []
  ign_fns = [ign_fn for ign_fn in initial_ign_fns]
  json.dump(ign_fns, open(args.ignore_functions_file, "w"))

  executables = list(collect_files(args.directory))

  # Process all the files
  num_exes = len(executables)
  for count, (directory, filename) in enumerate(executables):
    output_name = f"{filename}.pkl"
    item_path = os.path.join(directory, filename)
    output_path = os.path.join(args.output_directory, output_name)

    print(f"{count + 1}/{num_exes} Processing {filename}... ", end="")
    sys.stdout.flush()

    if os.path.exists(output_path) and not args.overwrite:
      print("Generated. Skipping")
      generated_file_name = output_path
    else:

      try:
        with open(item_path, "rb") as f:
          elffile = ELFFile(f)
          if not elffile.has_dwarf_info():
            print(f"File {filename} has no dwarf info, Skipping")
            continue
      except Exception as ex:
        print(f"???? {ex}")
        continue

      cmd = f"time \
              ./{command} datagen {item_path} {output_path} -m {args.method} \
              --ignore-functions {args.ignore_functions_file} \
              --parallel"

      if args.verbose:
        cmd += " -v"

      err = os.system(cmd)
      if not err:
        print("Success")
        generated_file_name = output_path
      else:
        print("Failed")
        generated_file_name = None
        if args.exit_when_failed:
          return

    # if generated_file_name is not None:
    if os.path.exists(output_path):
      processed.append(generated_file_name)

    print(f"processed count: {len(processed)}")

  # Merge all of them
  merge_path = os.path.join(args.output_directory, args.merge_file)
  datasets = " ".join(processed)
  os.system(f"./{command} datamerge {datasets} -o {merge_path}")

  # Create split
  split_path = os.path.join(args.output_directory, "splits")
  train_path = os.path.join(split_path, args.train)
  validation_path = os.path.join(split_path, args.validation)
  test_path = os.path.join(split_path, args.test)
  os.system(f"mkdir -p {split_path}")
  os.system(f"./{command} datasplit {merge_path} \
              --train {train_path} \
              --validation {validation_path} \
              --test {test_path}")

if __name__ == "__main__":
  args = parser().parse_args()
  main(args)
